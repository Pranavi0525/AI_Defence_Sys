"""Tests for the v2 layer (generators, features, graph, fusion/policy). Run: PYTHONPATH=src pytest tests/test_v2_suite.py -v"""
import re
import numpy as np
import pytest

from v2.gen import build_v2_records, NEW_FAMILIES
from v2.features import extract_features_v2, graph_features
from v2 import fusion

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture(scope="module")
def recs():
    return build_v2_records(seed=5, n_legit=120, hard_share=0.5, n_per_family=12)


def test_deterministic():
    a = build_v2_records(seed=5, n_legit=30, n_per_family=5)
    b = build_v2_records(seed=5, n_legit=30, n_per_family=5)
    assert [r["trace_id"] for r in a] == [r["trace_id"] for r in b]


def test_labels_and_families(recs):
    fam = {r["attack_family"] for r in recs if r["fraud"] == 1}
    assert fam == set(NEW_FAMILIES)
    assert all(r["fraud"] == 0 for r in recs if r["attack_family"].startswith("legitimate"))


def test_no_id_format_leak(recs):
    """Every entity id has the same uuid format for legit and fraud, so id text cannot reveal the label."""
    for r in recs:
        for e in r["events"]:
            for k in ("device_id", "beneficiary_id", "merchant_id", "customer_id"):
                if e.get(k):
                    assert UUID.match(e[k]), (k, e[k])


def test_window_and_ordering(recs):
    for r in recs:
        ts = [e["timestamp"] for e in r["events"]]
        assert ts == sorted(ts)
        assert r["observation_window"] == [ts[0], ts[-1]]


def _min_record(statuses):
    ev = [{"event_type": "SESSION_LOGIN", "timestamp": "2025-01-01T00:00:00", "customer_id": "c", "login_attempt_count": 1, "auth_success": True}]
    for i, s in enumerate(statuses):
        ev.append({"event_type": "TRANSACTION", "timestamp": f"2025-01-01T00:0{i+1}:00", "customer_id": "c", "amount": "100.00",
                   "transaction_status": s, "channel": "mobile"})
    return {"trace_id": "t", "customer_id": "c", "events": ev, "observation_window": [ev[0]["timestamp"], ev[-1]["timestamp"]]}


def test_pending_is_not_failure():
    assert extract_features_v2(_min_record(["pending", "pending", "pending"]))["failed_transaction_count"] == 0
    assert extract_features_v2(_min_record(["failed", "completed"]))["failed_transaction_count"] == 1
    assert extract_features_v2(_min_record(["failed", "completed"]))["failed_then_completed"] == 1


def test_graph_shared_device_degree_is_history_only():
    def r(cust, dev, ts):
        ev = [{"event_type": "SESSION_LOGIN", "timestamp": ts, "customer_id": cust, "device_id": dev}]
        return {"trace_id": f"t_{cust}", "customer_id": cust, "events": ev, "observation_window": [ts, ts]}
    g = graph_features([r("a", "d1", "2025-01-01T00:00:00"), r("b", "d1", "2025-01-02T00:00:00"),
                        r("c", "d1", "2025-01-03T00:00:00"), r("d", "d2", "2025-01-04T00:00:00")]).set_index("trace_id")
    assert [g.loc[f"t_{c}", "g_device_shared_customers"] for c in "abcd"] == [0, 1, 2, 0]
    assert g.loc["t_c", "g_component_size"] == 3 and g.loc["t_d", "g_component_size"] == 1


def test_weights_reweight_to_prevalence():
    y = np.array([1] * 10 + [0] * 90)
    w = fusion.weights(y, 0.01)
    assert abs(w[y == 1].sum() / w.sum() - 0.01) < 1e-9


def test_thresholds_ordered_and_review_cap_respected():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    s = np.r_[rng.beta(8, 2, 100), rng.beta(2, 8, 900)]
    t_r, t_b = fusion.choose_thresholds(s, y, prevalence=0.05, review_cap=0.05)
    assert t_r <= t_b
    w = fusion.weights(y, 0.05)
    d = fusion.decide(s, t_r, t_b)
    assert (w * (d == 1)).sum() / w.sum() <= 0.05 + 1e-9


def test_perfect_scores_cost_zero_when_blocked():
    y = np.array([1, 1, 0, 0]); w = fusion.weights(y, 0.5)
    d = np.array([2, 2, 0, 0])
    assert fusion.expected_cost(d, y, w, 500, 0.9, 12, 25) == 0.0


def test_stealth_removes_shared_devices_for_synthetic_identity():
    from v2.gen import G, Pools, gen_synthetic_identity
    def shared_rate(stealth):
        g = G(3); recs = gen_synthetic_identity(g, Pools(g), 60, stealth)
        devs = [e["device_id"] for r in recs for e in r["events"] if e["event_type"] == "SESSION_LOGIN"]
        return 1 - len(set(devs)) / len(devs)
    assert shared_rate(0.0) > 0.3 and shared_rate(1.0) == 0.0


def test_stealth_deepfake_kyc_looks_like_genuine_applicant():
    from v2.gen import G, Pools, gen_deepfake_kyc
    g = G(4)
    plain = [e for r in gen_deepfake_kyc(g, Pools(g), 80, 0.0) for e in r["events"] if e["event_type"] == "KYC_VERIFICATION"]
    stealth = [e for r in gen_deepfake_kyc(g, Pools(g), 80, 1.0) for e in r["events"] if e["event_type"] == "KYC_VERIFICATION"]
    assert np.mean([e["liveness_score"] for e in stealth]) > np.mean([e["liveness_score"] for e in plain]) + 0.05
    assert np.mean([e["retries"] for e in stealth]) < np.mean([e["retries"] for e in plain])
