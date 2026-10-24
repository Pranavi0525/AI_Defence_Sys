"""Serving-layer tests: feature equality with v1, numpy==sklearn novelty scores, history-only graph, no train/serve skew, drift."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_features_match_v1_extractor_on_real_traces():
    import blue_team_pipeline as btp
    from v2.serving.core_features import extract_features_core
    corpus = json.loads((ROOT / "reports" / "app_corpus_raw.json").read_text(encoding="utf-8"))
    changed = {"failed_transaction_count", "failed_then_completed", "amount_change_after_failure", "trace_id"}
    for item in corpus[:40]:
        rec = item["observable_trace"] if "observable_trace" in item else item
        rec = dict(rec, trace_id=rec.get("trace_id", "t"))
        a, b = btp.extract_features(rec), extract_features_core(rec)
        for k in a:
            if k not in changed:
                assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k


def _toy(cust, dev, ts):
    ev = [{"event_type": "SESSION_LOGIN", "timestamp": ts, "customer_id": cust, "device_id": dev}]
    return {"trace_id": f"t_{cust}", "customer_id": cust, "events": ev, "observation_window": [ts, ts]}


def test_graph_features_use_only_history():
    from v2.features import graph_features
    recs = [_toy("a", "d1", "2025-01-01T00:00:00"), _toy("b", "d1", "2025-01-02T00:00:00"), _toy("c", "d1", "2025-01-03T00:00:00")]
    g = graph_features(recs).set_index("trace_id")
    assert [g.loc[f"t_{c}", "g_device_shared_customers"] for c in "abc"] == [0, 1, 2]     # the first trace sees nobody
    assert [g.loc[f"t_{c}", "g_component_size"] for c in "abc"] == [1, 2, 3]
    shuffled = graph_features([recs[2], recs[0], recs[1]]).set_index("trace_id")          # input order must not matter
    assert shuffled.loc["t_a", "g_device_shared_customers"] == 0


def test_entity_store_json_roundtrip_preserves_features():
    from v2.serving.store import EntityStore
    st = EntityStore()
    for r in [_toy("a", "d1", "2025-01-01T00:00:00"), _toy("b", "d1", "2025-01-02T00:00:00")]:
        st.ingest(r)
    st2 = EntityStore.from_json(st.to_json())
    probe = _toy("c", "d1", "2025-01-03T00:00:00")
    assert st.features(probe) == st2.features(probe)


def test_drift_detects_shift_and_ignores_noise():
    from v2.serving import drift
    rng = np.random.default_rng(0)
    ref = drift.build_reference(rng.normal(0, 1, (2000, 1)), ["x"])
    same = drift.psi(ref, rng.normal(0, 1, (500, 1)), ["x"])["x"]
    shifted = drift.psi(ref, rng.normal(1.5, 1, (500, 1)), ["x"])["x"]
    assert same < 0.10 and shifted > 0.25
    mon = drift.DriftMonitor(ref, ["x"])
    assert mon.report()["status"] == "insufficient_data"
    for v in rng.normal(1.5, 1, (80, 1)):
        mon.observe(v)
    assert mon.report()["status"] == "drift"


@pytest.fixture(scope="module")
def small_bundle(tmp_path_factory):
    from v2.features import build_world_b
    from v2.phases import FEATS_B, NOV_B
    from v2.serving.bundle import fit_bundle
    dfB, recs = build_world_b(return_records=True, n_legit=260, n_per_family=30, stealth=0.5, onboarding_share=0.2, young_share=0.12)
    out = tmp_path_factory.mktemp("bundle")
    b = fit_bundle(dfB, recs, FEATS_B, NOV_B, out)
    return dfB, recs, b, out, FEATS_B, NOV_B


def test_numpy_novelty_equals_sklearn(small_bundle):
    from v2.serving.numpy_models import NumpyNovelty, export_novelty
    dfB, _, b, _, FEATS_B, NOV_B = small_bundle
    ref = b["novelty"].score(dfB[NOV_B].values)
    mine = NumpyNovelty(export_novelty(b["novelty"])).score(dfB[NOV_B].values)
    for k in ref:
        assert np.allclose(ref[k], mine[k], atol=1e-9), k


def test_served_scores_equal_training_side_scores(small_bundle):
    from v2.fusion import _logit
    from v2.serving.scorer import V2Scorer
    dfB, _, b, out, FEATS_B, NOV_B = small_bundle
    sc = V2Scorer(out)
    p = b["xgb"].predict_proba(dfB[FEATS_B].values)[:, 1]
    nv = b["novelty"].score(dfB[NOV_B].values)
    ref = b["stack"].predict_proba(np.c_[_logit(p), b["ecdf_if"](nv["iforest"]), b["ecdf_ae"](nv["autoencoder"])])[:, 1]
    keys = set(FEATS_B) | set(NOV_B)
    got = [sc.score_features({k: row[k] for k in keys}) for _, row in dfB.iloc[::7].iterrows()]
    assert np.allclose(ref[::7], [g["stacked_probability"] for g in got], atol=1e-6)
    dec_ref = np.where(ref[::7] >= b["t_block"], 2, np.where(ref[::7] >= b["t_review"], 1, 0))
    assert (dec_ref == np.array([g["decision_index"] for g in got])).all()


def test_no_train_serve_skew_for_graph_features(small_bundle):
    """Scoring the training traces one by one in time order through the API's scorer (empty store, commit=True)
    must reproduce exactly the graph features the model was TRAINED on."""
    from v2.serving.scorer import V2Scorer
    from v2.serving.store import EntityStore
    dfB, recs, _, out, _, _ = small_bundle
    sc = V2Scorer(out)
    sc.store = EntityStore()
    gcols = [c for c in dfB.columns if c.startswith("g_")]
    expected = dfB.set_index("trace_id")[gcols]
    for rec in sorted(recs, key=lambda r: (r["events"][0]["timestamp"], r["trace_id"])):
        out_ = sc.score(rec, commit=True)
        for c in gcols:
            assert out_["graph_features"][c] == expected.loc[rec["trace_id"], c], (rec["trace_id"], c)
