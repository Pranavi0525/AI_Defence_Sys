"""Tests for /api/v2/* (full cascade). Uses the real bundle in models/v2 (loaded by the real app lifespan)."""
from __future__ import annotations

import copy

import pytest

from v2.gen import G, Pools, gen_legit, gen_deepfake_kyc


@pytest.fixture(scope="module")
def sample_traces():
    g = G(99)
    pools = Pools(g)
    legit = gen_legit(g, pools, 3, 0.0)
    fraud = gen_deepfake_kyc(g, pools, 3, 0.0)
    return legit, fraud


def _payload(rec, commit=False):
    return {"trace": {k: rec[k] for k in ("trace_id", "customer_id", "events", "observation_window")}, "commit": commit}


class TestStatusAndCard:
    def test_status_ready(self, app_client):
        body = app_client.get("/api/v2/status").json()
        assert body["ready"] is True and body["model_version"].startswith("v2-")
        assert body["commit_enabled"] is False

    def test_readyz_exposes_v2(self, app_client):
        assert "v2_ready" in app_client.get("/readyz").json()

    def test_model_card_states_limitations(self, app_client):
        card = app_client.get("/api/v2/model-card").json()
        assert card["limitations"] and "SYNTHETIC" in card["limitations"][0]
        assert 0 < card["t_review"] <= card["t_block"] <= 1


class TestScoring:
    def test_scores_all_stages(self, app_client, sample_traces):
        legit, fraud = sample_traces
        r = app_client.post("/api/v2/score", json=_payload(fraud[0]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
        assert set(body["stages"]) >= {"supervised_xgboost", "novelty_isolation_forest", "novelty_autoencoder",
                                       "graph_history_only", "fusion_stacked"}
        assert 0 <= body["risk_score"] <= 1 and body["committed"] is False
        assert body["top_contributing_features"] and "disclaimer" in body

    def test_deepfake_scores_higher_than_routine_legit(self, app_client, sample_traces):
        legit, fraud = sample_traces
        s_l = [app_client.post("/api/v2/score", json=_payload(r)).json()["risk_score"] for r in legit]
        s_f = [app_client.post("/api/v2/score", json=_payload(r)).json()["risk_score"] for r in fraud]
        assert max(s_f) > min(s_l)

    def test_unknown_event_type_rejected(self, app_client, sample_traces):
        bad = copy.deepcopy(_payload(sample_traces[0][0]))
        bad["trace"]["events"][0]["event_type"] = "NOT_AN_EVENT"
        assert app_client.post("/api/v2/score", json=bad).status_code == 422

    def test_negative_amount_rejected(self, app_client, sample_traces):
        bad = copy.deepcopy(_payload(sample_traces[0][0]))
        for e in bad["trace"]["events"]:
            if e["event_type"] == "TRANSACTION":
                e["amount"] = "-5"
                break
        assert app_client.post("/api/v2/score", json=bad).status_code == 422

    def test_bad_timestamp_rejected(self, app_client, sample_traces):
        bad = copy.deepcopy(_payload(sample_traces[0][0]))
        bad["trace"]["events"][0]["timestamp"] = "yesterday"
        assert app_client.post("/api/v2/score", json=bad).status_code == 422

    def test_too_many_events_rejected(self, app_client, sample_traces):
        big = copy.deepcopy(_payload(sample_traces[0][0]))
        big["trace"]["events"] = big["trace"]["events"][:1] * 201
        assert app_client.post("/api/v2/score", json=big).status_code == 422


class TestCommitGuard:
    def test_commit_forbidden_by_default(self, app_client, sample_traces):
        assert app_client.post("/api/v2/score", json=_payload(sample_traces[0][0], commit=True)).status_code == 403

    def test_commit_works_when_enabled_and_grows_store(self, app_client, sample_traces, monkeypatch):
        monkeypatch.setenv("V2_ALLOW_COMMIT", "true")
        before = app_client.get("/api/v2/status").json()["entity_store"]["customers"]
        assert app_client.post("/api/v2/score", json=_payload(sample_traces[1][0], commit=True)).status_code == 200
        assert app_client.get("/api/v2/status").json()["entity_store"]["customers"] == before + 1


class TestDrift:
    def test_drift_reports_after_enough_traffic(self, app_client, sample_traces):
        legit, fraud = sample_traces
        for i in range(60):
            rec = (legit + fraud)[i % 6]
            app_client.post("/api/v2/score", json=_payload(rec))
        body = app_client.get("/api/v2/drift").json()
        assert body["status"] in {"stable", "watch", "drift"} and body["n_observed"] >= 50
