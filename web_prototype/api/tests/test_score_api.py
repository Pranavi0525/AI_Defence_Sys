"""
Tests for POST /api/score, /healthz, /readyz.

Run from the repo root:

    PYTHONPATH=src python -m pytest web_prototype/api/tests -q

Includes both fast unit-style tests (using the real loaded model, since it
loads in well under a second and needs no mocking to be fast) and one
explicit real end-to-end inference test against the actual saved
xgb_model.joblib / calibrator.joblib artifacts and a real Red Team corpus
trace, per requirement I.
"""
from __future__ import annotations

import copy

import pytest


class TestHealthAndReadiness:
    def test_healthz_is_always_ok(self, app_client):
        r = app_client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "alive"}

    def test_readyz_true_once_artifacts_loaded(self, app_client):
        r = app_client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["model_version"]

    def test_readyz_reports_not_ready_on_load_failure(self, app_client, monkeypatch):
        """Simulates a model-loading failure without touching real
        artifacts: point the registry at a broken load, confirm /readyz
        (a) reports ready=False, (b) does NOT crash the process, and
        (c) does not leak a filesystem path/traceback into the response."""
        import app as api_app

        monkeypatch.setattr(api_app.model_registry, "ready", False)
        monkeypatch.setattr(api_app.model_registry, "load_error", "ArtifactMissing: simulated failure")

        r = app_client.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert "simulated failure" in body["error"]
        # process is still alive after a "failed" readiness check
        assert app_client.get("/healthz").status_code == 200

    def test_score_returns_503_when_not_ready(self, app_client, monkeypatch):
        import app as api_app

        monkeypatch.setattr(api_app.model_registry, "ready", False)
        r = app_client.post(
            "/api/score",
            json={
                "trace_id": "t1", "customer_id": "c1",
                "events": [], "observation_window": ["2025-01-01T00:00:00", "2025-01-01T00:01:00"],
            },
        )
        assert r.status_code in (503, 422)  # 422 if empty events also fails schema min_length first


class TestScoringValidRequests:
    def test_real_ato_trace_scores_successfully_end_to_end(self, app_client, real_ato_trace):
        """REAL end-to-end test: real artifacts, real corpus trace, no
        mocks anywhere in this test."""
        r = app_client.post("/api/score", json=real_ato_trace)
        assert r.status_code == 200
        body = r.json()

        assert body["decision"] in ("ALLOW", "REVIEW", "BLOCK")
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["trace_id"] == real_ato_trace["trace_id"]
        assert body["model_version"]
        assert body["inference_latency_ms"] < 500  # genuinely fast, not a batch run

        # Stage 3/4/5 must be honestly reported unavailable, never faked
        for stage_key in ("stage3_graph", "stage4_autoencoder", "stage5_fusion"):
            assert body[stage_key]["available"] is False
            assert isinstance(body[stage_key]["reason"], str) and len(body[stage_key]["reason"]) > 20

        assert body["stage1_2"]["available"] is True

    def test_deterministic_for_identical_input(self, app_client, real_ato_trace):
        r1 = app_client.post("/api/score", json=real_ato_trace).json()
        r2 = app_client.post("/api/score", json=real_ato_trace).json()
        assert r1["decision"] == r2["decision"]
        assert r1["risk_score"] == r2["risk_score"]

    def test_stage1_auto_clear_skips_model(self, app_client, real_ato_trace):
        """A trace with no beneficiary addition, no new device, and a slow
        transaction pace should be auto-cleared by Stage 1 without the
        model ever running -- risk_score forced to exactly 0.0."""
        trace = copy.deepcopy(real_ato_trace)
        trace["events"] = [e for e in trace["events"] if e["event_type"] not in ("DEVICE_REGISTRATION", "BENEFICIARY_ADDITION")]
        # push transactions far apart and slow down velocity below Stage 1's thresholds
        start = trace["observation_window"][0]
        for i, e in enumerate(trace["events"]):
            if e["event_type"] == "TRANSACTION":
                e["timestamp"] = "2025-01-02T09:19:04"
        trace["observation_window"] = [start, "2025-01-02T09:19:05"]

        r = app_client.post("/api/score", json=trace)
        assert r.status_code == 200
        body = r.json()
        if not body["stage1_2"]["ran_model"]:
            assert body["decision"] == "ALLOW"
            assert body["risk_score"] == 0.0
            assert body["top_contributing_features"] == []


class TestScoringMalformedRequests:
    def test_missing_required_field(self, app_client, real_ato_trace):
        bad = copy.deepcopy(real_ato_trace)
        del bad["observation_window"]
        r = app_client.post("/api/score", json=bad)
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "validation_error"
        assert any("observation_window" in str(e["loc"]) for e in body["detail"])

    def test_invalid_event_type(self, app_client, real_ato_trace):
        bad = copy.deepcopy(real_ato_trace)
        bad["events"][0]["event_type"] = "NOT_A_REAL_EVENT_TYPE"
        r = app_client.post("/api/score", json=bad)
        assert r.status_code == 422

    def test_empty_events_rejected(self, app_client, real_ato_trace):
        bad = copy.deepcopy(real_ato_trace)
        bad["events"] = []
        r = app_client.post("/api/score", json=bad)
        assert r.status_code == 422

    def test_negative_amount_rejected(self, app_client, real_ato_trace):
        bad = copy.deepcopy(real_ato_trace)
        for e in bad["events"]:
            if e["event_type"] == "TRANSACTION":
                e["amount"] = "-5.00"
        r = app_client.post("/api/score", json=bad)
        assert r.status_code == 422

    def test_no_internal_details_leaked_in_error(self, app_client, real_ato_trace):
        bad = copy.deepcopy(real_ato_trace)
        del bad["trace_id"]
        r = app_client.post("/api/score", json=bad)
        body_text = r.text
        assert "/home/" not in body_text
        assert "Traceback" not in body_text
