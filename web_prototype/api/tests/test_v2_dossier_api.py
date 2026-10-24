"""Tests for POST /api/v2/dossier (uses the real bundle in models/v2)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _payload(name, **extra):
    j = json.loads((ROOT / "web_prototype" / "examples" / name).read_text(encoding="utf-8"))
    tr = j.get("trace", j)
    return {"trace": {k: tr[k] for k in ("trace_id", "customer_id", "events", "observation_window")}, **extra}


class TestDossier:
    def test_block_case_returns_dossier_and_markdown(self, app_client):
        r = app_client.post("/api/v2/dossier", json=_payload("v2_sample_deepfake_kyc.json", family="DEEPFAKE_KYC"))
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["dossier"]["decision"] in ("REVIEW", "BLOCK") and "## Case" in b["markdown"]
        assert b["dossier"]["acting_party"]["basis"] == "ASSUMED"

    def test_allow_case_has_no_dossier_by_default(self, app_client):
        b = app_client.post("/api/v2/dossier", json=_payload("v2_sample_routine_legit.json")).json()
        assert b["dossier"] is None and "ALLOW" in b["note"]

    def test_include_allow_forces_dossier(self, app_client):
        b = app_client.post("/api/v2/dossier", json=_payload("v2_sample_routine_legit.json", include_allow=True)).json()
        assert b["dossier"]["decision"] == "ALLOW"

    def test_unknown_family_rejected(self, app_client):
        assert app_client.post("/api/v2/dossier", json=_payload("v2_sample_deepfake_kyc.json", family="NOT_A_FAMILY")).status_code == 422

    def test_dossier_does_not_change_entity_store(self, app_client):
        before = app_client.get("/api/v2/status").json()["entity_store"]
        app_client.post("/api/v2/dossier", json=_payload("v2_sample_deepfake_kyc.json"))
        assert app_client.get("/api/v2/status").json()["entity_store"] == before
