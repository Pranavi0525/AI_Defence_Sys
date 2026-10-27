"""Tests for /api/card/* (credit-card module) and its isolation from the digital-payment API.

Uses the committed models/card bundle (SYNTHETIC simulator-trained) through the real app lifespan.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from v2.card.sim import simulate

API_DIR = Path(__file__).resolve().parent.parent
COLS = ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant", "declined"]


@pytest.fixture(scope="module")
def sim_rows():
    df = simulate(n_cards=40, days=30, seed=1, events_per_pattern=5)
    return json.loads(df[COLS].to_json(orient="records")), df


def _card_hist(rows, card):
    return sorted([r for r in rows if r["card"] == card], key=lambda r: r["ts"])


class TestStatusAndModelCard:
    def test_status_ready_and_labelled_synthetic(self, app_client):
        b = app_client.get("/api/card/status").json()
        assert b["ready"] is True and b["synthetic_trained"] is True
        assert b["model_version"].startswith("card-xgb-synthetic-") and b["feature_count"] == 27
        assert "not a real-world fraud probability" in b["disclaimer"]

    def test_model_card_has_evaluation_and_limitations(self, app_client):
        c = app_client.get("/api/card/model-card").json()
        assert c["synthetic"] is True and "SYNTHETIC" in c["evaluation"]["label"]
        assert {"tp", "fp", "fn", "tn", "precision", "recall", "f1"} <= set(c["evaluation"]["at_threshold"])
        assert c["evaluation"]["threshold_sweep_by_validation_fpr"]
        assert c["input_schema"]["required"] == ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant"]
        assert c["limitations"] and "probability" in c["score_meaning"]


class TestPredict:
    def test_single_prediction_shape(self, app_client, sim_rows):
        rows, _ = sim_rows
        hist = _card_hist(rows, rows[0]["card"])
        r = app_client.post("/api/card/predict", json={"transaction": hist[-1], "history": hist[:-1]})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["classification"] in {"FRAUD", "LEGITIMATE"} and b["is_fraud"] == (b["classification"] == "FRAUD")
        assert 0 <= b["legit_percentile"] <= 1 and b["model_score"] >= 0
        assert b["prior_transactions_on_card"] == len(hist) - 1
        assert "not a real-world fraud probability" in b["disclaimer"] and "index" not in b
        assert b["is_fraud"] == (b["model_score"] >= b["threshold"])

    def test_no_history_is_allowed_and_warned(self, app_client, sim_rows):
        rows, _ = sim_rows
        b = app_client.post("/api/card/predict", json={"transaction": rows[0]}).json()
        assert any("no card history" in w for w in b["warnings"])

    def test_single_equals_batch(self, app_client, sim_rows):
        rows, _ = sim_rows
        card = max({r["card"] for r in rows}, key=lambda c: len(_card_hist(rows, c)))
        hist = _card_hist(rows, card)
        batch = app_client.post("/api/card/predict-batch", json={"transactions": hist}).json()
        for i in (0, len(hist) // 2, len(hist) - 1):
            single = app_client.post("/api/card/predict", json={"transaction": hist[i], "history": hist[:i]}).json()
            assert single["model_score"] == pytest.approx(batch["results"][i]["model_score"], abs=1e-9)
            assert single["classification"] == batch["results"][i]["classification"]

    def test_card_testing_burst_is_flagged(self, app_client, sim_rows):
        rows, df = sim_rows
        card = int(df[df.pattern == "CARD_TESTING"].card.iloc[0])
        res = app_client.post("/api/card/predict-batch", json={"transactions": _card_hist(rows, card)}).json()
        assert res["summary"]["flagged_fraud"] > 0

    @pytest.mark.parametrize("patch", [{"amount": -1}, {"amount": 0}, {"lat": 120}, {"lon": 300}, {"ts": -5}, {"merchant": ""}, {"surprise": 1}])
    def test_invalid_single_inputs_are_422(self, app_client, sim_rows, patch):
        rows, _ = sim_rows
        r = app_client.post("/api/card/predict", json={"transaction": {**rows[0], **patch}})
        assert r.status_code == 422 and r.json()["error"] == "validation_error"

    def test_bad_channel_is_422_with_field(self, app_client, sim_rows):
        rows, _ = sim_rows
        r = app_client.post("/api/card/predict", json={"transaction": {**rows[0], "channel": "WIRE"}})
        assert r.status_code == 422 and r.json()["error"] == "invalid_card_input"
        assert r.json()["detail"][0]["field"] == "channel"

    def test_missing_field_is_422(self, app_client, sim_rows):
        rows, _ = sim_rows
        t = {k: v for k, v in rows[0].items() if k != "amount"}
        assert app_client.post("/api/card/predict", json={"transaction": t}).status_code == 422


class TestBatch:
    def test_batch_results_and_summary(self, app_client, sim_rows):
        rows, _ = sim_rows
        b = app_client.post("/api/card/predict-batch", json={"transactions": rows}).json()
        assert len(b["results"]) == len(rows) == b["summary"]["transactions"]
        assert b["summary"]["flagged_fraud"] + b["summary"]["legitimate"] == len(rows)
        assert [r["index"] for r in b["results"]] == list(range(len(rows)))
        assert "not a real-world fraud probability" in b["disclaimer"]

    def test_csv_and_json_agree(self, app_client, sim_rows):
        rows, _ = sim_rows
        sub = rows[:300]
        csv = "card,ts,amount,mcc,channel,lat,lon,merchant,declined\n" + "\n".join(
            ",".join(str(r[c]) for c in COLS) for r in sub)
        a = app_client.post("/api/card/predict-batch", json={"csv_text": csv}).json()
        b = app_client.post("/api/card/predict-batch", json={"transactions": sub}).json()
        assert [x["model_score"] for x in a["results"]] == pytest.approx([x["model_score"] for x in b["results"]])

    def test_csv_declined_column_optional_and_header_case_insensitive(self, app_client):
        csv = " Card , TS ,Amount,MCC,Channel,Lat,Lon,Merchant\n1,1000,50,5411,pos,13.08,80.27,M5411_1\n1,2000,60,5411,POS,13.08,80.27,M5411_1\n"
        r = app_client.post("/api/card/predict-batch", json={"csv_text": csv})
        assert r.status_code == 200 and r.json()["summary"]["transactions"] == 2

    def test_csv_bad_cell_names_row_and_field(self, app_client):
        csv = "card,ts,amount,mcc,channel,lat,lon,merchant\n1,1000,50,5411,POS,13.08,80.27,M1\n1,2000,abc,5411,POS,13.08,80.27,M1\n"
        r = app_client.post("/api/card/predict-batch", json={"csv_text": csv})
        assert r.status_code == 422
        assert {"row": 1, "field": "amount"}.items() <= r.json()["detail"][0].items()

    def test_csv_empty_cell_is_missing_not_imputed(self, app_client):
        csv = "card,ts,amount,mcc,channel,lat,lon,merchant\n1,1000,,5411,POS,13.08,80.27,M1\n"
        r = app_client.post("/api/card/predict-batch", json={"csv_text": csv})
        assert r.status_code == 422 and r.json()["detail"][0]["field"] == "amount"

    def test_csv_missing_column_and_empty(self, app_client):
        assert app_client.post("/api/card/predict-batch", json={"csv_text": "card,ts\n1,2\n"}).status_code == 422
        assert app_client.post("/api/card/predict-batch", json={"csv_text": ""}).status_code == 422
        assert app_client.post("/api/card/predict-batch", json={"csv_text": "card,ts,amount,mcc,channel,lat,lon,merchant\n"}).status_code == 422

    def test_exactly_one_source_required(self, app_client, sim_rows):
        rows, _ = sim_rows
        assert app_client.post("/api/card/predict-batch", json={}).status_code == 422
        assert app_client.post("/api/card/predict-batch", json={"transactions": rows[:2], "csv_text": "x"}).status_code == 422


# ----------------------------------------------------------------------------- isolation between the two modules
class TestIsolation:
    def _payment_still_works(self, client):
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/health").status_code == 200
        body = client.get("/api/v2/status").json()
        assert body["ready"] is True
        assert "v2_ready" in client.get("/readyz").json()
        assert client.get("/api/reports/dashboard").status_code == 200

    def test_missing_card_bundle_only_disables_card_routes(self, tmp_path, monkeypatch, sim_rows):
        from fastapi.testclient import TestClient
        import app as api_app
        rows, _ = sim_rows
        monkeypatch.setenv("CARD_MODEL_DIR", str(tmp_path))              # empty directory: no artifacts
        with TestClient(api_app.app) as c:
            st = c.get("/api/card/status").json()
            assert st["ready"] is False and "missing" in st["error"] and "No fallback" in st["error"]
            for path, body in (("/api/card/predict", {"transaction": rows[0]}), ("/api/card/predict-batch", {"transactions": rows[:3]})):
                r = c.post(path, json=body)
                assert r.status_code == 503 and "missing" in r.json()["detail"]
            assert c.get("/api/card/model-card").status_code == 503
            self._payment_still_works(c)
        monkeypatch.undo()

    def test_corrupted_card_bundle_only_disables_card_routes(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        import app as api_app
        src = Path(__file__).resolve().parents[3] / "models" / "card"
        (tmp_path / "policy.json").write_bytes((src / "policy.json").read_bytes())
        (tmp_path / "xgb.json").write_bytes(b"{}")
        monkeypatch.setenv("CARD_MODEL_DIR", str(tmp_path))
        with TestClient(api_app.app) as c:
            assert c.get("/api/card/status").json()["ready"] is False
            self._payment_still_works(c)
        monkeypatch.undo()

    def test_unexpected_card_crash_returns_500_envelope_and_payment_is_unaffected(self, app_client, sim_rows, monkeypatch):
        from fastapi.testclient import TestClient
        import app as api_app
        import card_routes
        rows, _ = sim_rows
        with TestClient(api_app.app, raise_server_exceptions=False) as c:
            monkeypatch.setattr(card_routes.registry.scorer, "score_batch", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
            r = c.post("/api/card/predict-batch", json={"transactions": rows[:3]})
            assert r.status_code == 500 and r.json()["error"] == "internal_server_error" and "boom" not in r.text
            self._payment_still_works(c)

    def test_card_import_failure_cannot_stop_the_payment_app(self):
        """In a fresh interpreter, make `import card_routes` fail; the payment API must still start and serve."""
        code = textwrap.dedent(f"""
            import sys, os
            os.environ["ENVIRONMENT"] = "development"
            sys.path.insert(0, {str(API_DIR)!r})
            sys.modules["card_routes"] = None            # -> ImportError on `import card_routes`
            import app as api_app
            assert api_app.card_routes is None
            from fastapi.testclient import TestClient
            with TestClient(api_app.app) as c:
                assert c.get("/healthz").status_code == 200
                assert c.get("/api/v2/status").json()["ready"] is True
                assert c.get("/api/card/status").status_code == 404
            print("OK")
        """)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-500:] + r.stderr[-1500:]

    def test_card_endpoints_do_not_depend_on_payment_models_being_loaded(self, app_client, sim_rows, monkeypatch):
        import app as api_app
        import v2_routes
        rows, _ = sim_rows
        monkeypatch.setattr(api_app.model_registry, "ready", False)          # v1 payment model "down"
        monkeypatch.setattr(v2_routes.registry, "scorer", None)              # v2 payment cascade "down"
        r = app_client.post("/api/card/predict", json={"transaction": rows[0]})
        assert r.status_code == 200
        assert app_client.get("/api/v2/status").json()["ready"] is False     # payment side really is down in this test
        assert app_client.get("/readyz").status_code == 503
