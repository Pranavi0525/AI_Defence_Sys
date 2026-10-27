"""API-level edge cases for the card module: validation errors, CSV handling, history cleaning, response fields."""
from __future__ import annotations

import pytest

DAY = 86400
T0 = 20 * DAY + 10 * 3600
COLS = "card,ts,amount,mcc,channel,lat,lon,merchant,declined"


def txn(**kw):
    d = dict(card="C-1001", ts=T0 + 7 * DAY + 900, amount=560, mcc=5411, channel="POS", lat=13.0827, lon=80.2707, merchant="M5411_3", declined=0)
    d.update(kw); return d


def history(card="C-1001"):
    amt = [640, 380, 910, 455, 520, 700, 1500]
    return [txn(card=card, ts=T0 + i * DAY + i * 1237, amount=amt[i], merchant="M5411_3") for i in range(7)]


class TestExtremeAndInvalidAmounts:
    def test_ten_billion_is_rejected_by_the_business_limit_not_scored(self, app_client):
        r = app_client.post("/api/card/predict", json={"transaction": txn(amount=1e10), "history": history()})
        b = r.json()
        assert r.status_code == 422 and b["error"] == "invalid_card_input" and "model_score" not in b
        assert b["detail"][0]["field"] == "amount" and "INR" in b["detail"][0]["problem"]

    @pytest.mark.parametrize("amt", [0, -1])
    def test_zero_and_negative_are_422(self, app_client, amt):
        assert app_client.post("/api/card/predict", json={"transaction": txn(amount=amt)}).status_code == 422

    def test_nan_and_infinity_in_json_are_422(self, app_client):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = '{"transaction": ' + str(__import__("json").dumps(txn(amount=1))).replace('"amount": 1,', f'"amount": {token},') + "}"
            r = app_client.post("/api/card/predict", content=body, headers={"Content-Type": "application/json"})
            assert r.status_code == 422, token

    @pytest.mark.parametrize("patch", [{"card": ""}, {"ts": -1}, {"ts": 5e12}, {"lat": 95}, {"lon": 200}, {"mcc": "ABC"}, {"mcc": 99999}, {"merchant": ""}])
    def test_other_invalid_fields_are_422(self, app_client, patch):
        assert app_client.post("/api/card/predict", json={"transaction": txn(**patch), "history": history()}).status_code == 422

    def test_missing_card_is_422(self, app_client):
        t = txn(); del t["card"]
        assert app_client.post("/api/card/predict", json={"transaction": t}).status_code == 422

    def test_limit_is_published_with_units(self, app_client):
        s = app_client.get("/api/card/status").json(); c = app_client.get("/api/card/model-card").json()
        assert s["currency"] == "INR" and s["max_transaction_amount"] > 0
        assert c["input_schema"]["max_transaction_amount_env"] == "CARD_MAX_TXN_AMOUNT" and c["amount_support"]["train_max_amount"] > 0

    def test_status_metadata_separates_limit_and_training_support(self, app_client):
        s = app_client.get("/api/card/status").json()
        assert s["currency"] == "INR"
        assert s["configured_serving_input_limit"] == 1_000_000.0
        assert s["max_transaction_amount"] == 1_000_000.0
        assert isinstance(s["training_amount_support"], dict)
        assert s["training_amount_support"]["train_max_amount"] > 0
        assert s["serving_limit_exceeds_training_support"] is True
        # Synthetic training support must not be described as issuer policy
        status_text = str(s).lower()
        assert "issuer" not in status_text and "policy" not in status_text or "policy" in str(s.get("disclaimer", "")).lower()

    def test_batch_api_out_of_training_range_support(self, app_client):
        # 600,000 INR is above training support (~465k) but below serving limit (1M)
        t_inside = txn(amount=560, ts=T0 + 7 * DAY + 100)
        t_outside = txn(amount=600_000, ts=T0 + 7 * DAY + 200)
        res = app_client.post("/api/card/predict-batch", json={"transactions": history() + [t_inside, t_outside]}).json()
        results = res["results"]
        # Row 7 is inside training support
        assert results[7]["outside_training_range"] is False
        assert results[7]["score_reliable"] is True
        assert results[7]["features_outside_training_range"] == []
        # Row 8 is outside training support
        assert results[8]["outside_training_range"] is True
        assert results[8]["score_reliable"] is False
        assert "amount" in results[8]["features_outside_training_range"]
        assert any("outside the range" in w for w in results[8]["warnings"])
        # Batch summary reflects count
        assert res["summary"]["outside_training_range"] == 1

    def test_single_and_batch_api_consistency_for_extreme_amount(self, app_client):
        t = txn(amount=600_000, ts=T0 + 7 * DAY + 200)
        single = app_client.post("/api/card/predict", json={"transaction": t, "history": history()}).json()
        batch = app_client.post("/api/card/predict-batch", json={"transactions": history() + [t]}).json()["results"][-1]
        assert single["outside_training_range"] == batch["outside_training_range"] == True
        assert single["score_reliable"] == batch["score_reliable"] == False
        assert single["features_outside_training_range"] == batch["features_outside_training_range"]
        assert "amount" in single["features_outside_training_range"]
        assert single["classification"] == batch["classification"]
        assert single["model_score"] == pytest.approx(batch["model_score"], abs=1e-9)
        assert any("outside the range" in w for w in single["warnings"])
        assert any("outside the range" in w for w in batch["warnings"])

    def test_configuration_error_handling(self, monkeypatch):
        from v2.card.limits import LimitsConfigError, max_txn_amount

        for bad in ["abc", "-1", "0", "nan", "inf", "-inf"]:
            monkeypatch.setenv("CARD_MAX_TXN_AMOUNT", bad)
            with pytest.raises(LimitsConfigError):
                max_txn_amount()


class TestHistoryOverApi:
    def test_amount_reaches_the_model_unchanged_and_response_is_json_safe(self, app_client):
        r = app_client.post("/api/card/predict", json={"transaction": txn(amount=123456.78), "history": history()}).json()
        assert r["features"]["amount"] == 123456.78 and r["prior_transactions_on_card"] == 7 == r["history_rows_used"]

    def test_no_history_has_null_not_nan(self, app_client):
        r = app_client.post("/api/card/predict", json={"transaction": txn()})
        assert r.status_code == 200 and r.json()["features"]["amt_z"] is None

    def test_other_card_rows_are_dropped_and_reported(self, app_client):
        a = app_client.post("/api/card/predict", json={"transaction": txn(), "history": history()}).json()
        b = app_client.post("/api/card/predict", json={"transaction": txn(), "history": history() + history("C-2")}).json()
        assert b["model_score"] == a["model_score"] and b["history_dropped"]["other_card"] == 7

    def test_dashboard_example_ordinary_card_and_burst(self, app_client):
        burst = [txn(card="C-2002", ts=T0 + 7 * DAY + 40 * (i + 1), amount=[2.5, 4, 9.99, 3, 14, 6][i], mcc=[5411, 5812, 5311, 5651, 4111, 5814][i],
                     channel="ECOM", lat=19.076, lon=72.8777, merchant="M0_t%d" % (i % 3), declined=[1, 1, 0, 1, 0, 1][i]) for i in range(6)]
        t = txn(card="C-2002", ts=T0 + 7 * DAY + 335, amount=6400, mcc=5947, channel="ECOM", lat=19.076, lon=72.8777, merchant="M0_t1")
        ok = app_client.post("/api/card/predict", json={"transaction": txn(), "history": history()}).json()
        bu = app_client.post("/api/card/predict", json={"transaction": t, "history": history("C-2002") + burst}).json()
        assert ok["classification"] == "LEGITIMATE" and bu["classification"] == "FRAUD" and bu["features"]["dec_10m"] == 4 and bu["features"]["cnt_10m"] == 6


class TestCsvOverApi:
    def test_empty_and_header_only_csv_are_clean_422(self, app_client):
        for text in ["", "   \n", COLS + "\n"]:
            r = app_client.post("/api/card/predict-batch", json={"csv_text": text})
            assert r.status_code == 422 and r.json()["error"] == "invalid_card_input"

    def test_malformed_csv_names_row_and_field(self, app_client):
        r = app_client.post("/api/card/predict-batch", json={"csv_text": COLS + "\nC1,100,abc,5411,POS,13,80,M1,0\nC1,200,50,5411,POS,13,80,M1,0\n"})
        assert r.status_code == 422 and {"row": 0, "field": "amount"}.items() <= r.json()["detail"][0].items()

    def test_duplicate_header_column_is_rejected(self, app_client):
        r = app_client.post("/api/card/predict-batch", json={"csv_text": "card,ts,amount,amount,mcc,channel,lat,lon,merchant\nC1,1,5,6,5411,POS,13,80,M1\n"})
        assert r.status_code == 422

    def test_csv_amount_over_limit_rejected_with_row(self, app_client):
        r = app_client.post("/api/card/predict-batch", json={"csv_text": COLS + "\nC1,100,1e10,5411,POS,13,80,M1,0\n"})
        assert r.status_code == 422 and r.json()["detail"][0]["field"] == "amount"

    def test_row_with_missing_cell_is_rejected_not_imputed(self, app_client):
        r = app_client.post("/api/card/predict-batch", json={"csv_text": COLS + "\n,100,50,5411,POS,13,80,M1,0\n"})
        assert r.status_code == 422 and r.json()["detail"][0]["field"] == "card"

    def test_unsorted_csv_rows_scored_in_time_order(self, app_client):
        rows = history()
        asc = COLS + "\n" + "\n".join(",".join(str(r[c]) for c in COLS.split(",")) for r in rows)
        desc = COLS + "\n" + "\n".join(",".join(str(r[c]) for c in COLS.split(",")) for r in reversed(rows))
        a = app_client.post("/api/card/predict-batch", json={"csv_text": asc}).json()["results"]
        b = app_client.post("/api/card/predict-batch", json={"csv_text": desc}).json()["results"]
        assert [x["model_score"] for x in a] == pytest.approx([x["model_score"] for x in reversed(b)], abs=1e-12)

    def test_large_batch_over_cap_is_rejected(self, app_client):
        r = app_client.post("/api/card/predict-batch", json={"transactions": [txn()] * 50001})
        assert r.status_code == 422
