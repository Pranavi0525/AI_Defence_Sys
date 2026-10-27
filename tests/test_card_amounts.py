"""Edge-case + large-amount tests for the card module (SYNTHETIC data; committed models/card bundle).

Covers: amount/feature correctness, causality, cross-card isolation, history cleaning, validation limits, and how the served
model treats large legitimate purchases vs high-value fraud scenarios. See reports/card/large_amount_investigation.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v2.card.features import FEATURES, build_features
from v2.card.limits import DEFAULT_MAX_TXN_AMOUNT, ENV_MAX_TXN_AMOUNT
from v2.card.serving import CardInputError, CardModelError, CardScorer, validate_rows
from v2.card.sim import simulate

REPO = Path(__file__).resolve().parents[1]
DAY = 86400
T0 = 20 * DAY + 10 * 3600
M = ["M5411_3", "M5812_2", "M5411_3", "M5411_1", "M5812_2", "M5411_3", "M5541_4"]
MCC = [5411, 5812, 5411, 5411, 5812, 5411, 5541]
AMT = [640, 380, 910, 455, 520, 700, 1500]


def hist(card="C-1001"):
    return [dict(card=card, ts=T0 + i * DAY + i * 1237, amount=AMT[i], mcc=MCC[i], channel="POS", lat=13.0827, lon=80.2707,
                 merchant=M[i], declined=0) for i in range(7)]


def tx(amount=560, **kw):
    d = dict(card="C-1001", ts=T0 + 7 * DAY + 900, amount=amount, mcc=5411, channel="POS", lat=13.0827, lon=80.2707,
             merchant="M5411_3", declined=0)
    d.update(kw)
    return d


@pytest.fixture(scope="module")
def scorer():
    return CardScorer(REPO / "models" / "card")


@pytest.fixture(scope="module")
def fresh_frame_scores(scorer):
    """A simulator draw with a seed the model was NOT trained on (extended mode), scored in one batch."""
    df = simulate(n_cards=250, days=60, seed=11, events_per_pattern=12, extended=True)
    cols = ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant", "declined"]
    res = scorer.score_batch(df[cols].to_dict("records"))
    out = df.reset_index(drop=True).copy()
    out["flag"] = [r["is_fraud"] for r in res]
    return out


# --------------------------------------------------------------------------- features
class TestFeatures:
    def test_amount_features_and_no_self_inclusion(self):
        f = build_features(pd.DataFrame(hist() + [tx(9000)]).assign(mcc=lambda d: d.mcc.astype(str)))
        last = f.iloc[-1]
        assert last.hist_len == 7 and last.amount == 9000 and last.log_amount == pytest.approx(math.log1p(9000))
        prior = np.log1p(AMT)
        z = (math.log1p(9000) - prior.mean()) / math.sqrt(max((prior ** 2).mean() - prior.mean() ** 2, 1e-6))
        assert last.amt_z == pytest.approx(z, rel=1e-6)                       # statistics from the 7 PRIOR rows only
        assert last.amt_ratio_max == pytest.approx(9000 / max(AMT))

    def test_history_statistics_do_not_change_when_current_amount_changes(self):
        a = build_features(pd.DataFrame(hist() + [tx(500)]).assign(mcc=lambda d: d.mcc.astype(str))).iloc[-1]
        b = build_features(pd.DataFrame(hist() + [tx(5e5)]).assign(mcc=lambda d: d.mcc.astype(str))).iloc[-1]
        for k in ["hist_len", "gap_s", "cnt_10m", "cnt_1h", "cnt_24h", "sum_24h", "new_mcc", "new_merchant", "dist_home_km"]:
            assert a[k] == b[k], k

    def test_feature_order_is_the_policy_order(self, scorer):
        assert scorer.features == FEATURES and list(scorer.booster.feature_names) == FEATURES

    def test_extreme_amount_features_are_finite_and_not_clipped(self):
        f = build_features(pd.DataFrame(hist() + [tx(1e10)]).assign(mcc=lambda d: d.mcc.astype(str))).iloc[-1]
        assert all(np.isfinite(f[k]) for k in ["amount", "log_amount", "amt_z", "amt_ratio_max"])
        assert f.amount == 1e10 and f.amt_ratio_max == pytest.approx(1e10 / 1500)

    def test_rapid_repeat_velocity(self):
        rows = hist() + [tx(100, ts=T0 + 7 * DAY + 30 * i, merchant="MX", declined=int(i % 2)) for i in range(6)]
        f = build_features(pd.DataFrame(rows).assign(mcc=lambda d: d.mcc.astype(str)))
        assert list(f.cnt_10m.tail(6)) == [0, 1, 2, 3, 4, 5]
        assert f.dec_10m.iloc[-1] == 2 and f.same_merch_24h.iloc[-1] == 5      # prior declines (rows 1,3) -> 2; own flag excluded

    def test_spending_spike_features(self):
        f = build_features(pd.DataFrame(hist() + [tx(560), tx(60000, ts=T0 + 8 * DAY)]).assign(mcc=lambda d: d.mcc.astype(str)))
        assert f.amt_z.iloc[-1] > 5 and f.amt_ratio_max.iloc[-1] > 30 and f.amt_z.iloc[-2] < 2

    def test_unsorted_input_is_processed_chronologically(self, scorer):
        rows = hist() + [tx()]
        shuffled = [rows[i] for i in (3, 7, 0, 5, 1, 6, 2, 4)]
        a = scorer.score_one(rows[-1], rows[:-1])
        b = scorer.score_one(shuffled[1], [r for r in shuffled if r is not shuffled[1]])
        assert a["model_score"] == pytest.approx(b["model_score"], abs=1e-12) and b["prior_transactions_on_card"] == 7


# --------------------------------------------------------------------------- validation
BAD = [({"amount": 0}, "amount"), ({"amount": -1}, "amount"), ({"amount": float("nan")}, "amount"), ({"amount": float("inf")}, "amount"),
       ({"amount": -float("inf")}, "amount"), ({"card": None}, "card"), ({"card": "  "}, "card"), ({"ts": -1}, "ts"), ({"ts": float("nan")}, "ts"),
       ({"ts": "yesterday"}, "ts"), ({"ts": 5e12}, "ts"), ({"lat": 91}, "lat"), ({"lat": float("nan")}, "lat"), ({"lon": -181}, "lon"),
       ({"lon": float("inf")}, "lon"), ({"mcc": "ABC"}, "mcc"), ({"mcc": 12345}, "mcc"), ({"mcc": -5}, "mcc"), ({"mcc": 54.5}, "mcc"),
       ({"mcc": None}, "mcc"), ({"merchant": ""}, "merchant"), ({"amount": 1e10}, "amount"), ({"amount": DEFAULT_MAX_TXN_AMOUNT * 1.0001}, "amount")]


class TestValidation:
    @pytest.mark.parametrize("patch,field", BAD)
    def test_bad_value_is_rejected_before_the_model(self, scorer, patch, field):
        with pytest.raises(CardInputError) as e:
            scorer.score_one(tx(**patch), hist())
        assert field in {x["field"] for x in e.value.errors}

    def test_row_numbers_point_at_the_offending_input_row(self, scorer):
        h = hist(); h[2]["amount"] = -5
        with pytest.raises(CardInputError) as e:
            scorer.score_one(tx(), h)
        assert e.value.errors[0]["row"] == 2

    def test_limit_message_states_units_and_is_not_a_prediction(self, scorer):
        with pytest.raises(CardInputError) as e:
            scorer.score_one(tx(1e10), hist())
        msg = e.value.errors[0]["problem"]
        assert "INR" in msg and "CARD_MAX_TXN_AMOUNT" in msg and "not scored" in msg

    def test_default_simulator_cap_and_separation_from_serving(self):
        from v2.card.limits import DEFAULT_SIM_MAX_AMOUNT, DEFAULT_MAX_TXN_AMOUNT
        assert DEFAULT_SIM_MAX_AMOUNT == 1_000_000.0
        assert DEFAULT_MAX_TXN_AMOUNT == 1_000_000.0
        from v2.card.sim import simulate
        import inspect
        sig = inspect.signature(simulate)
        assert sig.parameters["max_amount"].default is None

    def test_training_simulator_configuration(self):
        from v2.card.train import train_bundle
        import inspect
        sig = inspect.signature(train_bundle)
        assert sig.parameters["max_amount"].default is None

    def test_serving_limit_exceeds_training_support_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = CardScorer(REPO / "models" / "card")
            assert s.serving_limit_exceeds_training_support is True
            assert any("exceeds recorded training support" in str(item.message) for item in w)

    def test_limit_is_inclusive_and_configurable(self, monkeypatch):
        s_default = CardScorer(REPO / "models" / "card")
        # Amount exactly at the default serving limit -> accepted
        assert s_default.score_one(tx(DEFAULT_MAX_TXN_AMOUNT), hist())["classification"] in ("FRAUD", "LEGITIMATE")
        # Amount just above the default serving limit -> rejected
        with pytest.raises(CardInputError):
            s_default.score_one(tx(DEFAULT_MAX_TXN_AMOUNT + 0.01), hist())

        # Environment override at load time: 5,000 INR
        monkeypatch.setenv(ENV_MAX_TXN_AMOUNT, "5000")
        s5k = CardScorer(REPO / "models" / "card")
        assert s5k.amount_limit == 5000.0
        # Exactly at serving limit -> accepted
        assert s5k.score_one(tx(5000), hist())["prior_transactions_on_card"] == 7
        # Just above serving limit -> rejected
        with pytest.raises(CardInputError):
            s5k.score_one(tx(5000.01), hist())

        # Higher override: 2e7 INR
        monkeypatch.setenv(ENV_MAX_TXN_AMOUNT, "2e7")
        s20m = CardScorer(REPO / "models" / "card")
        assert s20m.amount_limit == 2e7
        assert s20m.score_one(tx(1.5e7), hist())["classification"]        # operator raised the limit

    @pytest.mark.parametrize("bad", ["abc", "-1", "0", "nan", "inf", "-inf"])
    def test_invalid_limit_config_fails_loudly(self, bad, monkeypatch):
        monkeypatch.setenv(ENV_MAX_TXN_AMOUNT, bad)
        with pytest.raises(CardModelError):
            CardScorer(REPO / "models" / "card")

    def test_mcc_int_and_string_are_the_same_merchant_category(self):
        a = validate_rows([tx(mcc=5411)]); b = validate_rows([tx(mcc="5411")]); c = validate_rows([tx(mcc="0742")])
        assert a.mcc.iloc[0] == b.mcc.iloc[0] == "5411" and c.mcc.iloc[0] == "0742"


# --------------------------------------------------------------------------- history handling
class TestHistory:
    def test_ordinary_transaction_valid_prediction(self, scorer):
        r = scorer.score_one(tx(560), hist())
        assert r["classification"] == "LEGITIMATE" and r["prior_transactions_on_card"] == 7 and r["history_rows_used"] == 7
        assert 0 <= r["legit_percentile"] <= 1 and set(r["features"]) == set(FEATURES)

    def test_no_history_works_and_is_flagged(self, scorer):
        r = scorer.score_one(tx(), None)
        assert r["prior_transactions_on_card"] == 0 and any("no card history" in w for w in r["warnings"])
        assert r["features"]["amt_z"] is None                              # NaN is reported as null, never invented

    def test_other_cards_history_never_contaminates(self, scorer):
        other = [dict(h, card="C-9999", amount=h["amount"] * 50) for h in hist()]
        noisy = [dict(card="C-9999", ts=T0 + 7 * DAY + 800 + i, amount=3, mcc=5999, channel="ECOM", lat=19.0, lon=72.8, merchant="Z", declined=1) for i in range(20)]
        a = scorer.score_one(tx(), hist())
        b = scorer.score_one(tx(), hist() + other + noisy)
        assert b["model_score"] == pytest.approx(a["model_score"], abs=1e-12) and b["features"] == a["features"]
        assert b["history_dropped"]["other_card"] == 27 and b["history_rows_used"] == 7 and any("different card" in w for w in b["warnings"])

    def test_card_id_int_vs_string_is_the_same_card(self, scorer):
        h = [dict(x, card=1001) for x in hist()]
        r = scorer.score_one(tx(card="1001"), h)
        assert r["prior_transactions_on_card"] == 7 and r["history_dropped"]["other_card"] == 0

    def test_duplicate_history_rows_are_dropped_first_kept(self, scorer):
        h = hist(); dup = h + [dict(h[3]), dict(h[3]), dict(h[5])]
        a = scorer.score_one(tx(), h); b = scorer.score_one(tx(), dup)
        assert b["history_dropped"]["duplicate"] == 3 and b["prior_transactions_on_card"] == 7
        assert b["model_score"] == pytest.approx(a["model_score"], abs=1e-12)

    def test_current_transaction_is_excluded_from_its_own_history(self, scorer):
        t = tx(); a = scorer.score_one(t, hist()); b = scorer.score_one(t, hist() + [dict(t)])
        assert b["history_dropped"]["same_as_transaction"] == 1 and b["features"] == a["features"]

    def test_future_history_is_ignored(self, scorer):
        fut = [tx(777, ts=T0 + 9 * DAY)]
        a = scorer.score_one(tx(), hist()); b = scorer.score_one(tx(), hist() + fut)
        assert b["history_dropped"]["later_than_transaction"] == 1 and b["model_score"] == pytest.approx(a["model_score"], abs=1e-12)

    def test_previous_declines_enter_features_but_own_flag_does_not(self, scorer):
        h = hist() + [tx(50, ts=T0 + 7 * DAY + 500 + i * 10, merchant="MT", declined=1) for i in range(4)]
        r1 = scorer.score_one(tx(declined=0), h); r2 = scorer.score_one(tx(declined=1), h)
        assert r1["features"]["dec_10m"] == 4 and r1["features"] == r2["features"]


# --------------------------------------------------------------------------- model behaviour (SYNTHETIC)
class TestModelBehaviour:
    def test_large_legitimate_purchases_are_not_automatically_fraud(self, fresh_frame_scores):
        big = fresh_frame_scores[(fresh_frame_scores.amount >= 20000) & (fresh_frame_scores.is_fraud == 0)]
        assert len(big) >= 50
        assert big.flag.mean() < 0.35, f"{big.flag.mean():.1%} of large legit purchases flagged"
        assert big.flag.sum() < len(big)

    def test_overall_false_positive_rate_stays_controlled(self, fresh_frame_scores):
        leg = fresh_frame_scores[fresh_frame_scores.is_fraud == 0]
        assert leg.flag.mean() < 0.03

    def test_high_value_fraud_scenarios_are_mostly_detected(self, fresh_frame_scores):
        # NB: the second part checks the SYNTHETIC scenarios the simulator defines, not real fraud.
        d = fresh_frame_scores
        hv = d[(d.is_fraud == 1) & d.pattern.isin(["HIGH_VALUE_BUSTOUT", "SPEND_SPIKE_FAMILIAR", "NEW_CARD_FIRST_USE"])]
        assert len(hv) >= 30 and hv.flag.mean() > 0.5

    def test_more_than_one_feature_is_involved_no_single_amount_rule(self, scorer):
        # the same large amount is scored differently in different contexts -> the model is not an amount threshold
        s = {k: scorer.score_one(t, hist())["model_score"] for k, t in {
            "familiar": tx(50000), "new_merchant_far": tx(50000, mcc=5947, merchant="M5947_fx3", channel="ECOM", lat=19.076, lon=72.8777)}.items()}
        assert s["familiar"] != pytest.approx(s["new_merchant_far"], rel=1e-3)

    def test_extreme_amount_inside_limit_is_flagged_as_outside_training_range(self, scorer):
        r = scorer.score_one(tx(DEFAULT_MAX_TXN_AMOUNT), hist())
        assert "amount" in r["features_outside_training_range"]
        assert r["outside_training_range"] is True
        assert r["score_reliable"] is False
        assert any("outside the range" in w for w in r["warnings"])

    def test_ordinary_amount_is_inside_training_range(self, scorer):
        r = scorer.score_one(tx(560), hist())
        assert r["features_outside_training_range"] == []
        assert r["outside_training_range"] is False
        assert r["score_reliable"] is True

    def test_transactions_above_training_support_are_not_rejected(self, scorer):
        # 600,000 INR is above realized training support (~465,290.34) but within serving limit (1,000,000)
        # It must NOT be rejected, but must be scored with outside_training_range=True and score_reliable=False
        r = scorer.score_one(tx(600_000), hist())
        assert r["classification"] in ("FRAUD", "LEGITIMATE")
        assert 0 <= r["model_score"] <= 1
        assert r["outside_training_range"] is True
        assert r["score_reliable"] is False
        assert "amount" in r["features_outside_training_range"]
        assert any("outside the range" in w for w in r["warnings"])

    def test_batch_prediction_support_warnings_and_consistency(self, scorer):
        t_inside = tx(560, ts=T0 + 7 * DAY + 100)
        t_outside = tx(600_000, ts=T0 + 7 * DAY + 200)
        rows = hist() + [t_inside, t_outside]
        batch_res = scorer.score_batch(rows)

        # Check inside row (row 7)
        b_in = batch_res[7]
        assert b_in["outside_training_range"] is False
        assert b_in["score_reliable"] is True
        assert b_in["features_outside_training_range"] == []
        assert not any("outside the range" in w for w in b_in["warnings"])

        # Check outside row (row 8)
        b_out = batch_res[8]
        assert b_out["outside_training_range"] is True
        assert b_out["score_reliable"] is False
        assert "amount" in b_out["features_outside_training_range"]
        assert any("outside the range" in w for w in b_out["warnings"])

        # Consistency with single prediction
        s_out = scorer.score_one(t_outside, hist() + [t_inside])
        assert s_out["outside_training_range"] == b_out["outside_training_range"]
        assert s_out["score_reliable"] == b_out["score_reliable"]
        assert s_out["features_outside_training_range"] == b_out["features_outside_training_range"]
        assert s_out["model_score"] == pytest.approx(b_out["model_score"], abs=1e-9)
        assert s_out["classification"] == b_out["classification"]

        # Summary reports outside_training_range count
        summary = CardScorer.summarize(batch_res)
        assert summary["outside_training_range"] == 1
        assert summary["transactions"] == len(rows)

    def test_amount_sweep_monotonicity_diagnostic(self, scorer):
        """Diagnostic, NOT a requirement: score vs amount with every other input fixed. Allowed to be non-monotonic (a tree model on
        multi-feature interactions); we assert only that it is bounded in [0,1] and that scores rise from ordinary to very large."""
        amts = [100, 560, 2000, 5000, 10000, 20000, 50000, 1e5, 3e5, 1e6]
        sc = [scorer.score_one(tx(a, channel="ECOM"), hist())["model_score"] for a in amts]
        assert all(0 <= x <= 1 for x in sc) and sc[-1] > sc[1] * 10


class TestErrorAttribution:
    def test_transaction_errors_are_labelled_as_transaction_not_a_history_row(self, scorer):
        with pytest.raises(CardInputError) as e:
            scorer.score_one(tx(1e10), hist())
        assert e.value.errors[0]["source"] == "transaction" and e.value.errors[0]["row"] is None
        h = hist(); h[1]["amount"] = 0
        with pytest.raises(CardInputError) as e2:
            scorer.score_one(tx(), h)
        assert e2.value.errors[0]["source"] == "history" and e2.value.errors[0]["row"] == 1
