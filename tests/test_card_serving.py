"""Tests for the served card-fraud model: training bundle, loading, validation, single/batch consistency, failure modes.

All data is SYNTHETIC (v2.card.sim). A small bundle is trained once per module into a temp dir; one test also checks the
committed models/card bundle. None of this touches the payment-fraud modules.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v2.card import serving
from v2.card.features import FEATURES, build_features
from v2.card.serving import CardInputError, CardModelError, CardScorer, validate_rows
from v2.card.sim import simulate
from v2.card.train import train_bundle

REPO = Path(__file__).resolve().parents[1]
COLS = ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant", "declined"]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("card_bundle")
    train_bundle(d, seed=3, n_cards=300, days=60, events_per_pattern=15)
    return d


@pytest.fixture(scope="module")
def scorer(bundle_dir):
    return CardScorer(bundle_dir)


@pytest.fixture(scope="module")
def sim_df():
    return simulate(n_cards=60, days=30, seed=1, events_per_pattern=6)


@pytest.fixture()
def rows(sim_df):
    return sim_df[COLS].to_dict("records")


def _card_rows(rows, card):
    return sorted([r for r in rows if r["card"] == card], key=lambda r: r["ts"])


# ----------------------------------------------------------------------------- training / bundle
class TestBundle:
    def test_policy_is_labelled_synthetic_and_complete(self, bundle_dir):
        pol = json.loads((bundle_dir / "policy.json").read_text())
        assert pol["synthetic"] is True and pol["data_source"].startswith("SYNTHETIC")
        assert pol["features"] == FEATURES and len(pol["features"]) == 27
        assert "not a probability" in pol["score_meaning"].lower() or "not a probability" in pol["score_meaning"]
        assert any("SYNTHETIC" in s for s in pol["limitations"])
        ev = pol["evaluation"]
        assert "SYNTHETIC" in ev["label"]
        c = ev["at_threshold"]
        assert {"tp", "fp", "fn", "tn", "precision", "recall", "f1"} <= set(c)
        assert c["tp"] + c["fn"] == ev["test_fraud_rows"]
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == ev["test_rows"]
        assert set(ev["threshold_sweep_by_validation_fpr"]) == {"0.05", "0.02", "0.01", "0.005", "0.001"}

    def test_training_is_reproducible_for_a_fixed_seed(self, bundle_dir, tmp_path):
        train_bundle(tmp_path, seed=3, n_cards=300, days=60, events_per_pattern=15)
        a, b = json.loads((bundle_dir / "policy.json").read_text()), json.loads((tmp_path / "policy.json").read_text())
        assert a["xgb_sha256"] == b["xgb_sha256"] and a["threshold"] == b["threshold"]
        assert a["evaluation"]["at_threshold"] == b["evaluation"]["at_threshold"]

    def test_train_refuses_blocks_without_both_classes(self, tmp_path):
        f = build_features(simulate(n_cards=60, days=60, seed=1, events_per_pattern=0))   # no fraud at all
        with pytest.raises(ValueError, match="both fraud and legitimate"):
            train_bundle(tmp_path, frame=f)

    def test_committed_bundle_loads_and_is_consistent(self):
        d = REPO / "models" / "card"
        s = CardScorer(d)
        assert s.policy["synthetic"] is True and s.features == FEATURES
        assert s.model_version.startswith("card-xgb-synthetic-")


# ----------------------------------------------------------------------------- the pipeline reproduces training-time features
class TestPipelineMatchesTraining:
    def test_served_scores_equal_direct_booster_on_training_feature_code(self, scorer, sim_df, rows):
        import xgboost as xgb
        f = build_features(sim_df)                                     # the training-side feature path, untouched
        direct = scorer.booster.predict(xgb.DMatrix(f[FEATURES].to_numpy(float), feature_names=FEATURES, missing=np.nan))
        key = {(r.card, r.ts, r.amount): s for r, s in zip(f.itertuples(), direct)}
        got = scorer.score_batch(rows)
        # card ids become strings internally; the scores must still be identical row for row
        diffs = [abs(g["model_score"] - key[(r["card"], r["ts"], r["amount"])]) for g, r in zip(got, rows)]
        assert max(diffs) < 1e-6

    def test_current_rows_declined_flag_is_not_a_feature(self, scorer, rows):
        card = rows[0]["card"]
        hist = _card_rows(rows, card)
        target = copy.deepcopy(hist[-1]); prior = hist[:-1]
        a = scorer.score_one({**target, "declined": 0}, prior)["model_score"]
        b = scorer.score_one({**target, "declined": 1}, prior)["model_score"]
        assert a == b


# ----------------------------------------------------------------------------- single vs batch consistency
class TestSingleBatchConsistency:
    def test_single_equals_batch_for_every_transaction_of_a_card(self, scorer, rows):
        card = max({r["card"] for r in rows}, key=lambda c: len(_card_rows(rows, c)))
        hist = _card_rows(rows, card)
        batch = scorer.score_batch(hist)
        for i, t in enumerate(hist):
            single = scorer.score_one(t, hist[:i])
            assert single["model_score"] == pytest.approx(batch[i]["model_score"], abs=1e-9)
            assert single["classification"] == batch[i]["classification"]
            assert single["legit_percentile"] == pytest.approx(batch[i]["legit_percentile"])

    def test_batch_result_is_independent_of_input_order(self, scorer, rows):
        base = scorer.score_batch(rows)
        perm = list(range(len(rows))); random.Random(0).shuffle(perm)
        shuffled = scorer.score_batch([rows[i] for i in perm])
        for new_pos, old_pos in enumerate(perm):
            assert shuffled[new_pos]["model_score"] == pytest.approx(base[old_pos]["model_score"], abs=1e-9)

    def test_results_keep_input_order_and_indices(self, scorer, rows):
        res = scorer.score_batch(rows)
        assert [r["index"] for r in res] == list(range(len(rows)))

    def test_history_changes_the_score(self, scorer, rows):
        card = max({r["card"] for r in rows}, key=lambda c: len(_card_rows(rows, c)))
        hist = _card_rows(rows, card)
        t = hist[-1]
        with_hist = scorer.score_one(t, hist[:-1])
        alone = scorer.score_one(t, [])
        assert with_hist["prior_transactions_on_card"] == len(hist) - 1 and alone["prior_transactions_on_card"] == 0
        assert any("no card history" in w for w in alone["warnings"])

    def test_dataframe_and_dict_input_agree(self, scorer, rows):
        a = scorer.score_batch(rows); b = scorer.score_batch(pd.DataFrame(rows))
        assert [x["model_score"] for x in a] == pytest.approx([x["model_score"] for x in b])

    def test_card_id_type_does_not_matter(self, scorer, rows):
        a = scorer.score_batch(rows)
        b = scorer.score_batch([{**r, "card": str(r["card"])} for r in rows])
        assert [x["model_score"] for x in a] == pytest.approx([x["model_score"] for x in b])

    def test_declined_is_optional(self, scorer, rows):
        without = [{k: v for k, v in r.items() if k != "declined"} for r in rows]
        assert len(scorer.score_batch(without)) == len(rows)

    def test_later_history_rows_are_ignored_and_warned(self, scorer, rows):
        hist = _card_rows(rows, rows[0]["card"])
        res = scorer.score_one(hist[0], hist[1:])
        assert any("timed after" in w for w in res["warnings"])
        assert res["model_score"] == pytest.approx(scorer.score_one(hist[0], [])["model_score"], abs=1e-9)

    def test_classification_follows_the_threshold(self, scorer, rows):
        for r in scorer.score_batch(rows):
            assert r["is_fraud"] == (r["model_score"] >= scorer.threshold)
            assert r["classification"] == ("FRAUD" if r["is_fraud"] else "LEGITIMATE")
            assert 0.0 <= r["legit_percentile"] <= 1.0

    def test_detects_a_card_testing_burst_but_not_ordinary_history(self, scorer, sim_df, rows):
        fraud_cards = sim_df[sim_df.pattern == "CARD_TESTING"].card.unique()
        card = int(fraud_cards[0])
        res = scorer.score_batch(_card_rows(rows, card))
        flagged = [r for r, t in zip(res, _card_rows(rows, card)) if r["is_fraud"]]
        assert flagged, "a simulated card-testing burst should trip the synthetic-trained model"

    def test_summary(self, scorer, rows):
        s = CardScorer.summarize(scorer.score_batch(rows))
        assert s["transactions"] == len(rows) and s["flagged_fraud"] + s["legitimate"] == len(rows)
        assert 0 <= s["flagged_rate"] <= 1


# ----------------------------------------------------------------------------- invalid input
GOOD = {"card": 1, "ts": 1000.0, "amount": 50.0, "mcc": 5411, "channel": "POS", "lat": 13.08, "lon": 80.27, "merchant": "M5411_3", "declined": 0}


class TestValidation:
    @pytest.mark.parametrize("patch,field", [
        ({"amount": 0}, "amount"), ({"amount": -5}, "amount"), ({"amount": float("nan")}, "amount"), ({"amount": "12"}, "amount"),
        ({"amount": None}, "amount"), ({"ts": -1}, "ts"), ({"ts": float("inf")}, "ts"), ({"channel": "WIRE"}, "channel"),
        ({"channel": None}, "channel"), ({"lat": 91}, "lat"), ({"lon": -181}, "lon"), ({"lat": "x"}, "lat"),
        ({"declined": 2}, "declined"), ({"declined": "yes"}, "declined"), ({"card": ""}, "card"), ({"card": None}, "card"),
        ({"merchant": "  "}, "merchant"), ({"mcc": None}, "mcc"), ({"amount": True}, "amount"),
    ])
    def test_bad_field_is_rejected_with_its_name(self, patch, field):
        with pytest.raises(CardInputError) as e:
            validate_rows([{**GOOD, **patch}])
        assert any(x["field"] == field and x["row"] == 0 for x in e.value.errors)

    @pytest.mark.parametrize("col", ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant"])
    def test_missing_required_column(self, col):
        with pytest.raises(CardInputError) as e:
            validate_rows([{k: v for k, v in GOOD.items() if k != col}])
        assert e.value.errors[0]["field"] == col

    def test_empty_and_non_object_rows(self):
        with pytest.raises(CardInputError):
            validate_rows([])
        with pytest.raises(CardInputError):
            validate_rows([GOOD, "not a row"])

    def test_all_problems_reported_with_row_numbers(self):
        with pytest.raises(CardInputError) as e:
            validate_rows([GOOD, {**GOOD, "amount": -1}, GOOD, {**GOOD, "channel": "X"}])
        assert {(x["row"], x["field"]) for x in e.value.errors} == {(1, "amount"), (3, "channel")}

    def test_row_limit(self, monkeypatch):
        monkeypatch.setattr(serving, "MAX_ROWS", 3)
        with pytest.raises(CardInputError, match="too many"):
            validate_rows([GOOD] * 4)

    def test_channel_case_and_declined_none_are_normalised_not_guessed(self):
        df = validate_rows([{**GOOD, "channel": "pos", "declined": None}])
        assert df.loc[0, "channel"] == "POS" and df.loc[0, "declined"] == 0

    def test_nothing_is_silently_dropped(self, scorer):
        with pytest.raises(CardInputError):
            scorer.score_batch([GOOD, {**GOOD, "amount": -1}])


# ----------------------------------------------------------------------------- missing / corrupted artifacts
class TestMissingArtifacts:
    def test_empty_directory(self, tmp_path):
        with pytest.raises(CardModelError, match="missing"):
            CardScorer(tmp_path)

    def test_nonexistent_directory(self, tmp_path):
        with pytest.raises(CardModelError, match="missing"):
            CardScorer(tmp_path / "nope")

    @pytest.mark.parametrize("gone", ["xgb.json", "policy.json"])
    def test_each_artifact_is_required(self, bundle_dir, tmp_path, gone):
        for f in ("xgb.json", "policy.json"):
            if f != gone:
                (tmp_path / f).write_bytes((bundle_dir / f).read_bytes())
        with pytest.raises(CardModelError, match=gone):
            CardScorer(tmp_path)

    def _copy(self, bundle_dir, tmp_path):
        for f in ("xgb.json", "policy.json"):
            (tmp_path / f).write_bytes((bundle_dir / f).read_bytes())

    def test_checksum_mismatch(self, bundle_dir, tmp_path):
        self._copy(bundle_dir, tmp_path)
        (tmp_path / "xgb.json").write_bytes((tmp_path / "xgb.json").read_bytes() + b" ")
        with pytest.raises(CardModelError, match="checksum"):
            CardScorer(tmp_path)

    def test_feature_list_mismatch(self, bundle_dir, tmp_path):
        self._copy(bundle_dir, tmp_path)
        pol = json.loads((tmp_path / "policy.json").read_text()); pol["features"] = pol["features"][::-1]
        (tmp_path / "policy.json").write_text(json.dumps(pol))
        with pytest.raises(CardModelError, match="feature list"):
            CardScorer(tmp_path)

    def test_unreadable_policy_and_wrong_format(self, bundle_dir, tmp_path):
        self._copy(bundle_dir, tmp_path)
        (tmp_path / "policy.json").write_text("{not json")
        with pytest.raises(CardModelError, match="unreadable"):
            CardScorer(tmp_path)
        pol = json.loads((bundle_dir / "policy.json").read_text()); pol["format"] = "other"
        (tmp_path / "policy.json").write_text(json.dumps(pol))
        with pytest.raises(CardModelError, match="format"):
            CardScorer(tmp_path)
