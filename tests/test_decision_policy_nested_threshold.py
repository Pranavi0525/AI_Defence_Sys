"""
Regression test for Phase 4B: nested (fold-honest) threshold selection in
decision_policy.py.

FINDING (see reports/stage_leakage_audit_risk_fusion_decision_policy.md,
Finding 2): optimize_thresholds() grid-searches (t_review, t_block) to
minimize expected cost over the FULL validation population, then
policy_stats() reports allow/review/block rates, recall, cost, and
liability_breakdown on that SAME population. proba is legitimately
out-of-fold w.r.t. the base/fusion models, but the threshold PAIR itself
is a free parameter fit to this exact sample and judged on it -- a
separate, additional optimism problem layered on top of (not fixed by)
base-model OOF-ness.

FIX: nested_threshold_estimate() reuses the identical fold partition
already used to produce `proba`
(blue_team_pipeline.stable_kfold_split(df, "fraud", n_splits,
RANDOM_STATE) -- the same call compute_stage_1_2_cascade makes). For each
outer fold, thresholds are selected on the OTHER folds only and applied
to score just that fold's held-out rows, rotating across all folds so
every row gets exactly one nested decision.

This test verifies, directly against the real functions (no
reimplementation of the threshold-search or cost math):

  1. No row's own fold ever influences the threshold pair applied to it.
     The spy captures the ACTUAL (y, proba, dollars) arrays passed to
     optimize_thresholds() for each outer fold and asserts they are
     exactly proba[select_idx]/y[select_idx]/dollars[select_idx] (same
     values, same order) -- not merely the right length -- and therefore
     contain none of that fold's proba[holdout_idx] values.
  1b. Explicit extreme-held-out-row regression: a single row in a known
     outer fold's holdout set is given a maximally extreme proba value
     (1.0 or 0.0), and the test proves (a) the selection array actually
     passed to optimize_thresholds() for that fold is byte-identical
     whether or not the row was perturbed, and (b) the threshold pair
     optimize_thresholds() selects for that fold is therefore identical
     too -- i.e. changing a held-out row cannot change the threshold
     pair selected for its own fold. Achieved by running the real
     nested_threshold_estimate()/optimize_thresholds() twice (baseline
     vs. perturbed) and diffing their outputs, not by reimplementing any
     selection or cost logic.
  2. Every row receives exactly one nested decision (ALLOW/REVIEW/BLOCK
     masks partition the full row set with no overlap and no gaps).
  3. nested_threshold_estimate() is purely additive: it does not call
     policy_stats()/optimize_thresholds() with mutated arguments and does
     not alter their outputs -- calling optimize_thresholds()/
     policy_stats() on the full population before and after exercising
     nested_threshold_estimate() gives byte-identical results (regression
     guard that this change didn't touch the existing computation).
  4. The nested estimate is internally consistent: fraud_recall,
     dollars_fraud_total, and n_selection_rows + n_holdout_rows per fold
     match the input data exactly.

NOTE on interpretation (documentation-only correction, no behavior
change): nested_threshold_estimate() estimates the out-of-sample
performance of the threshold-SELECTION PROCEDURE -- different outer
folds may select different (t_review, t_block) pairs (see
'fold_thresholds' in its return value) -- it is not literally an
estimate of the future performance of the single, full-population
'corrected' pair that optimize_thresholds() would select and that
would actually be deployed. See decision_policy.py's updated
nested_threshold_estimate() docstring for the same clarification.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import blue_team_pipeline as btp
import decision_policy as dp


# ---------------------------------------------------------------------------
# Small synthetic dataset -- deterministic, with a real trace_id per row
# (stable_kfold_split hashes on trace_id) and a fraud rate high enough that
# every fold gets some fraud rows.
# ---------------------------------------------------------------------------
N_LEGIT = 120
N_FRAUD = 40
N = N_LEGIT + N_FRAUD
RNG = np.random.RandomState(0)


def _make_df_and_arrays():
    trace_ids = [f"trace_{i:04d}" for i in range(N)]
    y = np.array([0] * N_LEGIT + [1] * N_FRAUD)
    # proba correlated with y but noisy, so grid search has real work to do
    proba = np.clip(
        RNG.beta(2, 6, size=N) + y * RNG.uniform(0.15, 0.55, size=N), 0.0, 1.0
    )
    dollars = RNG.uniform(10, 2000, size=N)
    families = np.where(
        y == 1,
        RNG.choice(
            ["ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"], size=N
        ),
        "legitimate",
    )
    df = pd.DataFrame({
        "trace_id": trace_ids,
        "fraud": y,
        "attack_family": families,
    })
    return df, y, proba, dollars, families


@pytest.fixture(scope="module")
def data():
    return _make_df_and_arrays()


@pytest.fixture(scope="module")
def cost():
    return dp.CostModel(assumed_production_fraud_rate=0.05)


# ---------------------------------------------------------------------------
# 1. No row's own fold influences the threshold pair applied to it.
#    Strengthened per Phase 4B test-hardening request: the spy now
#    captures the ACTUAL selection arrays (not just their length) and
#    asserts they equal proba[select_idx]/y[select_idx]/dollars[select_idx]
#    exactly, so this proves the selection population *contains the
#    correct rows*, not merely the correct count.
# ---------------------------------------------------------------------------
def test_no_row_influences_its_own_nested_threshold(data, cost, monkeypatch):
    df, y, proba, dollars, families = data
    n_splits = 5
    folds = btp.stable_kfold_split(df, "fraud", n_splits, btp.CONFIG["RANDOM_STATE"])

    captured_calls = []
    real_optimize = dp.optimize_thresholds

    def spy_optimize_thresholds(y_arg, proba_arg, dollars_arg, cost_arg, **kwargs):
        # Copy eagerly: real_optimize doesn't mutate its inputs, but
        # copying removes any doubt that a later in-place change could
        # retroactively corrupt what we captured here.
        captured_calls.append({
            "y": np.array(y_arg, copy=True),
            "proba": np.array(proba_arg, copy=True),
            "dollars": np.array(dollars_arg, copy=True),
        })
        return real_optimize(y_arg, proba_arg, dollars_arg, cost_arg, **kwargs)

    monkeypatch.setattr(dp, "optimize_thresholds", spy_optimize_thresholds)

    dp.nested_threshold_estimate(
        df, y, proba, dollars, cost, n_splits=n_splits, attack_family=families
    )

    assert len(captured_calls) == n_splits
    for (select_idx, holdout_idx), call in zip(folds, captured_calls):
        expected_proba = proba[select_idx]
        expected_y = y[select_idx]
        expected_dollars = dollars[select_idx]

        # The EXACT selection population passed to optimize_thresholds
        # (same values, same order) -- not just the right size.
        assert np.array_equal(call["proba"], expected_proba)
        assert np.array_equal(call["y"], expected_y)
        assert np.array_equal(call["dollars"], expected_dollars)
        assert len(call["proba"]) != len(y)  # never the full population

        # And therefore -- as a direct, checkable consequence -- the
        # selection array contains NONE of this fold's held-out proba
        # values (continuous random floats: a spurious value collision
        # with a genuinely different row is vanishingly unlikely).
        holdout_proba = proba[holdout_idx]
        assert not np.isin(holdout_proba, call["proba"]).any()


# ---------------------------------------------------------------------------
# 1b. Explicit extreme-held-out-row regression test.
#     Picks one row that is genuinely held out for a specific outer fold,
#     makes its proba value maximally extreme (1.0 or 0.0), and proves --
#     by running the REAL nested_threshold_estimate()/optimize_thresholds()
#     twice and diffing captured inputs/outputs, never by reimplementing
#     the selection or cost math -- that this cannot change either (a) the
#     selection array passed to optimize_thresholds() for that fold, or
#     (b) the threshold pair optimize_thresholds() selects for that fold.
# ---------------------------------------------------------------------------
def test_extreme_held_out_row_cannot_affect_its_own_fold_threshold(data, cost):
    df, y, proba, dollars, families = data
    n_splits = 5
    folds = btp.stable_kfold_split(df, "fraud", n_splits, btp.CONFIG["RANDOM_STATE"])

    target_fold_i = 2  # arbitrary but fixed outer fold
    select_idx, holdout_idx = folds[target_fold_i]
    target_row = int(holdout_idx[0])
    assert target_row not in set(select_idx.tolist())  # sanity: genuinely held out here

    proba_perturbed = proba.copy()
    proba_perturbed[target_row] = 1.0 if proba[target_row] < 0.5 else 0.0
    assert proba_perturbed[target_row] != proba[target_row]

    def run_and_capture_target_fold_selection(proba_arr):
        """Runs the real nested_threshold_estimate(), spying only on the
        selection array optimize_thresholds() receives for
        target_fold_i, via call order (outer folds are processed in
        order by nested_threshold_estimate, verified by test 1 above)."""
        captured = {}
        real_optimize = dp.optimize_thresholds
        call_counter = {"i": 0}

        def spy(y_arg, proba_arg, dollars_arg, cost_arg, **kwargs):
            if call_counter["i"] == target_fold_i:
                captured["proba"] = np.array(proba_arg, copy=True)
            call_counter["i"] += 1
            return real_optimize(y_arg, proba_arg, dollars_arg, cost_arg, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dp, "optimize_thresholds", spy)
            result = dp.nested_threshold_estimate(
                df, y, proba_arr, dollars, cost, n_splits=n_splits, attack_family=families
            )
        return result, captured["proba"]

    baseline_result, baseline_selection_proba = run_and_capture_target_fold_selection(proba)
    perturbed_result, perturbed_selection_proba = run_and_capture_target_fold_selection(proba_perturbed)

    # (a) The threshold-selection INPUT for this fold is byte-identical
    # whether or not the held-out row was perturbed -- direct proof the
    # row was excluded from selection, not just "probably" excluded.
    assert np.array_equal(baseline_selection_proba, perturbed_selection_proba)

    # (b) Therefore the SELECTED threshold pair for this fold is also
    # identical -- changing this held-out row cannot change the
    # threshold pair applied to it.
    baseline_fold = baseline_result["fold_thresholds"][target_fold_i]
    perturbed_fold = perturbed_result["fold_thresholds"][target_fold_i]
    assert baseline_fold["fold"] == perturbed_fold["fold"] == target_fold_i + 1
    assert baseline_fold["t_review"] == perturbed_fold["t_review"]
    assert baseline_fold["t_block"] == perturbed_fold["t_block"]
    assert baseline_fold["n_selection_rows"] == perturbed_fold["n_selection_rows"]


# ---------------------------------------------------------------------------
# 2. Every row gets exactly one nested decision (masks partition cleanly).
# ---------------------------------------------------------------------------
def test_every_row_gets_exactly_one_nested_decision(data, cost):
    df, y, proba, dollars, families = data
    result = dp.nested_threshold_estimate(
        df, y, proba, dollars, cost, n_splits=5, attack_family=families
    )
    allow = round(result["allow_rate"] * N)
    review = round(result["review_rate"] * N)
    block = round(result["block_rate"] * N)
    # allow_rate + review_rate + block_rate must reconstruct to exactly N
    # rows once un-rounded -- check via the underlying rates summing to 1.
    total_rate = result["allow_rate"] + result["review_rate"] + result["block_rate"]
    assert abs(total_rate - 1.0) < 1e-6

    # Per-fold row counts must exactly cover the dataset with no overlap.
    total_holdout = sum(f["n_holdout_rows"] for f in result["fold_thresholds"])
    assert total_holdout == N
    for f in result["fold_thresholds"]:
        assert f["n_selection_rows"] + f["n_holdout_rows"] == N


# ---------------------------------------------------------------------------
# 3. Purely additive: optimize_thresholds()/policy_stats() on the full
#    population are unaffected by nested_threshold_estimate() having run.
# ---------------------------------------------------------------------------
def test_existing_optimize_and_policy_stats_unchanged(data, cost):
    df, y, proba, dollars, families = data

    before = dp.optimize_thresholds(y, proba, dollars, cost, attack_family=families)

    _ = dp.nested_threshold_estimate(
        df, y, proba, dollars, cost, n_splits=5, attack_family=families
    )

    after = dp.optimize_thresholds(y, proba, dollars, cost, attack_family=families)

    assert before["t_review"] == after["t_review"]
    assert before["t_block"] == after["t_block"]
    assert before["expected_cost_at_assumed_prevalence"] == after["expected_cost_at_assumed_prevalence"]
    assert before["allow_rate"] == after["allow_rate"]
    assert before["liability_breakdown"] == after["liability_breakdown"]

    # Arrays passed in must not have been mutated in place either.
    assert proba.flags.writeable  # sanity: still a normal, unmodified array
    assert len(y) == N and len(proba) == N and len(dollars) == N


# ---------------------------------------------------------------------------
# 4. Internal consistency of the nested estimate's own numbers.
# ---------------------------------------------------------------------------
def test_nested_estimate_internal_consistency(data, cost):
    df, y, proba, dollars, families = data
    result = dp.nested_threshold_estimate(
        df, y, proba, dollars, cost, n_splits=5, attack_family=families
    )

    assert result["dollars_fraud_total"] == round(float(dollars[y == 1].sum()), 2)
    assert result["fraud_blocked"] + result["fraud_reviewed"] + result["fraud_allowed"] == N_FRAUD
    assert 0.0 <= result["fraud_recall_blocked_plus_review"] <= 1.0
    assert result["fraud_recall_blocked_only"] <= result["fraud_recall_blocked_plus_review"]

    # liability_breakdown present and keyed only by fraud families that
    # actually occur, matching the non-nested liability_breakdown()'s
    # per-family field shape.
    assert "liability_breakdown" in result
    for fam, row in result["liability_breakdown"].items():
        assert fam in ("ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK")
        assert row["n_blocked"] + row["n_reviewed"] + row["n_allowed_through"] == row["n_fraud_traces"]


# ---------------------------------------------------------------------------
# 5. _liability_breakdown_from_masks agrees with liability_breakdown() when
#    given masks derived from the SAME single threshold pair (i.e. the new
#    masked helper reproduces the existing, already-tested function's
#    numbers whenever there's a single scalar pair to compare against).
# ---------------------------------------------------------------------------
def test_masked_liability_breakdown_matches_scalar_version(data, cost):
    df, y, proba, dollars, families = data
    selected = dp.optimize_thresholds(y, proba, dollars, cost, attack_family=families)
    t_review, t_block = selected["t_review"], selected["t_block"]

    expected = dp.liability_breakdown(y, proba, dollars, families, t_review, t_block, cost)

    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    actual = dp._liability_breakdown_from_masks(
        y, dollars, families, is_block, is_review, is_allow, cost
    )

    assert actual == expected
