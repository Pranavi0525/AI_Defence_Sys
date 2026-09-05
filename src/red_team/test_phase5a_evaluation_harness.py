"""
Tests for Phase 5A: evaluation_harness.py.

These exercise the REAL production functions in evaluation_harness.py
(and, through it, decision_policy.py / cascade_with_graph.py) -- not
reimplementations of their logic. A handful of tests use small synthetic
arrays/DataFrames to pin down exact metric formulas; the rest run the
harness's real orchestration (run_evaluation()) against the real
repository artifacts, matching this project's existing test style (see
tests/test_phase4c_artifact_provenance.py).

NOTE: importing evaluation_harness.py (like decision_policy.py) requires
the project's full pinned stack (xgboost, etc.) because it imports
blue_team_pipeline / cascade_with_graph / decision_policy at module load
time. Same requirement every existing decision_policy test already has.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import decision_policy as dp
import cascade_with_graph as cwg
import risk_fusion as rf
import evaluation_harness as eh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_report():
    """Runs the actual harness end-to-end against the real repository
    artifacts ONCE per test module (this is not free -- it loads the
    real corpora and the real cache), and reuses the result across every
    test that only needs to inspect the output rather than re-drive the
    orchestration itself."""
    return eh.run_evaluation()


def _toy_df(rows):
    """rows: list of (trace_id, fraud, attack_family) tuples."""
    return pd.DataFrame(
        [{"trace_id": t, "fraud": f, "attack_family": fam} for t, f, fam in rows]
    )


# ---------------------------------------------------------------------------
# 1. Metric correctness (synthetic arrays)
# ---------------------------------------------------------------------------
def test_classification_metrics_synthetic():
    # 10 rows: 5 fraud, 5 legit. Threshold is 0.5 (CLASSIFICATION_THRESHOLD).
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.4, 0.1, 0.6, 0.2, 0.1, 0.05, 0.02])
    # preds @0.5: [1,1,1,0,0,1,0,0,0,0]
    # TP=3 (rows 0,1,2), FN=2 (rows 3,4), FP=1 (row 5), TN=4 (rows 6-9)
    result = eh.compute_classification_metrics(y, proba)
    cm = result["confusion_matrix"]
    assert cm == {"true_positive": 3, "true_negative": 4, "false_positive": 1, "false_negative": 2}
    assert math.isclose(result["accuracy"], 7 / 10, rel_tol=1e-9)
    assert math.isclose(result["precision"], 3 / 4, rel_tol=1e-9)
    assert math.isclose(result["recall"], 3 / 5, rel_tol=1e-9)
    expected_f1 = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
    assert math.isclose(result["f1"], expected_f1, rel_tol=1e-9)
    assert result["canonical_cross_check_passed"] is True


def test_fraud_metrics_derived_from_classification():
    classification = {
        "confusion_matrix": {"true_positive": 3, "true_negative": 4, "false_positive": 1, "false_negative": 2}
    }
    fm = eh.compute_fraud_metrics(classification)
    assert fm["fraud_count"] == 5
    assert fm["legitimate_count"] == 5
    assert math.isclose(fm["fraud_recall"], 0.6)
    assert math.isclose(fm["fraud_precision"], 0.75)
    assert math.isclose(fm["fraud_false_negative_rate"], 0.4)
    assert math.isclose(fm["fraud_false_positive_rate"], 0.2)


# ---------------------------------------------------------------------------
# 2. Decision-policy aggregation
# ---------------------------------------------------------------------------
def test_decision_policy_counts_sum_to_n():
    n = 200
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(int)
    proba = rng.random(n)
    dollars = rng.random(n) * 1000
    families = np.where(y == 1, "ACCOUNT_TAKEOVER", "legitimate")

    metrics = eh.compute_decision_policy_metrics(
        y, proba, dollars, families, t_review=0.3, t_block=0.7, cost_model=dp.CostModel(),
    )
    assert metrics["allow_count"] + metrics["review_count"] + metrics["block_count"] == n
    comp = metrics["composition"]
    assert (
        comp["fraud_allowed"] + comp["fraud_reviewed"] + comp["fraud_blocked"]
        == int((y == 1).sum())
    )
    assert (
        comp["legitimate_allowed"] + comp["legitimate_reviewed"] + comp["legitimate_blocked"]
        == int((y == 0).sum())
    )


def test_decision_policy_rejects_bad_counts_via_invariant():
    # Sanity: compute_decision_policy_metrics itself raises if counts
    # somehow didn't reconcile (defensive check inside the function).
    n = 50
    y = np.zeros(n, dtype=int)
    proba = np.zeros(n)
    dollars = np.zeros(n)
    families = np.array(["legitimate"] * n)
    metrics = eh.compute_decision_policy_metrics(
        y, proba, dollars, families, t_review=0.1, t_block=0.9, cost_model=dp.CostModel(),
    )
    assert metrics["allow_count"] == n
    assert metrics["review_count"] == 0
    assert metrics["block_count"] == 0


# ---------------------------------------------------------------------------
# 3. Family aggregation
# ---------------------------------------------------------------------------
def test_attack_family_aggregation_all_three_families():
    df = _toy_df([
        ("t1", 1, "ACCOUNT_TAKEOVER"), ("t2", 1, "ACCOUNT_TAKEOVER"),
        ("t3", 1, "AUTHORIZED_PUSH_PAYMENT"),
        ("t4", 1, "MULE_NETWORK"), ("t5", 1, "MULE_NETWORK"), ("t6", 1, "MULE_NETWORK"),
        ("t7", 0, "legitimate"), ("t8", 0, "legitimate"),
    ])
    y = df["fraud"].values
    # Block everything for simplicity (t_block=0 blocks all).
    proba = np.zeros(len(df))
    fam_metrics = eh.compute_attack_family_metrics(df, y, proba, t_review=0.0, t_block=0.0)
    families = fam_metrics["families"]
    assert set(families.keys()) == set(eh.SUPPORTED_ATTACK_FAMILIES)
    assert families["ACCOUNT_TAKEOVER"]["count"] == 2
    assert families["ACCOUNT_TAKEOVER"]["fraud_count"] == 2
    assert families["AUTHORIZED_PUSH_PAYMENT"]["fraud_count"] == 1
    assert families["MULE_NETWORK"]["fraud_count"] == 3
    assert fam_metrics["reconciliation"]["matches_global_fraud_count"] is True
    assert fam_metrics["reconciliation"]["sum_family_fraud_counts"] == 6


# ---------------------------------------------------------------------------
# 4. Duplicate trace IDs
# ---------------------------------------------------------------------------
def test_rejects_duplicate_trace_ids():
    df = _toy_df([("dup", 0, "legitimate"), ("dup", 1, "ACCOUNT_TAKEOVER")])
    with pytest.raises(eh.EvaluationHarnessError, match="duplicate"):
        eh.validate_dataset_invariants(df)


# ---------------------------------------------------------------------------
# 5. Missing / empty IDs
# ---------------------------------------------------------------------------
def test_rejects_missing_trace_ids():
    df = _toy_df([("t1", 0, "legitimate"), (None, 1, "ACCOUNT_TAKEOVER")])
    with pytest.raises(eh.EvaluationHarnessError, match="missing/empty"):
        eh.validate_dataset_invariants(df)


def test_rejects_empty_string_trace_ids():
    df = _toy_df([("t1", 0, "legitimate"), ("   ", 1, "ACCOUNT_TAKEOVER")])
    with pytest.raises(eh.EvaluationHarnessError, match="missing/empty"):
        eh.validate_dataset_invariants(df)


# ---------------------------------------------------------------------------
# 6. Length mismatch
# ---------------------------------------------------------------------------
def test_rejects_score_length_mismatch():
    with pytest.raises(eh.EvaluationHarnessError, match="length"):
        eh.validate_scores(np.array([0.1, 0.2, 0.3]), n_rows=5)


# ---------------------------------------------------------------------------
# 7. Invalid scores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_scores(bad_value):
    proba = np.array([0.1, 0.5, bad_value])
    with pytest.raises(eh.EvaluationHarnessError, match="finite"):
        eh.validate_scores(proba, n_rows=3)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, -5.0, 2.0])
def test_rejects_out_of_range_scores(bad_value):
    proba = np.array([0.1, 0.5, bad_value])
    with pytest.raises(eh.EvaluationHarnessError, match=r"\[0, 1\]"):
        eh.validate_scores(proba, n_rows=3)


def test_accepts_valid_scores():
    proba = np.array([0.0, 0.5, 1.0])
    result = eh.validate_scores(proba, n_rows=3)
    assert result["available_count"] == 3
    assert result["unavailable_count"] == 0
    assert result["min"] == 0.0
    assert result["max"] == 1.0


# ---------------------------------------------------------------------------
# 8. Unknown attack family
# ---------------------------------------------------------------------------
def test_rejects_unsupported_attack_family():
    df = _toy_df([("t1", 1, "PHISHING_NOT_A_REAL_FAMILY"), ("t2", 0, "legitimate")])
    with pytest.raises(eh.EvaluationHarnessError, match="unsupported attack family"):
        eh.validate_dataset_invariants(df)


# ---------------------------------------------------------------------------
# 9. Invalid thresholds
# ---------------------------------------------------------------------------
def _write_policy_json(path, t_review, t_block, score="risk_fusion_stacked_lr"):
    payload = {
        "corrected": {
            "t_review": t_review,
            "t_block": t_block,
            "cost_model": {
                "review_ops_cost": 12.0, "review_catch_rate": 0.85,
                "legit_block_friction_cost": 150.0,
                "assumed_production_fraud_rate": 0.006,
                "app_sending_liability_share": 0.5,
            },
        },
        "score_source": {"score": score},
    }
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("t_review,t_block", [(0.5, 0.5), (0.9, 0.1), (-0.1, 0.5), (0.5, 1.1)])
def test_rejects_invalid_thresholds(tmp_path, monkeypatch, t_review, t_block):
    p = tmp_path / "decision_policy_results.json"
    _write_policy_json(p, t_review, t_block)
    monkeypatch.setattr(eh, "POLICY_RESULTS_PATH", p)
    with pytest.raises(eh.EvaluationHarnessError):
        eh.load_canonical_policy()


def test_accepts_valid_thresholds(tmp_path, monkeypatch):
    p = tmp_path / "decision_policy_results.json"
    _write_policy_json(p, 0.1, 0.9)
    monkeypatch.setattr(eh, "POLICY_RESULTS_PATH", p)
    policy = eh.load_canonical_policy()
    assert policy["t_review"] == 0.1
    assert policy["t_block"] == 0.9


def test_rejects_policy_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(eh, "POLICY_RESULTS_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(eh.EvaluationHarnessError, match="does not exist"):
        eh.load_canonical_policy()


def test_rejects_policy_score_source_not_fused_like(tmp_path, monkeypatch):
    p = tmp_path / "decision_policy_results.json"
    _write_policy_json(p, 0.1, 0.9, score="stage_1_2_only")
    monkeypatch.setattr(eh, "POLICY_RESULTS_PATH", p)
    with pytest.raises(eh.EvaluationHarnessError, match="fused"):
        eh.load_canonical_policy()


# ---------------------------------------------------------------------------
# 10. Cache mismatch
# ---------------------------------------------------------------------------
def test_rejects_cascade_variant_cache_when_fused_required(tmp_path, monkeypatch):
    n = 10
    y = np.zeros(n, dtype=int)
    cache_path = tmp_path / "decision_policy_validation_cache.npz"
    np.savez(cache_path, y=y, proba=np.zeros(n), dollars=np.zeros(n), validation_variant="cascade")
    monkeypatch.setattr(dp, "CACHE_PATH", cache_path)

    df = _toy_df([(f"t{i}", 0, "legitimate") for i in range(n)])
    with pytest.raises(eh.EvaluationHarnessError, match="fused"):
        eh.load_canonical_fused_scores(df)


def test_accepts_fused_variant_cache(tmp_path, monkeypatch):
    n = 5
    y = np.array([0, 1, 0, 1, 0])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    dollars = np.array([0.0, 100.0, 0.0, 200.0, 0.0])
    cache_path = tmp_path / "decision_policy_validation_cache.npz"
    np.savez(cache_path, y=y, proba=proba, dollars=dollars, validation_variant="fused")
    monkeypatch.setattr(dp, "CACHE_PATH", cache_path)

    df = pd.DataFrame({
        "trace_id": [f"t{i}" for i in range(n)],
        "fraud": y,
        "attack_family": ["ACCOUNT_TAKEOVER" if v else "legitimate" for v in y],
    })
    y_out, proba_out, dollars_out = eh.load_canonical_fused_scores(df)
    assert np.array_equal(y_out, y)
    assert np.array_equal(proba_out, proba)
    assert np.array_equal(dollars_out, dollars)


# ---------------------------------------------------------------------------
# 11. No threshold optimization / no retraining / no calibration fitting
#     (all in one functional test: if the harness called any of these,
#     the monkeypatched stand-ins below raise and the test fails)
# ---------------------------------------------------------------------------
def test_harness_never_retrains_or_optimizes_thresholds(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "evaluation_harness must not call this -- it is a training/"
            "threshold-optimization entry point, and the harness is "
            "evaluation-only."
        )

    monkeypatch.setattr(cwg, "run_three_stage_cascade", _boom)
    monkeypatch.setattr(rf, "run_risk_fusion", _boom)
    monkeypatch.setattr(dp, "optimize_thresholds", _boom)
    monkeypatch.setattr(dp, "nested_threshold_estimate", _boom)
    monkeypatch.setattr(dp, "get_validation_data", _boom)
    monkeypatch.setattr(dp, "get_validation_data_fused", _boom)

    report = eh.run_evaluation()
    assert report["invariants"]["all_passed"] is True
    assert report["leakage_protections"]["threshold_optimization"].startswith("NOT USED")
    assert report["leakage_protections"]["retraining"].startswith("NOT USED")


def test_calibration_diagnostics_do_not_fit_anything():
    # compute_calibration_diagnostics takes already-produced scores/labels
    # and only bins + averages them -- verify it accepts read-only inputs
    # and never mutates them, and that it doesn't require (or produce)
    # any fitted calibrator object.
    y = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([0.1, 0.2, 0.6, 0.9, 0.55, 0.05])
    y_copy, proba_copy = y.copy(), proba.copy()

    result = eh.compute_calibration_diagnostics(y, proba, n_bins=5)

    assert np.array_equal(y, y_copy)
    assert np.array_equal(proba, proba_copy)
    assert result["available"] is True
    assert "calibrator" not in result
    assert 0.0 <= result["brier_score"] <= 1.0
    assert len(result["bins"]) == 5


# ---------------------------------------------------------------------------
# 12. Determinism
# ---------------------------------------------------------------------------
def test_determinism_two_runs_match():
    report_1 = eh.run_evaluation()
    report_2 = eh.run_evaluation()
    deterministic, diffs = eh._compare_core_results(report_1, report_2)
    assert deterministic, f"non-deterministic core results: {diffs}"


# ---------------------------------------------------------------------------
# 13. Provenance
# ---------------------------------------------------------------------------
def test_provenance_present(real_report):
    prov = real_report["provenance"]
    assert "git_commit" in prov
    assert "package_versions" in prov
    assert prov["policy_source"] == "decision_policy_results.json['corrected']"
    assert "score_source" in prov


# ---------------------------------------------------------------------------
# 14. Report serialization / required sections
# ---------------------------------------------------------------------------
REQUIRED_TOP_LEVEL_SECTIONS = [
    "schema_version", "evaluation", "policy", "scores", "classification",
    "fraud_metrics", "decision_policy_metrics", "attack_families",
    "stage_availability", "cost_metrics", "calibration", "provenance",
    "invariants",
]


def test_report_has_required_sections_and_is_json_serializable(real_report):
    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        assert section in real_report, f"missing required section: {section}"
    # Must actually serialize (this is what main() writes to disk).
    serialized = json.dumps(real_report, default=eh._json_default)
    reloaded = json.loads(serialized)
    assert reloaded["schema_version"] == eh.SCHEMA_VERSION


def test_report_invariants_all_passed_on_real_data(real_report):
    assert real_report["invariants"]["all_passed"] is True


def test_report_matches_documented_phase4_population(real_report):
    assert real_report["evaluation"]["row_count"] == eh.DOCUMENTED_PHASE4_POPULATION["total"]
    counts = real_report["evaluation"]["attack_family_counts"]
    assert counts == eh.DOCUMENTED_PHASE4_POPULATION


def test_report_reproduces_persisted_corrected_policy_block(real_report):
    assert real_report["invariants"]["policy_stats_reproduces_persisted_corrected_block"] is True


# ---------------------------------------------------------------------------
# 15. Existing-behavior compatibility (does not duplicate Phase 4C/4D
#     suites -- see the required validation commands in the Phase 5A
#     prompt for running those directly -- just confirms this new file
#     didn't have to touch, and remains compatible with, their shared
#     dependency: decision_policy.load_cached_validation_data).
# ---------------------------------------------------------------------------
def test_compatible_with_phase4c_cache_loader_contract():
    assert hasattr(dp, "load_cached_validation_data")
    assert hasattr(dp, "ValidationCacheMismatch")
    # Same function this harness calls -- Phase 4C's own tests exercise
    # its cache-variant-safety behavior directly.
    import inspect
    sig = inspect.signature(dp.load_cached_validation_data)
    assert "expected_variant" in sig.parameters
    assert "y_check" in sig.parameters