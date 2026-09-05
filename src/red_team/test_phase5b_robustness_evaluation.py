"""
Tests for Phase 5B: phase5b_robustness_evaluation.py.

Like src/red_team/test_phase5a_evaluation_harness.py, these exercise the
REAL production functions (and, through them, evaluation_harness.py /
decision_policy.py / risk_fusion.py / gcn.py / autoencoder.py) rather
than reimplementations of their logic. A `real_report` module-scoped
fixture runs the actual evaluator once and is reused by every test that
only needs to inspect its output; a handful of tests intentionally
monkeypatch real repository entry points to prove the hardened
invariants (threshold verification, canonical-data immutability, the
training/fitting guard) are ACTUALLY checked rather than self-asserted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import decision_policy as dp
import evaluation_harness as eh
import phase5b_robustness_evaluation as p5b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_report():
    """Runs the actual Phase 5B evaluator end-to-end ONCE per test
    module against the real repository artifacts."""
    return p5b.run_phase5b_evaluation()


@pytest.fixture(scope="module")
def baseline():
    return p5b.load_canonical_baseline()


# ===========================================================================
# A. Canonical baseline / Phase 5A regression (W)
# ===========================================================================
def test_canonical_baseline_matches_phase5a_population(real_report):
    phase5a = eh.run_evaluation()
    assert real_report["baseline"]["row_count"] == phase5a["evaluation"]["row_count"]
    assert real_report["baseline"]["attack_family_counts"] == phase5a["evaluation"]["attack_family_counts"]


def test_phase5a_regression_still_passes():
    """Phase 5B must not have touched Phase 5A's own code path."""
    report = eh.run_evaluation()
    assert report["invariants"]["all_passed"] is True


# ===========================================================================
# B. Independent threshold verification (B, C)
# ===========================================================================
def test_canonical_thresholds_independently_verified(real_report):
    tv = real_report["threshold_verification"]
    assert tv["passed"] is True
    assert tv["canonical_t_review"] == pytest.approx(0.0933)
    assert tv["canonical_t_block"] == pytest.approx(0.9643)
    assert tv["evaluator_t_review"] == tv["canonical_t_review"]
    assert tv["evaluator_t_block"] == tv["canonical_t_block"]
    assert real_report["invariants"]["thresholds_identical_to_canonical_policy"] is True


def test_altered_canonical_thresholds_are_detected(monkeypatch, baseline):
    """Proves the verification is REAL: if the independent re-load
    returns different thresholds than what the evaluator used, the
    check must fail -- not silently pass."""
    tampered_policy = dict(baseline["policy"])
    tampered_policy["t_review"] = 0.5  # deliberately wrong vs. canonical 0.0933

    original_loader = eh.load_canonical_policy

    def _fake_load_canonical_policy():
        real = original_loader()
        real["t_review"] = 0.0933  # canonical artifact itself is unchanged
        return real

    monkeypatch.setattr(eh, "load_canonical_policy", _fake_load_canonical_policy)
    result = p5b.independently_verify_thresholds(tampered_policy)
    assert result["passed"] is False
    assert result["t_review_match"] is False


def test_threshold_verification_uses_independent_reload(monkeypatch, baseline):
    """The check must call the real loader again, not just echo back the
    value it was given -- monkeypatch the loader to return an altered
    canonical value and confirm the comparison target actually moved."""
    original_loader = eh.load_canonical_policy

    def _fake_load_canonical_policy():
        real = original_loader()
        real["t_block"] = 0.1234
        return real

    monkeypatch.setattr(eh, "load_canonical_policy", _fake_load_canonical_policy)
    result = p5b.independently_verify_thresholds(baseline["policy"])
    assert result["canonical_t_block"] == 0.1234
    assert result["passed"] is False


# ===========================================================================
# C. Canonical-data immutability fingerprinting (D, E, V)
# ===========================================================================
def test_canonical_data_fingerprint_stable_across_evaluation(real_report):
    ci = real_report["canonical_immutability"]
    assert ci["passed"] is True
    assert ci["pre_evaluation_fingerprint"] == ci["post_evaluation_fingerprint"]
    assert real_report["invariants"]["canonical_data_never_overwritten"] is True


def test_fingerprint_detects_mutation(baseline):
    """A fingerprint taken after mutating a COPY of the canonical y
    array must differ from the original -- proving the fingerprint is
    value-sensitive, not just checking object identity or length."""
    fp1 = p5b.fingerprint_canonical_population(baseline)

    mutated = dict(baseline)
    mutated["y"] = baseline["y"].copy()
    mutated["y"][0] = 1 - mutated["y"][0]
    fp2 = p5b.fingerprint_canonical_population(mutated)

    assert fp1["array_fingerprint_sha256"] != fp2["array_fingerprint_sha256"]


def test_fingerprint_unaffected_by_scenario_generation_copies(baseline):
    """Building every scenario population (which must copy, never
    share/mutate, the canonical arrays) must not change the canonical
    fingerprint."""
    fp_before = p5b.fingerprint_canonical_population(baseline)
    for scenario in p5b.PREVALENCE_SCENARIOS:
        p5b.build_scenario_population(baseline, scenario)
    fp_after = p5b.fingerprint_canonical_population(baseline)
    assert fp_before == fp_after


def test_scenario_generation_does_not_mutate_canonical_arrays(baseline):
    y_before = baseline["y"].copy()
    proba_before = baseline["proba"].copy()
    for scenario in p5b.PREVALENCE_SCENARIOS:
        p5b.build_scenario_population(baseline, scenario)
    assert np.array_equal(baseline["y"], y_before)
    assert np.array_equal(baseline["proba"], proba_before)


def test_cache_file_not_modified_by_evaluation(real_report):
    ci = real_report["canonical_immutability"]
    assert (
        ci["pre_evaluation_fingerprint"]["cache_file_sha256"]
        == ci["post_evaluation_fingerprint"]["cache_file_sha256"]
    )
    assert ci["pre_evaluation_fingerprint"]["cache_file_sha256"] is not None


# ===========================================================================
# D. Training/fitting/optimization guard (F, G, H, I)
# ===========================================================================
def test_normal_evaluation_passes_the_guard(real_report):
    g = real_report["training_fitting_guard"]
    assert g["passed"] is True
    assert g["violation"] is None
    assert set(g["guarded_targets"]) == set(p5b.FROZEN_GUARD_TARGETS)
    assert real_report["invariants"]["no_training_optimization_fitting_occurred"] is True


def _resolve_target(name: str):
    import autoencoder as ae
    import gcn
    import risk_fusion as rf
    from sklearn.linear_model import LogisticRegression

    mapping = {
        "gcn.train": (gcn, "train"),
        "autoencoder.train": (ae, "train"),
        "risk_fusion.run_risk_fusion": (rf, "run_risk_fusion"),
        "risk_fusion.fit_fusion_oof": (rf, "fit_fusion_oof"),
        "decision_policy.optimize_thresholds": (dp, "optimize_thresholds"),
        "decision_policy.nested_threshold_estimate": (dp, "nested_threshold_estimate"),
        "sklearn.linear_model.LogisticRegression.fit": (LogisticRegression(), "fit"),
    }
    return mapping[name]


@pytest.mark.parametrize("target_name", list(p5b.FROZEN_GUARD_TARGETS))
def test_guard_raises_for_each_forbidden_target(target_name):
    """For every single guarded entry point, calling it while the guard
    is active must raise Phase5BFrozenEvaluationViolation -- proving
    this is a real per-target guard, not a single blanket flag."""
    with pytest.raises(p5b.Phase5BFrozenEvaluationViolation):
        with p5b.frozen_execution_guard():
            owner, attr = _resolve_target(target_name)
            getattr(owner, attr)()  # any call, any args -- should raise before doing anything


def test_guard_restores_originals_after_violation():
    """Even when a violation is raised, the guard must restore the
    original callables on exit (never leave a global monkeypatch
    behind)."""
    import gcn
    original_train = gcn.train
    with pytest.raises(p5b.Phase5BFrozenEvaluationViolation):
        with p5b.frozen_execution_guard():
            gcn.train()
    assert gcn.train is original_train


def test_guard_restores_originals_after_clean_exit():
    import gcn
    original_train = gcn.train
    with p5b.frozen_execution_guard():
        pass
    assert gcn.train is original_train


def test_phase5b_evaluator_does_not_invoke_forbidden_operations(real_report):
    """The real evaluation ran fully inside frozen_execution_guard() in
    run_phase5b_evaluation() -- this asserts the guard result recorded
    on the actual run."""
    assert real_report["training_fitting_guard"]["violation"] is None
    assert real_report["training_fitting_guard"]["passed"] is True


def test_guard_does_not_leak_globally_to_unrelated_code():
    """Scoped guard: outside the `with` block, forbidden targets behave
    normally again (this repo's OTHER tests must be unaffected)."""
    import gcn
    with p5b.frozen_execution_guard():
        pass
    with pytest.raises(Exception) as exc_info:
        gcn.train()
    assert not isinstance(exc_info.value, p5b.Phase5BFrozenEvaluationViolation)


# ===========================================================================
# E. Requested vs. achieved prevalence (L, M, N, O, P)
# ===========================================================================
def test_all_scenarios_report_requested_and_achieved_prevalence(real_report):
    for scenario in real_report["scenarios"]:
        for field in (
            "requested_prevalence", "achieved_prevalence", "prevalence_error",
            "fraud_count", "legitimate_count", "total_count",
        ):
            assert field in scenario, f"missing {field!r} in scenario {scenario['scenario_name']!r}"


def test_baseline_scenario_has_no_requested_prevalence(real_report):
    baseline_scenario = next(s for s in real_report["scenarios"] if s["severity"] == "baseline")
    assert baseline_scenario["requested_prevalence"] is None
    assert baseline_scenario["prevalence_error"] is None
    assert baseline_scenario["fraud_count"] == 97 + 156 + 121  # canonical fraud count


def test_extreme_0_1_pct_scenario_documents_finite_population_gap(real_report):
    """1 fraud row / 1183 total = 0.084531%, not exactly 0.1% -- the
    report must show the real achieved value and a nonzero error, never
    claim 'exactly 0.1%'."""
    extreme = next(s for s in real_report["scenarios"] if s["severity"] == "extreme")
    assert extreme["requested_prevalence"] == pytest.approx(0.001)
    assert extreme["fraud_count"] == 1
    assert extreme["legitimate_count"] == 1182
    assert extreme["total_count"] == 1183
    expected_achieved = 1 / 1183
    assert extreme["achieved_prevalence"] == pytest.approx(expected_achieved, abs=1e-8)
    assert extreme["achieved_prevalence"] != pytest.approx(0.001, abs=1e-6)
    expected_error = abs(expected_achieved - 0.001)
    assert extreme["prevalence_error"] == pytest.approx(expected_error, abs=1e-8)
    assert extreme["prevalence_error"] > 0


def test_prevalence_error_mathematically_consistent(real_report):
    for scenario in real_report["scenarios"]:
        if scenario["requested_prevalence"] is None:
            continue
        expected = round(
            abs(scenario["achieved_prevalence"] - scenario["requested_prevalence"]), 8
        )
        assert scenario["prevalence_error"] == pytest.approx(expected, abs=1e-8)


def test_family_counts_sum_to_total_fraud_count(real_report):
    for scenario in real_report["scenarios"]:
        family_targets = scenario["perturbation_parameters"]["family_fraud_targets"]
        assert sum(family_targets.values()) == scenario["fraud_count"]


def test_total_count_equals_fraud_plus_legitimate(real_report):
    for scenario in real_report["scenarios"]:
        assert scenario["total_count"] == scenario["fraud_count"] + scenario["legitimate_count"]
        assert scenario["legitimate_count"] == 1182  # canonical legit population, never resampled


def test_no_loose_blanket_tolerance_on_extreme_prevalence(real_report):
    """Guards against reintroducing `abs(achieved - requested) < 0.01`
    style tolerances: the 0.1% scenario's real error is small but
    nonzero and must be reported as such, not rounded away."""
    extreme = next(s for s in real_report["scenarios"] if s["severity"] == "extreme")
    assert extreme["prevalence_error"] > 1e-5
    assert extreme["prevalence_error"] < 0.01


# ===========================================================================
# F. Scenario-generation correctness (J, N, O, P, Q)
# ===========================================================================
def test_deterministic_scenario_generation(baseline):
    scenario = next(s for s in p5b.PREVALENCE_SCENARIOS if s["severity"] == "moderate")
    rec1 = p5b.build_scenario_population(baseline, scenario)
    rec2 = p5b.build_scenario_population(baseline, scenario)
    assert np.array_equal(rec1["population"]["y"], rec2["population"]["y"])
    assert list(rec1["population"]["resampled_id"]) == list(rec2["population"]["resampled_id"])


def test_unique_scenario_ids_within_a_scenario(baseline):
    for scenario in p5b.PREVALENCE_SCENARIOS:
        rec = p5b.build_scenario_population(baseline, scenario)
        ids = rec["population"]["resampled_id"]
        assert len(ids) == len(set(ids.tolist()))


def test_scores_preserved_under_resampling(baseline):
    """Every resampled row's score must equal exactly the canonical
    score of the source row it was drawn from -- resampling never
    recomputes or perturbs a score."""
    canon_lookup = dict(zip(baseline["trace_id"].tolist(), baseline["proba"].tolist()))
    for scenario in p5b.PREVALENCE_SCENARIOS:
        rec = p5b.build_scenario_population(baseline, scenario)
        pop = rec["population"]
        for src_id, score in zip(pop["source_trace_id"].tolist(), pop["proba"].tolist()):
            assert canon_lookup[src_id] == pytest.approx(score)


def test_confusion_matrix_correctness_synthetic():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.2, 0.8, 0.1])  # 1 TP, 1 FN, 1 FP, 1 TN at 0.5
    classification = eh.compute_classification_metrics(y, proba)
    cm = classification["confusion_matrix"]
    assert cm["true_positive"] == 1
    assert cm["false_negative"] == 1
    assert cm["false_positive"] == 1
    assert cm["true_negative"] == 1


# ===========================================================================
# G. Full-evaluation determinism (K)
# ===========================================================================
def test_deterministic_full_evaluation():
    r1 = p5b.run_phase5b_evaluation()
    r2 = p5b.run_phase5b_evaluation()
    deterministic, diffs = p5b._compare_core_results(r1, r2)
    assert deterministic, diffs


# ===========================================================================
# H. Degradation classification (R)
# ===========================================================================
def test_degradation_category_is_recall_based(real_report):
    for name, summary in real_report["degradation_summary"].items():
        assert "degradation_category" in summary
        assert summary["degradation_category"] in {
            "stable", "mild degradation", "material degradation", "severe degradation",
            "stable (recall improved)", "unavailable",
        }


def test_baseline_scenario_degradation_is_stable(real_report):
    baseline_name = next(s["scenario_name"] for s in p5b.PREVALENCE_SCENARIOS if s["severity"] == "baseline")
    assert real_report["degradation_summary"][baseline_name]["degradation_category"] == "stable"


def test_precision_and_cost_documented_as_prevalence_dependent(real_report):
    for summary in real_report["degradation_summary"].values():
        assert "classification_note" in summary
        assert "prevalence" in summary["classification_note"].lower()


# ===========================================================================
# I. Unimplemented robustness categories (S)
# ===========================================================================
def test_five_categories_explicitly_not_implemented(real_report):
    not_impl = real_report["metadata"]["not_implemented_categories"]
    assert set(not_impl.keys()) == {
        "feature_noise", "timing_jitter", "missing_information",
        "behavioral_drift", "graph_sparsity",
    }
    for reason in not_impl.values():
        assert isinstance(reason, str) and len(reason) > 20


def test_report_never_claims_unimplemented_categories_were_evaluated(real_report):
    scenario_names = {s["scenario_name"] for s in real_report["scenarios"]}
    for cat in real_report["metadata"]["not_implemented_categories"]:
        assert not any(cat in name for name in scenario_names)
    assert real_report["metadata"]["implemented_category"] == "class_imbalance_prevalence_shift"


# ===========================================================================
# J. Report schema (T)
# ===========================================================================
def test_report_schema_required_sections(real_report):
    required = [
        "schema_version", "metadata", "baseline", "frozen_policy", "scenarios",
        "scenario_results", "attack_family_results", "degradation_summary",
        "invariants", "threshold_verification", "canonical_immutability",
        "training_fitting_guard", "leakage_protections", "provenance", "conclusions",
    ]
    for key in required:
        assert key in real_report, f"missing top-level report section {key!r}"


def test_report_is_json_serializable(real_report):
    serialized = json.dumps(real_report, default=p5b._json_default)
    reloaded = json.loads(serialized)
    assert reloaded["invariants"]["all_passed"] is True


def test_provenance_present(real_report):
    prov = real_report["provenance"]
    for key in ("git_commit", "python_version", "package_versions",
                "policy_artifact_provenance", "cache_provenance", "phase5b_seed_root"):
        assert key in prov


def test_persisted_report_file_matches_schema():
    """The persisted phase5b_robustness_report.json (produced by
    `python phase5b_robustness_evaluation.py`) must itself satisfy the
    schema -- not just an in-memory report object."""
    with open(p5b.REPORT_PATH) as f:
        persisted = json.load(f)
    assert persisted["invariants"]["all_passed"] is True
    assert "determinism_check" in persisted
    assert persisted["determinism_check"]["deterministic"] is True


# ===========================================================================
# K. No duplicate evaluator (single authoritative file)
# ===========================================================================
def test_single_authoritative_evaluator_module():
    from pathlib import Path
    import phase5b_robustness_evaluation as mod
    assert Path(mod.__file__).resolve() == (p5b.REPO_ROOT / "phase5b_robustness_evaluation.py").resolve()
    assert not (p5b.REPO_ROOT / "tests" / "phase5b_robustness_evaluation.py").exists()
