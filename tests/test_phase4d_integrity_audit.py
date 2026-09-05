"""
Regression tests for Phase 4D: the final Blue Team integrity/freeze audit.

These tests exercise the REAL functions in phase4d_integrity_audit.py
(which itself calls decision_policy.py's, consistency_check.py's, and
blue_team_pipeline.py's real, unmodified functions -- see that module's
own docstring for which sections delegate to Phase 4C rather than
re-implementing). Nothing here is a reimplementation of the audit logic;
these tests call phase4d_integrity_audit's public check_* / section_*
functions directly, the same way test_phase4c_artifact_provenance.py
exercises decision_policy.py and consistency_check.py directly.

Two kinds of tests are included, per requirement I ("at least one real
test using actual repository artifacts, not a mock"):
  1. Synthetic-fixture tests -- fast, deterministic, isolate one specific
     invariant (e.g. a mismatched liability breakdown) without touching
     any real file on disk.
  2. Real-artifact tests -- run the actual check/section functions against
     whatever decision_policy_results.json / misses.jsonl / case_reports.json
     / decision_policy_validation_cache.npz / web API currently exist in
     this repo checkout.

NOTE: importing phase4d_integrity_audit.py (and therefore this test
module) requires the project's full pinned stack (xgboost, sklearn,
pandas, etc. -- see requirements.txt), because it imports decision_policy
/ blue_team_pipeline / consistency_check at module load time, exactly
like test_phase4c_artifact_provenance.py and
test_decision_policy_nested_threshold.py already do. Nothing new is
introduced here.

Tests do not require network access and do not commit or push anything.
No test writes to phase4d_integrity_audit_results.json or any other
real repo artifact -- file-writing tests use pytest's tmp_path / monkeypatch.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import blue_team_pipeline as btp
import decision_policy as dp
import consistency_check as cc
import phase4d_integrity_audit as p4d


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------
def _minimal_valid_decision_policy() -> dict:
    """A decision_policy_results.json-shaped dict that satisfies every
    Section A invariant, used as a baseline that individual tests mutate
    to break exactly one thing at a time."""
    liability_row = lambda: {
        "liable_side": "SENDING", "acting_side": "SENDING",
        "sending_liability_share": 1.0, "receiving_liability_share": 0.0,
        "n_fraud_traces": 5,
    }
    corrected = {
        "t_review": 0.1, "t_block": 0.9,
        "cost_model": {"assumed_production_fraud_rate": 0.006},
        "liability_breakdown": {
            "ACCOUNT_TAKEOVER": liability_row(),
            "AUTHORIZED_PUSH_PAYMENT": liability_row(),
            "MULE_NETWORK": liability_row(),
        },
    }
    naive = {
        "t_review": 0.5, "t_block": 0.99,
        "cost_model": {"assumed_production_fraud_rate": None},
    }
    return {
        "naive": naive,
        "corrected": corrected,
        "score_source": {"score": "risk_fusion_stacked_lr"},
        "nested_fold_honest_estimate": {
            "fold_thresholds": [
                {"fold": 1, "n_selection_rows": 80, "n_holdout_rows": 20},
                {"fold": 2, "n_selection_rows": 80, "n_holdout_rows": 20},
                {"fold": 3, "n_selection_rows": 80, "n_holdout_rows": 20},
                {"fold": 4, "n_selection_rows": 80, "n_holdout_rows": 20},
                {"fold": 5, "n_selection_rows": 80, "n_holdout_rows": 20},
            ],
        },
        "methodology_note": {
            "threshold_selection_optimism": (
                "The 'corrected' block selects thresholds by grid search on the "
                "full population and is judged on that same population -- a "
                "source of optimism. See nested_fold_honest_estimate."
            ),
        },
    }


def _minimal_valid_miss(trace_id="t1", t_review=0.1, t_block=0.9, score=0.05,
                         attack_family="ACCOUNT_TAKEOVER") -> dict:
    return {
        "trace_id": trace_id, "customer_id": "cust-1",
        "attack_family": attack_family, "attack_difficulty": "easy",
        "final_decision": "ALLOW", "final_score": score,
        "t_review": t_review, "t_block": t_block,
        "dollars_in_trace": 100.0, "stage1_escalated_to_ml": True,
        "graph_connected": False,
        "stage1_rules_checked": [], "behavioral_features": {},
        "reason_for_miss": "test fixture",
    }


# ===========================================================================
# Section A -- decision policy integrity
# ===========================================================================
class TestSectionA:
    def test_valid_policy_passes_all(self):
        dpres = _minimal_valid_decision_policy()
        for check in p4d.section_a(dpres):
            assert check["passed"] is True, check

    def test_missing_artifact_skips_all(self):
        for check in p4d.section_a(None):
            assert check["passed"] is None

    def test_t_review_not_below_t_block_fails(self):
        dpres = _minimal_valid_decision_policy()
        dpres["corrected"]["t_review"] = 0.95
        dpres["corrected"]["t_block"] = 0.90
        result = p4d.check_a1_threshold_ordering_and_bounds(dpres)
        assert result["passed"] is False
        assert "strictly below" in result["details"][0]

    def test_threshold_out_of_bounds_fails(self):
        dpres = _minimal_valid_decision_policy()
        dpres["corrected"]["t_block"] = 1.5
        result = p4d.check_a1_threshold_ordering_and_bounds(dpres)
        assert result["passed"] is False

    def test_equal_thresholds_fail_strict_ordering(self):
        dpres = _minimal_valid_decision_policy()
        dpres["corrected"]["t_review"] = 0.5
        dpres["corrected"]["t_block"] = 0.5
        result = p4d.check_a1_threshold_ordering_and_bounds(dpres)
        assert result["passed"] is False

    def test_missing_liability_family_fails(self):
        dpres = _minimal_valid_decision_policy()
        del dpres["corrected"]["liability_breakdown"]["MULE_NETWORK"]
        result = p4d.check_a2_liability_breakdown_complete(dpres)
        assert result["passed"] is False
        assert "MULE_NETWORK" in result["details"][0]

    def test_liability_family_missing_required_field_fails(self):
        dpres = _minimal_valid_decision_policy()
        del dpres["corrected"]["liability_breakdown"]["ACCOUNT_TAKEOVER"]["liable_side"]
        result = p4d.check_a2_liability_breakdown_complete(dpres)
        assert result["passed"] is False

    def test_null_prevalence_fails(self):
        dpres = _minimal_valid_decision_policy()
        dpres["corrected"]["cost_model"]["assumed_production_fraud_rate"] = None
        result = p4d.check_a3_prevalence_represented(dpres)
        assert result["passed"] is False

    def test_prevalence_out_of_range_fails(self):
        dpres = _minimal_valid_decision_policy()
        dpres["corrected"]["cost_model"]["assumed_production_fraud_rate"] = 1.5
        result = p4d.check_a3_prevalence_represented(dpres)
        assert result["passed"] is False

    def test_naive_block_secretly_reweighted_fails(self):
        """If 'naive' also had reweighting enabled, it would no longer be
        a genuine unweighted contrast against 'corrected' -- this is
        exactly the naive-vs-corrected distinction diagnose_prevalence_bug
        exists to make visible."""
        dpres = _minimal_valid_decision_policy()
        dpres["naive"]["cost_model"]["assumed_production_fraud_rate"] = 0.006
        result = p4d.check_a4_final_policy_is_prevalence_corrected(dpres)
        assert result["passed"] is False
        assert "naive" in result["details"][0]

    def test_missing_naive_block_fails(self):
        dpres = _minimal_valid_decision_policy()
        del dpres["naive"]
        result = p4d.check_a4_final_policy_is_prevalence_corrected(dpres)
        assert result["passed"] is False

    def test_nested_estimate_missing_fails(self):
        dpres = _minimal_valid_decision_policy()
        del dpres["nested_fold_honest_estimate"]
        result = p4d.check_a5_nested_fold_honest_disclosure(dpres)
        assert result["passed"] is False

    def test_nested_estimate_rows_dont_cover_population_fails(self):
        dpres = _minimal_valid_decision_policy()
        # Corrupt one fold's holdout count so the folds no longer sum to
        # the full population.
        dpres["nested_fold_honest_estimate"]["fold_thresholds"][0]["n_holdout_rows"] = 999
        result = p4d.check_a5_nested_fold_honest_disclosure(dpres)
        assert result["passed"] is False

    def test_missing_methodology_disclosure_fails(self):
        dpres = _minimal_valid_decision_policy()
        dpres["methodology_note"] = {}
        result = p4d.check_a5_nested_fold_honest_disclosure(dpres)
        assert result["passed"] is False


# ===========================================================================
# Section B -- artifact consistency (delegation to consistency_check.py)
# ===========================================================================
class TestSectionBDelegatesToConsistencyCheck:
    def test_section_b_calls_are_identical_to_direct_cc_calls(self):
        """phase4d's section_b must not reimplement any comparison logic --
        verify its output is byte-identical to calling consistency_check's
        functions directly with the same inputs."""
        dpres = _minimal_valid_decision_policy()
        misses = [_minimal_valid_miss(t_review=dpres["corrected"]["t_review"],
                                       t_block=dpres["corrected"]["t_block"])]
        case_reports = {}

        via_p4d = p4d.section_b(misses, dpres, case_reports)

        direct = [
            cc.check_threshold_agreement(misses, dpres),
            cc.check_case_reports_threshold_agreement(case_reports, dpres),
            cc.check_miss_completeness(misses, case_reports),
        ]
        score_check, label_check = cc.check_shared_trace_consistency(misses, case_reports)
        direct += [score_check, label_check]

        assert via_p4d == direct

    def test_mismatched_threshold_pair_fails(self):
        dpres = _minimal_valid_decision_policy()
        misses = [_minimal_valid_miss(t_review=0.999, t_block=0.9999)]
        checks = p4d.section_b(misses, dpres, {})
        threshold_check = next(c for c in checks if c["name"] == "threshold_agreement")
        assert threshold_check["passed"] is False


# ===========================================================================
# Section C -- validation cache integrity (Phase 4C B-2 regression guard)
# ===========================================================================
class TestSectionC:
    def _write_cache(self, path, *, variant, y=None):
        y = np.array([0, 1, 0, 1]) if y is None else y
        if variant is None:
            np.savez(path, y=y, proba=np.array([0.1, 0.9, 0.2, 0.8]),
                      dollars=np.array([1.0, 2.0, 3.0, 4.0]))
        else:
            np.savez(path, y=y, proba=np.array([0.1, 0.9, 0.2, 0.8]),
                      dollars=np.array([1.0, 2.0, 3.0, 4.0]), validation_variant=variant)

    def test_no_cache_file_skips(self, tmp_path):
        result = p4d.check_c1_cache_self_identifies(tmp_path / "nonexistent.npz")
        assert result["passed"] is None

    def test_missing_variant_tag_fails_c1(self, tmp_path):
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant=None)
        result = p4d.check_c1_cache_self_identifies(cache_path)
        assert result["passed"] is False
        assert "B-2" in result["details"][0]

    def test_valid_variant_tag_passes_c1(self, tmp_path):
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="fused")
        result = p4d.check_c1_cache_self_identifies(cache_path)
        assert result["passed"] is True

    def test_invalid_variant_value_fails_c1(self, tmp_path):
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="not_a_real_variant")
        result = p4d.check_c1_cache_self_identifies(cache_path)
        assert result["passed"] is False

    def test_c2_cross_consumption_is_correctly_refused(self, tmp_path):
        """The core Phase 4C B-2 regression guard: a 'cascade'-tagged cache
        must never satisfy a 'fused' request or vice versa. This test
        never touches the real decision_policy_validation_cache.npz --
        check_c2 itself copies to a fresh temp file internally, and this
        test additionally points it at a tmp_path fixture cache."""
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="cascade")
        original = dp.CACHE_PATH
        try:
            result = p4d.check_c2_cache_variant_cannot_be_silently_cross_consumed(cache_path)
        finally:
            assert dp.CACHE_PATH == original, "check_c2 must restore dp.CACHE_PATH even on success"
        assert result["passed"] is True

    def test_c2_restores_cache_path_on_exception(self, tmp_path, monkeypatch):
        """Even if something inside the probe raises unexpectedly, dp.CACHE_PATH
        must be restored -- this check must never leave global state mutated
        for whatever runs after it (e.g. a real decision_policy.py call in
        the same process)."""
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="fused")
        original = dp.CACHE_PATH

        real_load = dp.load_cached_validation_data
        call_count = {"n": 0}

        def _boom(variant, y_check=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated unexpected failure")
            return real_load(variant, y_check=y_check)

        monkeypatch.setattr(dp, "load_cached_validation_data", _boom)
        with pytest.raises(RuntimeError):
            p4d.check_c2_cache_variant_cannot_be_silently_cross_consumed(cache_path)
        assert dp.CACHE_PATH == original

    def test_c3_mismatched_score_source_fails_direct(self, tmp_path):
        dpres = _minimal_valid_decision_policy()
        dpres["score_source"]["score"] = "risk_fusion_stacked_lr"
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="cascade")
        result = p4d.check_c3_cache_matches_decision_policy_variant(cache_path, dpres)
        assert result["passed"] is False

    def test_c3_matching_score_source_passes(self, tmp_path):
        dpres = _minimal_valid_decision_policy()
        dpres["score_source"]["score"] = "risk_fusion_stacked_lr"
        cache_path = tmp_path / "cache.npz"
        self._write_cache(cache_path, variant="fused")
        result = p4d.check_c3_cache_matches_decision_policy_variant(cache_path, dpres)
        assert result["passed"] is True


# ===========================================================================
# Section D -- provenance integrity (delegation)
# ===========================================================================
class TestSectionD:
    def test_dirty_tree_is_warning_not_failure(self):
        dpres = {"_artifact_metadata": {"git_commit": "abc123", "git_dirty": True}}
        misses_meta = {"git_commit": "abc123", "git_dirty": True}
        case_reports = {"_artifact_metadata": {"git_commit": "abc123", "git_dirty": False}}
        checks = p4d.section_d(dpres, misses_meta, case_reports)
        assert len(checks) == 1
        assert checks[0]["passed"] is True  # dirty tree alone must not fail this
        assert any("WARNING" in d for d in checks[0]["details"])

    def test_disagreeing_commits_fail(self):
        dpres = {"_artifact_metadata": {"git_commit": "abc123", "git_dirty": False}}
        misses_meta = {"git_commit": "DIFFERENT", "git_dirty": False}
        checks = p4d.section_d(dpres, misses_meta, None)
        assert checks[0]["passed"] is False

    def test_freeze_procedure_note_present_and_explains_regeneration(self):
        assert "regenerate" in p4d.FREEZE_PROCEDURE_NOTE.lower()
        assert "clean" in p4d.FREEZE_PROCEDURE_NOTE.lower()

    def test_freeze_note_does_not_demand_clean_tree_for_ordinary_runs(self):
        """Section D itself must never turn a dirty tree into a FAIL --
        confirmed above in test_dirty_tree_is_warning_not_failure. This
        additionally confirms run_all_checks() as a whole doesn't fail
        just because the real repo's artifacts were dirty-generated."""
        result = p4d.run_all_checks()
        provenance = next(c for c in result["checks"] if c["name"] == "provenance_agreement")
        if any("dirty working tree" in d for d in provenance["details"]):
            assert provenance["passed"] is not False


# ===========================================================================
# Section E -- online/offline boundary
# ===========================================================================
class TestSectionE:
    def test_schema_check_runs_against_real_api_schemas(self):
        """Requires the web API stack (fastapi/pydantic-settings) to be
        importable; skip gracefully if it isn't rather than failing the
        whole suite on an unrelated missing optional dependency."""
        pytest.importorskip("fastapi")
        checks = p4d.section_e()
        names = {c["name"] for c in checks}
        assert "e1_schema_offline_stages_typed_unavailable" in names
        e1 = next(c for c in checks if c["name"] == "e1_schema_offline_stages_typed_unavailable")
        assert e1["passed"] is True

    def test_schema_would_fail_if_stage3_were_typed_available(self):
        """Adversarial check: construct a fake schemas-like namespace where
        stage3_graph is (incorrectly) typed as StageScore, and confirm
        check_e1 actually catches it rather than trivially passing."""
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class FakeStageScore(BaseModel):
            available: bool = True

        class FakeStageUnavailable(BaseModel):
            available: bool = False

        class FakeScoreResponse(BaseModel):
            stage1_2: FakeStageScore
            stage3_graph: FakeStageScore  # BUG: should be FakeStageUnavailable
            stage4_autoencoder: FakeStageUnavailable
            stage5_fusion: FakeStageUnavailable

        class FakeSchemas:
            ScoreResponse = FakeScoreResponse
            StageScore = FakeStageScore
            StageUnavailable = FakeStageUnavailable

        result = p4d.check_e1_schema_never_types_offline_stages_as_available(FakeSchemas)
        assert result["passed"] is False
        assert "stage3_graph" in result["details"][0]

    def test_runtime_check_against_real_model_and_real_corpus_trace(self):
        """The requirement-I real end-to-end test: loads the actual saved
        xgb_model.joblib/calibrator.joblib and scores a real Red Team ATO
        corpus trace, then asserts Stage 3/4/5 are honestly reported
        unavailable. Skips (does not fail) if the model artifacts or corpus
        aren't present in this checkout, or the API stack isn't installed."""
        pytest.importorskip("fastapi")
        pytest.importorskip("shap")
        checks = p4d.section_e()
        e2 = next((c for c in checks if c["name"] == "e2_score_trace_offline_stages_unavailable_at_runtime"), None)
        assert e2 is not None
        if e2["passed"] is None:
            pytest.skip(f"e2 skipped in this environment: {e2['details']}")
        assert e2["passed"] is True


# ===========================================================================
# Section F -- Stage 4 threshold single-source-of-truth
# ===========================================================================
class TestSectionF:
    def test_real_modules_agree_on_decision_threshold(self):
        result = p4d.check_f1_stage4_decision_threshold_agreement()
        assert result["passed"] in (True, None)  # None only if cascade modules can't import

    def test_drifted_threshold_is_detected(self, monkeypatch):
        """Adversarial check: simulate one consumer having drifted to a
        different DECISION_THRESHOLD and confirm check_f1 catches it."""
        import cascade_with_graph as cwg
        original = cwg.DECISION_THRESHOLD
        monkeypatch.setattr(cwg, "DECISION_THRESHOLD", original + 0.1)
        try:
            result = p4d.check_f1_stage4_decision_threshold_agreement()
        finally:
            assert cwg.DECISION_THRESHOLD == original + 0.1  # monkeypatch not yet undone
        assert result["passed"] is False
        assert "drifted" in result["details"][0]


# ===========================================================================
# Section G -- miss/corpus accounting
# ===========================================================================
class TestSectionG:
    def test_valid_misses_pass(self):
        misses = [_minimal_valid_miss(trace_id="t1", attack_family="ACCOUNT_TAKEOVER"),
                  _minimal_valid_miss(trace_id="t2", attack_family="MULE_NETWORK")]
        for check in p4d.section_g(misses):
            assert check["passed"] is True, check

    def test_empty_misses_skip(self):
        """g1/g2 depend on misses.jsonl and must skip when it's empty; g3
        is a static check against dp.LIABILITY_SIDE and doesn't depend on
        misses at all, so it still runs (and passes) regardless."""
        checks = {c["name"]: c for c in p4d.section_g([])}
        assert checks["g1_misses_jsonl_structurally_valid"]["passed"] is None
        assert checks["g2_miss_attack_family_labels_valid"]["passed"] is None
        assert checks["g3_supported_attack_families_unchanged"]["passed"] is True

    def test_missing_required_key_fails_g1(self):
        miss = _minimal_valid_miss()
        del miss["dollars_in_trace"]
        result = p4d.check_g1_misses_structurally_valid([miss])
        assert result["passed"] is False
        assert "dollars_in_trace" in result["details"][0]

    def test_non_allow_decision_fails_g1(self):
        miss = _minimal_valid_miss()
        miss["final_decision"] = "REVIEW"
        result = p4d.check_g1_misses_structurally_valid([miss])
        assert result["passed"] is False

    def test_score_above_t_review_fails_g1(self):
        """misses.jsonl is defined as score < t_review by construction
        (see miss_collector.collect_misses); a record violating that is
        structurally invalid regardless of what final_decision says."""
        miss = _minimal_valid_miss(score=0.5, t_review=0.1, t_block=0.9)
        result = p4d.check_g1_misses_structurally_valid([miss])
        assert result["passed"] is False

    def test_inverted_thresholds_fail_g1(self):
        miss = _minimal_valid_miss(t_review=0.9, t_block=0.1, score=0.05)
        result = p4d.check_g1_misses_structurally_valid([miss])
        assert result["passed"] is False

    def test_unsupported_attack_family_fails_g2(self):
        miss = _minimal_valid_miss(attack_family="SOME_NEW_FAMILY")
        result = p4d.check_g2_miss_attack_families_valid([miss])
        assert result["passed"] is False
        assert "SOME_NEW_FAMILY" in result["details"][0]

    def test_g3_matches_real_liability_side(self):
        """This is intentionally a tautology against the real
        dp.LIABILITY_SIDE (EXPECTED_ATTACK_FAMILIES is derived from it) --
        its purpose is to catch someone editing LIABILITY_SIDE's keys
        without updating this audit's hardcoded expectation, which would
        otherwise silently start rubber-stamping a changed family set."""
        result = p4d.check_g3_supported_families_unchanged()
        assert result["passed"] is True


# ===========================================================================
# Section H -- reproducibility / determinism
# ===========================================================================
class TestSectionH:
    def test_stable_fold_id_deterministic(self):
        result = p4d.check_h1_stable_fold_id_deterministic()
        assert result["passed"] is True

    def test_stable_kfold_split_deterministic(self):
        result = p4d.check_h2_stable_kfold_split_deterministic()
        assert result["passed"] is True

    def test_stable_fold_id_actually_deterministic_property(self):
        """Direct property test of the underlying (unmodified) function,
        independent of the audit's own wrapper -- belt and suspenders."""
        for trace_id in ["atk-abc123", "legit_sess_xyz", ""]:
            a = btp.stable_fold_id(trace_id, random_state=42)
            b = btp.stable_fold_id(trace_id, random_state=42)
            assert a == b

    def test_stable_kfold_split_position_independence_property(self):
        """Direct property test: two dataframes containing the SAME rows
        in a DIFFERENT order must assign each trace_id to the same fold --
        this is the entire point of stable_fold (see blue_team_pipeline.py's
        own module docstring on the Round1->Round2 phantom-regression bug
        it fixes)."""
        df = pd.DataFrame({
            "trace_id": [f"t{i}" for i in range(30)],
            "fraud": [i % 4 == 0 for i in range(30)],
        })
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

        folds_orig = btp.stable_kfold_split(df, "fraud", n_splits=3, random_state=42)
        folds_shuf = btp.stable_kfold_split(shuffled, "fraud", n_splits=3, random_state=42)

        fold_of = {}
        for fold_i, (_, test_idx) in enumerate(folds_orig):
            for i in test_idx:
                fold_of[df.at[i, "trace_id"]] = fold_i
        fold_of_shuf = {}
        for fold_i, (_, test_idx) in enumerate(folds_shuf):
            for i in test_idx:
                fold_of_shuf[shuffled.at[i, "trace_id"]] = fold_i

        assert fold_of == fold_of_shuf


# ===========================================================================
# Full-run integration: run_all_checks() / main() against the REAL repo
# ===========================================================================
class TestFullRunAgainstRealRepo:
    def test_run_all_checks_executes_without_raising(self):
        """Smoke test: the full audit must run to completion against
        whatever real artifacts currently exist in this checkout, without
        raising, and must produce well-formed check dicts."""
        result = p4d.run_all_checks()
        assert result["verdict"] in ("PASS", "FAIL", "NO_CHECKS_RAN")
        assert isinstance(result["checks"], list) and result["checks"]
        for c in result["checks"]:
            assert set(c.keys()) >= {"name", "passed", "details"}
            assert c["passed"] in (True, False, None)
            assert isinstance(c["details"], list)

    def test_verdict_is_fail_iff_any_check_failed(self):
        result = p4d.run_all_checks()
        any_fail = any(c["passed"] is False for c in result["checks"])
        assert (result["verdict"] == "FAIL") == any_fail

    def test_main_writes_results_file_and_returns_expected_exit_code(self, tmp_path, monkeypatch):
        """main() must write phase4d_integrity_audit_results.json and
        return 0 on PASS/SKIPPED-only, 1 on any real FAIL -- verified by
        redirecting RESULTS_PATH to tmp_path so this test never overwrites
        the real repo's results file."""
        fake_results_path = tmp_path / "phase4d_integrity_audit_results.json"
        monkeypatch.setattr(p4d, "RESULTS_PATH", fake_results_path)

        exit_code = p4d.main()

        assert fake_results_path.exists()
        with open(fake_results_path) as f:
            written = json.load(f)
        assert written["verdict"] in ("PASS", "FAIL", "NO_CHECKS_RAN")
        expected_code = 1 if written["verdict"] == "FAIL" else 0
        assert exit_code == expected_code

    def test_main_never_writes_outside_results_path(self, tmp_path, monkeypatch):
        """Guard against main() accidentally touching any real pipeline
        source/model file (spec item I: 'without modifying model/source
        files'). We can't enumerate every file in the repo, but we can
        confirm main() only ever opens RESULTS_PATH for writing by
        patching builtins.open to record write-mode calls."""
        fake_results_path = tmp_path / "results.json"
        monkeypatch.setattr(p4d, "RESULTS_PATH", fake_results_path)

        import builtins
        real_open = builtins.open
        write_paths = []

        def _tracking_open(file, mode="r", *args, **kwargs):
            if "w" in mode or "a" in mode or "x" in mode:
                write_paths.append(str(file))
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _tracking_open)
        p4d.main()

        assert all(p == str(fake_results_path) for p in write_paths), (
            f"main() opened unexpected path(s) for writing: {write_paths}"
        )

    def test_consistency_check_and_phase4d_agree_on_shared_checks(self):
        """Since section_b/section_d delegate to consistency_check.py
        verbatim, running consistency_check.main()'s underlying checks and
        phase4d's should report the same pass/fail for the checks they
        share, against the same real on-disk artifacts."""
        misses = cc._load_jsonl(cc.MISSES_PATH)
        misses_meta = cc._load_jsonl_metadata(cc.MISSES_PATH)
        decision_policy = cc._load_json(cc.DECISION_POLICY_PATH)
        case_reports = cc._load_json(cc.CASE_REPORTS_PATH)

        cc_checks = {c["name"]: c["passed"] for c in [
            cc.check_threshold_agreement(misses, decision_policy),
            cc.check_case_reports_threshold_agreement(case_reports, decision_policy),
            cc.check_miss_completeness(misses, case_reports),
        ]}
        score_check, label_check = cc.check_shared_trace_consistency(misses, case_reports)
        cc_checks[score_check["name"]] = score_check["passed"]
        cc_checks[label_check["name"]] = label_check["passed"]
        cc_checks["provenance_agreement"] = cc.check_provenance_agreement(
            decision_policy, misses_meta, case_reports)["passed"]

        result = p4d.run_all_checks()
        p4d_checks = {c["name"]: c["passed"] for c in result["checks"]}

        for name, passed in cc_checks.items():
            assert p4d_checks[name] == passed, f"{name} disagrees: cc={passed}, p4d={p4d_checks[name]}"
