"""
phase4d_integrity_audit.py
============================
Phase 4D -- final Blue Team integrity / freeze audit (pre-Phase-5).

WHAT THIS IS
------------
A READ-ONLY audit over the artifacts already on disk (decision_policy_
results.json, misses.jsonl, blue_team_output/explainability/case_reports.json,
decision_policy_validation_cache.npz, the online API's schema/inference
module, and the pipeline's stable-fold helpers). It does not retrain
anything, does not regenerate any of the artifacts it inspects, and does
not modify decision_policy.py / blue_team_pipeline.py / gcn.py /
autoencoder.py / risk_fusion.py / the web API.

Wherever Phase 4C (consistency_check.py) already checks something this
audit needs (cross-artifact threshold/score/label/provenance agreement),
this script calls consistency_check.py's own functions directly instead
of re-implementing that logic a second, divergent way -- see the
`import consistency_check as cc` block below and checks B/D.

INVARIANTS COVERED (spec letters A-H)
--------------------------------------
A. Decision policy integrity        -> section A
B. Artifact consistency              -> section B (delegates to cc.*)
C. Validation cache integrity        -> section C
D. Provenance integrity              -> section D (delegates to cc.*)
E. Online/offline boundary           -> section E
F. Threshold/artifact agreement      -> section F
G. Miss/corpus accounting            -> section G
H. Reproducibility/determinism       -> section H

Every check function returns a dict:
    {"name": str, "passed": True | False | None, "details": [str, ...]}
`passed is None` means SKIPPED (a required input artifact was missing --
this is NOT a failure of the audit itself, it just means that particular
invariant couldn't be evaluated this run). `passed is False` is the only
thing that makes the overall verdict FAIL.

A dirty working tree is explicitly represented as a WARNING (see
check D / consistency_check.check_provenance_agreement's existing
"dirty_true" handling) and never as a FAIL. The freeze procedure --
i.e. what to actually do about it before release -- is documented in
FREEZE_PROCEDURE_NOTE below, not enforced as a check against the
ordinary dirty development tree.

Run (read-only, safe at any time):
    PYTHONPATH=src python3 phase4d_integrity_audit.py

Writes phase4d_integrity_audit_results.json next to this script and
returns exit code 0 if every check is PASS or SKIPPED, 1 if any check
genuinely FAILs.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import blue_team_pipeline as btp          # noqa: E402  -- unmodified
import decision_policy as dp              # noqa: E402  -- unmodified
import consistency_check as cc            # noqa: E402  -- unmodified, reused verbatim

REPO_ROOT = Path(__file__).parent
DECISION_POLICY_PATH = REPO_ROOT / "decision_policy_results.json"
MISSES_PATH = REPO_ROOT / "misses.jsonl"
CASE_REPORTS_PATH = REPO_ROOT / "blue_team_output" / "explainability" / "case_reports.json"
CACHE_PATH = REPO_ROOT / "decision_policy_validation_cache.npz"
RESULTS_PATH = REPO_ROOT / "phase4d_integrity_audit_results.json"

# The three attack families this whole pipeline (Red Team corpora, Stage 1
# rules, the role-aware liability model, miss accounting) is built around.
# Sourced from decision_policy.LIABILITY_SIDE -- the single existing place
# that already enumerates them -- rather than hardcoded a second time here,
# so this list can't silently drift out of sync with the real one.
EXPECTED_ATTACK_FAMILIES = frozenset(dp.LIABILITY_SIDE.keys())

FREEZE_PROCEDURE_NOTE = (
    "This audit does NOT require the current working tree to be clean -- "
    "that would be a false failure during ordinary development, where "
    "artifacts are routinely regenerated against uncommitted changes. "
    "check_provenance_agreement (Phase 4C, reused here as section D) "
    "reports a dirty tree as an explicit WARNING, not a FAIL. Before an "
    "actual release/freeze: (1) commit all pending changes, (2) regenerate "
    "decision_policy_results.json, misses.jsonl, and "
    "blue_team_output/explainability/case_reports.json from that clean "
    "commit (in that order: decision_policy.py, then miss_collector.py "
    "and/or explainability.py), (3) re-run this audit and "
    "consistency_check.py against the freshly regenerated artifacts, and "
    "only then tag/freeze. Regenerating from a dirty tree and freezing "
    "anyway defeats the point of git_commit provenance stamping."
)


# ---------------------------------------------------------------------------
# small shared loaders (identical semantics to consistency_check.py's own,
# reused by name where possible so both scripts agree on what "the current
# artifacts" even means)
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    return cc._load_json(path)


def _load_jsonl(path: Path):
    return cc._load_jsonl(path)


def _load_jsonl_metadata(path: Path):
    return cc._load_jsonl_metadata(path)


def _skip(name: str, msg: str) -> dict:
    return {"name": name, "passed": None, "details": [msg]}


def _ok(name: str, msg: str) -> dict:
    return {"name": name, "passed": True, "details": [msg]}


def _fail(name: str, msg: str) -> dict:
    return {"name": name, "passed": False, "details": [msg]}


# ===========================================================================
# A. DECISION POLICY INTEGRITY
# ===========================================================================
def check_a1_threshold_ordering_and_bounds(decision_policy: dict | None) -> dict:
    name = "a1_threshold_ordering_and_bounds"
    if decision_policy is None:
        return _skip(name, "decision_policy_results.json not found -- skipped.")
    corrected = decision_policy.get("corrected", {})
    t_review = corrected.get("t_review")
    t_block = corrected.get("t_block")
    if t_review is None or t_block is None:
        return _fail(name, f"corrected block missing t_review/t_block: {corrected}")
    problems = []
    if not (0.0 <= t_review <= 1.0):
        problems.append(f"t_review={t_review} not in [0, 1]")
    if not (0.0 <= t_block <= 1.0):
        problems.append(f"t_block={t_block} not in [0, 1]")
    if not (t_review < t_block):
        problems.append(f"t_review ({t_review}) is not strictly below t_block ({t_block})")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(name, f"t_review={t_review} < t_block={t_block}, both in [0, 1].")


def check_a2_liability_breakdown_complete(decision_policy: dict | None) -> dict:
    name = "a2_liability_breakdown_complete"
    if decision_policy is None:
        return _skip(name, "decision_policy_results.json not found -- skipped.")
    breakdown = decision_policy.get("corrected", {}).get("liability_breakdown", {})
    present = set(breakdown.keys())
    missing = EXPECTED_ATTACK_FAMILIES - present
    if missing:
        return _fail(
            name,
            f"liability_breakdown is missing required families: {sorted(missing)} "
            f"(has: {sorted(present)}, required: {sorted(EXPECTED_ATTACK_FAMILIES)})."
        )
    # Sanity-check each present family actually carries the fields
    # liability_breakdown()/_liability_breakdown_from_masks() always produce.
    required_fields = {
        "liable_side", "acting_side", "sending_liability_share",
        "receiving_liability_share", "n_fraud_traces",
    }
    bad = {}
    for fam in EXPECTED_ATTACK_FAMILIES:
        row = breakdown.get(fam, {})
        missing_fields = required_fields - set(row.keys())
        if missing_fields:
            bad[fam] = sorted(missing_fields)
    if bad:
        return _fail(name, f"Family entries missing required fields: {bad}")
    return _ok(
        name,
        f"liability_breakdown present for all {len(EXPECTED_ATTACK_FAMILIES)} required "
        f"families: {sorted(EXPECTED_ATTACK_FAMILIES)}."
    )


def check_a3_prevalence_represented(decision_policy: dict | None) -> dict:
    name = "a3_prevalence_represented_in_cost_model"
    if decision_policy is None:
        return _skip(name, "decision_policy_results.json not found -- skipped.")
    cost_model = decision_policy.get("corrected", {}).get("cost_model", {})
    rate = cost_model.get("assumed_production_fraud_rate")
    if rate is None:
        return _fail(
            name,
            "corrected.cost_model.assumed_production_fraud_rate is null -- the "
            "deployed 'corrected' policy is not actually prevalence-corrected."
        )
    if not (0.0 < rate < 1.0):
        return _fail(name, f"assumed_production_fraud_rate={rate} is not a sane prevalence in (0, 1).")
    return _ok(name, f"corrected.cost_model.assumed_production_fraud_rate={rate} is represented and sane.")


def check_a4_final_policy_is_prevalence_corrected(decision_policy: dict | None) -> dict:
    """The artifact must expose BOTH the naive and corrected optima (see
    diagnose_prevalence_bug), and the naive one must literally be computed
    with reweighting disabled (assumed_production_fraud_rate=None) so it is
    a genuine contrast, not a second copy of the corrected policy -- and the
    thing actually meant to be deployed ('corrected') must be the reweighted
    one."""
    name = "a4_final_policy_is_prevalence_corrected"
    if decision_policy is None:
        return _skip(name, "decision_policy_results.json not found -- skipped.")
    if "naive" not in decision_policy or "corrected" not in decision_policy:
        return _fail(name, f"Expected both 'naive' and 'corrected' blocks, got keys: {list(decision_policy.keys())}")
    naive_rate = decision_policy["naive"].get("cost_model", {}).get("assumed_production_fraud_rate")
    corrected_rate = decision_policy["corrected"].get("cost_model", {}).get("assumed_production_fraud_rate")
    if naive_rate is not None:
        return _fail(name, f"'naive' block's cost_model.assumed_production_fraud_rate={naive_rate}, expected null (reweighting disabled) so it's a genuine unweighted contrast.")
    if corrected_rate is None:
        return _fail(name, "'corrected' block's cost_model.assumed_production_fraud_rate is null -- it is not actually the prevalence-corrected policy.")
    return _ok(
        name,
        f"'naive' uses no reweighting (assumed_production_fraud_rate=null) and "
        f"'corrected' (the deployed policy) uses assumed_production_fraud_rate="
        f"{corrected_rate} -- the two are a genuine, documented contrast."
    )


def check_a5_nested_fold_honest_disclosure(decision_policy: dict | None) -> dict:
    """Section A's 'do not optimize thresholds directly on a final
    evaluation/test set' requirement. optimize_thresholds() does select and
    judge the 'corrected' pair on the same population (a real, documented
    limitation -- see decision_policy.py's own methodology_note), so this
    audit does not pretend that pair is nested/held-out. What it verifies
    is that (a) the fold-honest nested estimate required to responsibly
    report this limitation is actually present, internally consistent
    (every row appears in exactly one held-out fold), and (b) the
    methodology_note disclosing the limitation is present and non-empty --
    i.e. the artifact is honest about what 'corrected' is and is not an
    estimate of, rather than silently presenting it as already fold-honest.
    """
    name = "a5_nested_fold_honest_disclosure"
    if decision_policy is None:
        return _skip(name, "decision_policy_results.json not found -- skipped.")
    nested = decision_policy.get("nested_fold_honest_estimate")
    if not nested:
        return _fail(name, "'nested_fold_honest_estimate' block is missing.")
    fold_thresholds = nested.get("fold_thresholds")
    if not fold_thresholds:
        return _fail(name, "'nested_fold_honest_estimate.fold_thresholds' is missing or empty.")

    # Every outer fold partitions the SAME population into (selection,
    # holdout), so n_selection_rows + n_holdout_rows must be identical
    # across every fold -- and the holdouts, which are disjoint and
    # exhaustive by construction (stable_kfold_split), must sum to that
    # same population size.
    fold_totals = [f.get("n_selection_rows", 0) + f.get("n_holdout_rows", 0) for f in fold_thresholds]
    population_size = fold_totals[0]
    if len(set(fold_totals)) > 1:
        return _fail(
            name,
            f"fold_thresholds' (n_selection_rows + n_holdout_rows) totals are not "
            f"identical across folds: {fold_totals}. Every outer fold should "
            f"partition the same population."
        )
    total_holdout = sum(f.get("n_holdout_rows", 0) for f in fold_thresholds)
    if total_holdout != population_size:
        return _fail(
            name,
            f"Fold holdout rows don't sum to the full population: "
            f"sum(n_holdout_rows)={total_holdout} vs population size "
            f"(n_selection_rows+n_holdout_rows per fold)={population_size}. "
            f"Every row should appear in exactly one outer fold's holdout."
        )
    note = decision_policy.get("methodology_note", {}).get("threshold_selection_optimism", "")
    if not note or "optimism" not in note.lower():
        return _fail(
            name,
            "methodology_note.threshold_selection_optimism is missing or doesn't "
            "actually disclose the threshold-selection optimism issue."
        )
    return _ok(
        name,
        f"nested_fold_honest_estimate present with {len(fold_thresholds)} outer "
        f"fold(s) covering all {total_holdout} rows exactly once, and "
        f"methodology_note discloses the threshold-selection-optimism limitation "
        f"of the single full-population 'corrected' pair."
    )


def section_a(decision_policy: dict | None) -> list[dict]:
    return [
        check_a1_threshold_ordering_and_bounds(decision_policy),
        check_a2_liability_breakdown_complete(decision_policy),
        check_a3_prevalence_represented(decision_policy),
        check_a4_final_policy_is_prevalence_corrected(decision_policy),
        check_a5_nested_fold_honest_disclosure(decision_policy),
    ]


# ===========================================================================
# B. ARTIFACT CONSISTENCY -- delegate entirely to consistency_check.py
# ===========================================================================
def section_b(misses: list[dict], decision_policy: dict | None, case_reports: dict | None) -> list[dict]:
    """Reuses consistency_check.py's own check functions verbatim (import
    consistency_check as cc at module scope above) -- Phase 4D does not
    re-implement any of this cross-artifact comparison logic a second time.
    """
    checks = [
        cc.check_threshold_agreement(misses, decision_policy),
        cc.check_case_reports_threshold_agreement(case_reports, decision_policy),
        cc.check_miss_completeness(misses, case_reports),
    ]
    score_check, label_check = cc.check_shared_trace_consistency(misses, case_reports)
    checks.append(score_check)
    checks.append(label_check)
    return checks


# ===========================================================================
# C. VALIDATION CACHE INTEGRITY
# ===========================================================================
def check_c1_cache_self_identifies(cache_path: Path) -> dict:
    name = "c1_cache_self_identifying_variant"
    if not cache_path.exists():
        return _skip(name, f"{cache_path.name} not found -- skipped.")
    data = np.load(cache_path, allow_pickle=False)
    if "validation_variant" not in data.files:
        return _fail(
            name,
            f"{cache_path.name} has no 'validation_variant' field -- this is "
            f"exactly the pre-Phase-4C state that caused the B-2 cache-identity "
            f"bug (silent cross-consumption between get_validation_data() and "
            f"get_validation_data_fused())."
        )
    variant = str(data["validation_variant"])
    if variant not in dp.VALID_VALIDATION_VARIANTS:
        return _fail(name, f"validation_variant={variant!r} is not one of {dp.VALID_VALIDATION_VARIANTS}.")
    return _ok(name, f"{cache_path.name} self-identifies as validation_variant={variant!r}.")


def check_c2_cache_variant_cannot_be_silently_cross_consumed(cache_path: Path) -> dict:
    """Regression guard for the confirmed Phase 4C B-2 root cause: a cache
    written for one validation_variant must be REFUSED (raise
    ValidationCacheMismatch), never silently accepted, by a caller
    requesting the other variant. This copies the REAL cache to a scratch
    temp file and probes it there -- it never writes to, mutates, or
    deletes the actual repo file, keeping this audit read-only.
    """
    name = "c2_cache_variant_cross_consumption_guard"
    if not cache_path.exists():
        return _skip(name, f"{cache_path.name} not found -- skipped.")
    data = np.load(cache_path, allow_pickle=False)
    if "validation_variant" not in data.files:
        return _skip(name, f"{cache_path.name} has no validation_variant tag -- covered by c1, skipping cross-consumption probe.")
    actual_variant = str(data["validation_variant"])
    other_variant = "cascade" if actual_variant == "fused" else "fused"

    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir) / "decision_policy_validation_cache.npz"
        np.savez(
            scratch, y=data["y"], proba=data["proba"], dollars=data["dollars"],
            validation_variant=actual_variant,
        )
        original_cache_path = dp.CACHE_PATH
        try:
            dp.CACHE_PATH = scratch
            try:
                dp.load_cached_validation_data(other_variant)
            except dp.ValidationCacheMismatch:
                pass
            else:
                return _fail(
                    name,
                    f"load_cached_validation_data({other_variant!r}) silently "
                    f"succeeded against a cache tagged validation_variant="
                    f"{actual_variant!r} -- this is a regression of the Phase 4C "
                    f"B-2 cache-identity bug."
                )
            # Sanity: requesting the CORRECT variant must still succeed.
            try:
                dp.load_cached_validation_data(actual_variant)
            except dp.ValidationCacheMismatch as e:
                return _fail(
                    name,
                    f"load_cached_validation_data({actual_variant!r}) unexpectedly "
                    f"raised against a cache genuinely tagged {actual_variant!r}: {e}"
                )
        finally:
            dp.CACHE_PATH = original_cache_path

    return _ok(
        name,
        f"A cache tagged validation_variant={actual_variant!r} is correctly "
        f"REFUSED (ValidationCacheMismatch) when {other_variant!r} is requested, "
        f"and correctly ACCEPTED when {actual_variant!r} is requested."
    )


def check_c3_cache_matches_decision_policy_variant(cache_path: Path, decision_policy: dict | None) -> dict:
    """decision_policy.py's own main() always calls
    get_validation_data_fused() (score_source.score == 'risk_fusion_stacked_lr'),
    so the cache that produced decision_policy_results.json should be
    tagged 'fused' if it is still the same run's cache."""
    name = "c3_cache_variant_matches_decision_policy_score_source"
    if decision_policy is None or not cache_path.exists():
        return _skip(name, "decision_policy_results.json and/or the cache file not found -- skipped.")
    data = np.load(cache_path, allow_pickle=False)
    if "validation_variant" not in data.files:
        return _skip(name, "cache has no validation_variant tag -- covered by c1.")
    cache_variant = str(data["validation_variant"])
    score_source = decision_policy.get("score_source", {}).get("score")
    if score_source == "risk_fusion_stacked_lr" and cache_variant != "fused":
        return _fail(
            name,
            f"decision_policy_results.json.score_source.score="
            f"'risk_fusion_stacked_lr' (i.e. it was produced by "
            f"get_validation_data_fused()), but the on-disk cache is currently "
            f"tagged validation_variant={cache_variant!r} -- the cache has since "
            f"been overwritten by a different script/run (e.g. miss_collector.py "
            f"or decision_policy.get_validation_data()) and no longer reflects "
            f"the run that produced decision_policy_results.json. This is not "
            f"unsafe (readers still refuse a mismatched variant per c2), but it "
            f"does mean the cache can no longer be used to cheaply reproduce "
            f"this exact decision_policy_results.json without a fresh run."
        )
    return _ok(
        name,
        f"Cache variant ({cache_variant!r}) is consistent with "
        f"decision_policy_results.json's recorded score_source "
        f"({score_source!r})."
    )


def section_c(cache_path: Path, decision_policy: dict | None) -> list[dict]:
    return [
        check_c1_cache_self_identifies(cache_path),
        check_c2_cache_variant_cannot_be_silently_cross_consumed(cache_path),
        check_c3_cache_matches_decision_policy_variant(cache_path, decision_policy),
    ]


# ===========================================================================
# D. PROVENANCE INTEGRITY -- delegate to consistency_check.py, add the
#    freeze-procedure note as separate (non-check) informational output.
# ===========================================================================
def section_d(decision_policy: dict | None, misses_meta: dict | None, case_reports: dict | None) -> list[dict]:
    return [cc.check_provenance_agreement(decision_policy, misses_meta, case_reports)]


# ===========================================================================
# E. ONLINE/OFFLINE BOUNDARY
# ===========================================================================
def _load_api_modules():
    """Imports web_prototype/api's schemas + inference modules the same way
    the API's own test suite does (see web_prototype/api/tests/conftest.py):
    by inserting web_prototype/api and src onto sys.path. Returns
    (schemas_module, inference_module) or raises ImportError, which callers
    turn into a SKIPPED check rather than an audit crash -- the API stack
    (fastapi/pydantic-settings) may legitimately not be installed in every
    environment this audit runs in."""
    api_dir = REPO_ROOT / "web_prototype" / "api"
    src_dir = REPO_ROOT / "src"
    for p in (str(api_dir), str(src_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import schemas as api_schemas       # noqa
    import inference as api_inference   # noqa
    return api_schemas, api_inference


def check_e1_schema_never_types_offline_stages_as_available(api_schemas) -> dict:
    """Static/structural check: the response schema's stage3/4/5 fields
    must be typed as StageUnavailable (available defaults to False), never
    StageScore (available defaults to True) -- i.e. it's not merely a
    convention of what score_trace() currently returns, it's baked into
    the response CONTRACT itself."""
    name = "e1_schema_offline_stages_typed_unavailable"
    fields = api_schemas.ScoreResponse.model_fields
    problems = []
    for field_name in ("stage3_graph", "stage4_autoencoder", "stage5_fusion"):
        annotation = fields[field_name].annotation
        if annotation is not api_schemas.StageUnavailable:
            problems.append(f"{field_name} is typed {annotation!r}, expected StageUnavailable")
    if fields["stage1_2"].annotation is not api_schemas.StageScore:
        problems.append(f"stage1_2 is typed {fields['stage1_2'].annotation!r}, expected StageScore")
    default_available = api_schemas.StageUnavailable.model_fields["available"].default
    if default_available is not False:
        problems.append(f"StageUnavailable.available defaults to {default_available!r}, expected False")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(
        name,
        "ScoreResponse.stage3_graph/stage4_autoencoder/stage5_fusion are all "
        "typed StageUnavailable (available defaults False); stage1_2 is typed "
        "StageScore."
    )


def check_e2_score_trace_never_reports_offline_stages_available(api_inference) -> dict:
    """Runtime check against the REAL loaded model artifacts and a REAL Red
    Team corpus trace (same fixture pattern as
    web_prototype/api/tests/conftest.py's real_ato_trace) -- not a mock.
    Loads the registry read-only (ModelRegistry.load() only reads existing
    .joblib/.csv artifacts, it does not fit or write anything)."""
    name = "e2_score_trace_offline_stages_unavailable_at_runtime"
    corpus_path = REPO_ROOT / "reports" / "ato_corpus_raw.json"
    if not corpus_path.exists():
        return _skip(name, f"{corpus_path} not found -- skipped.")

    registry = api_inference.ModelRegistry()
    registry.load()
    if not registry.ready:
        return _skip(name, f"Model registry failed to load ({registry.load_error}) -- skipped.")

    with open(corpus_path) as f:
        corpus = json.load(f)
    if not corpus:
        return _skip(name, f"{corpus_path} is empty -- skipped.")
    trace = corpus[0]["observable_trace"]

    result = api_inference.score_trace(trace, registry)
    problems = []
    for stage in ("stage3_graph", "stage4_autoencoder", "stage5_fusion"):
        if result[stage]["available"] is not False:
            problems.append(f"{stage}['available'] = {result[stage]['available']!r}, expected False")
        if not result[stage].get("reason"):
            problems.append(f"{stage} has no 'reason' explaining unavailability")
    if result["stage1_2"]["available"] is not True:
        problems.append(f"stage1_2['available'] = {result['stage1_2']['available']!r}, expected True")
    if result["decision"] not in ("ALLOW", "REVIEW", "BLOCK"):
        problems.append(f"decision={result['decision']!r} is not one of ALLOW/REVIEW/BLOCK")
    if not (0.0 <= result["risk_score"] <= 1.0):
        problems.append(f"risk_score={result['risk_score']} not in [0, 1]")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(
        name,
        f"score_trace() on a real ATO corpus trace reports stage3_graph/"
        f"stage4_autoencoder/stage5_fusion as available=False (with a reason "
        f"each), stage1_2 as available=True, and decision={result['decision']!r}."
    )


def section_e() -> list[dict]:
    name_import_fail = "e_online_offline_boundary"
    try:
        api_schemas, api_inference = _load_api_modules()
    except ImportError as e:
        return [_skip(name_import_fail, f"Could not import web_prototype/api modules ({e}) -- skipped.")]
    return [
        check_e1_schema_never_types_offline_stages_as_available(api_schemas),
        check_e2_score_trace_never_reports_offline_stages_available(api_inference),
    ]


# ===========================================================================
# F. THRESHOLD/ARTIFACT AGREEMENT (Stage 4 decision threshold)
# ===========================================================================
def check_f1_stage4_decision_threshold_agreement() -> dict:
    """Neither stage4_autoencoder_results.json nor
    three_stage_cascade_results.json stores an explicit threshold pair --
    both are metric reports (confusion matrices / precision-recall) computed
    with the single scalar btp.CONFIG['DECISION_THRESHOLD'], imported
    unchanged into cascade_with_graph.py and cascade_with_autoencoder.py as
    each module's own DECISION_THRESHOLD, and into the web API's
    ModelRegistry.decision_threshold. This checks that constant has not
    drifted into two different values across the modules that report
    Stage 1+2(+3)(+4) metrics and the ones that police the API's
    Stage-1+2-only online decision -- i.e. every consumer is still scoring
    against the same cutoff, so their respective confusion matrices /
    online decisions remain comparable/consistent with each other."""
    name = "f1_stage4_decision_threshold_single_source_of_truth"
    try:
        import cascade_with_graph as cwg
        import cascade_with_autoencoder as cae
    except ImportError as e:
        return _skip(name, f"Could not import cascade modules ({e}) -- skipped.")

    values = {
        "blue_team_pipeline.CONFIG['DECISION_THRESHOLD']": btp.CONFIG["DECISION_THRESHOLD"],
        "cascade_with_graph.DECISION_THRESHOLD": cwg.DECISION_THRESHOLD,
        "cascade_with_autoencoder.DECISION_THRESHOLD": cae.DECISION_THRESHOLD,
    }
    try:
        api_schemas, api_inference = _load_api_modules()
        registry = api_inference.ModelRegistry()
        registry.load()
        if registry.ready:
            values["web_prototype/api ModelRegistry.decision_threshold"] = registry.decision_threshold
    except ImportError:
        pass  # API stack optional; still check the three core modules above.

    distinct = set(values.values())
    if len(distinct) > 1:
        return _fail(
            name,
            f"DECISION_THRESHOLD has drifted into {len(distinct)} different "
            f"values across consumers: {values}. Stage 1+2(+3)(+4) metric "
            f"reports and the online API decision rule are no longer using a "
            f"consistent cutoff."
        )
    return _ok(
        name,
        f"All {len(values)} consumer(s) agree on DECISION_THRESHOLD="
        f"{next(iter(distinct))}: {list(values.keys())}."
    )


def section_f() -> list[dict]:
    return [check_f1_stage4_decision_threshold_agreement()]


# ===========================================================================
# G. MISS/CORPUS ACCOUNTING
# ===========================================================================
MISS_RECORD_REQUIRED_KEYS = {
    "trace_id", "customer_id", "attack_family", "attack_difficulty",
    "final_decision", "final_score", "t_review", "t_block",
    "dollars_in_trace", "stage1_escalated_to_ml", "graph_connected",
    "stage1_rules_checked", "behavioral_features", "reason_for_miss",
}


def check_g1_misses_structurally_valid(misses: list[dict]) -> dict:
    name = "g1_misses_jsonl_structurally_valid"
    if not misses:
        return _skip(name, "misses.jsonl not found or empty -- skipped.")
    problems = []
    for i, m in enumerate(misses):
        missing = MISS_RECORD_REQUIRED_KEYS - set(m.keys())
        if missing:
            problems.append(f"record {i} ({m.get('trace_id', '?')}) missing keys: {sorted(missing)}")
            continue
        if m["final_decision"] != "ALLOW":
            problems.append(f"record {i} ({m['trace_id']}) final_decision={m['final_decision']!r}, expected 'ALLOW' (misses.jsonl is ALLOW-only by construction)")
        if not (0.0 <= m["final_score"] <= 1.0):
            problems.append(f"record {i} ({m['trace_id']}) final_score={m['final_score']} not in [0, 1]")
        if not (m["t_review"] <= m["t_block"]):
            problems.append(f"record {i} ({m['trace_id']}) t_review ({m['t_review']}) > t_block ({m['t_block']})")
        if m["final_score"] >= m["t_review"]:
            problems.append(f"record {i} ({m['trace_id']}) final_score ({m['final_score']}) >= t_review ({m['t_review']}) -- should not have resolved to ALLOW")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(name, f"All {len(misses)} miss record(s) carry the required fields and are internally consistent (ALLOW, score < t_review <= t_block).")


def check_g2_miss_attack_families_valid(misses: list[dict]) -> dict:
    name = "g2_miss_attack_family_labels_valid"
    if not misses:
        return _skip(name, "misses.jsonl not found or empty -- skipped.")
    bad = sorted({m.get("attack_family") for m in misses} - EXPECTED_ATTACK_FAMILIES)
    if bad:
        return _fail(
            name,
            f"misses.jsonl contains attack_family value(s) outside the "
            f"supported set: {bad} (supported: {sorted(EXPECTED_ATTACK_FAMILIES)})."
        )
    return _ok(name, f"Every miss's attack_family is one of the supported families: {sorted(EXPECTED_ATTACK_FAMILIES)}.")


def check_g3_supported_families_unchanged() -> dict:
    name = "g3_supported_attack_families_unchanged"
    expected = {"ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"}
    if EXPECTED_ATTACK_FAMILIES != expected:
        return _fail(
            name,
            f"decision_policy.LIABILITY_SIDE's keys have changed to "
            f"{sorted(EXPECTED_ATTACK_FAMILIES)}, expected exactly {sorted(expected)}."
        )
    return _ok(name, f"decision_policy.LIABILITY_SIDE still covers exactly {sorted(expected)}.")


def section_g(misses: list[dict]) -> list[dict]:
    return [
        check_g1_misses_structurally_valid(misses),
        check_g2_miss_attack_families_valid(misses),
        check_g3_supported_families_unchanged(),
    ]


# ===========================================================================
# H. REPRODUCIBILITY / DETERMINISM
# ===========================================================================
def check_h1_stable_fold_id_deterministic() -> dict:
    name = "h1_stable_fold_id_deterministic"
    trace_ids = ["atk-1179705d", "legit_sess_edd9f6a1-5a23-4f2c-8f28-370b2cd0aeb2", "atk-693a3017"]
    seed = btp.CONFIG["RANDOM_STATE"]
    first = [btp.stable_fold_id(t, seed) for t in trace_ids]
    second = [btp.stable_fold_id(t, seed) for t in trace_ids]
    if first != second:
        return _fail(name, f"stable_fold_id was not deterministic across repeated calls: {first} vs {second}")
    # Different seeds should (almost always) produce a different partition --
    # not required to differ for every single id, but not all ids should
    # collide either, or the salt isn't doing anything.
    alt = [btp.stable_fold_id(t, seed + 1) for t in trace_ids]
    if alt == first:
        return _fail(name, "stable_fold_id produced identical hashes for two different random_state seeds -- the seed salt appears to have no effect.")
    return _ok(name, "stable_fold_id(trace_id, seed) is deterministic for a fixed seed and seed-sensitive across seeds.")


def check_h2_stable_kfold_split_deterministic() -> dict:
    name = "h2_stable_kfold_split_deterministic"
    import pandas as pd
    df = pd.DataFrame({
        "trace_id": [f"trace-{i:03d}" for i in range(40)],
        "fraud": [1 if i % 5 == 0 else 0 for i in range(40)],
    })
    seed = btp.CONFIG["RANDOM_STATE"]
    folds_a = btp.stable_kfold_split(df, "fraud", n_splits=5, random_state=seed)
    folds_b = btp.stable_kfold_split(df, "fraud", n_splits=5, random_state=seed)
    if len(folds_a) != len(folds_b):
        return _fail(name, f"Different number of folds across identical calls: {len(folds_a)} vs {len(folds_b)}")
    for i, ((tr_a, te_a), (tr_b, te_b)) in enumerate(zip(folds_a, folds_b)):
        if not (np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)):
            return _fail(name, f"Fold {i} differs across identical stable_kfold_split calls (train/test indices not reproduced).")
    # Membership must be keyed on trace_id, not row position: shuffling the
    # df's rows should not change which rows land in which fold, unlike
    # StratifiedKFold's positional partitioning (the bug stable_fold fixes).
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    folds_shuffled = btp.stable_kfold_split(shuffled, "fraud", n_splits=5, random_state=seed)
    orig_fold_of_id = {}
    for fold_i, (_, test_idx) in enumerate(folds_a):
        for i in test_idx:
            orig_fold_of_id[df.at[i, "trace_id"]] = fold_i
    shuf_fold_of_id = {}
    for fold_i, (_, test_idx) in enumerate(folds_shuffled):
        for i in test_idx:
            shuf_fold_of_id[shuffled.at[i, "trace_id"]] = fold_i
    if orig_fold_of_id != shuf_fold_of_id:
        return _fail(name, "Row order changed a trace_id's fold assignment -- stable_kfold_split appears positionally dependent, defeating its purpose.")
    return _ok(name, "stable_kfold_split is deterministic for a fixed (df, seed) and each trace_id's fold assignment is invariant to row order/position.")


def section_h() -> list[dict]:
    return [
        check_h1_stable_fold_id_deterministic(),
        check_h2_stable_kfold_split_deterministic(),
    ]


# ===========================================================================
# Main
# ===========================================================================
def run_all_checks() -> dict:
    decision_policy = _load_json(DECISION_POLICY_PATH)
    misses = _load_jsonl(MISSES_PATH)
    misses_meta = _load_jsonl_metadata(MISSES_PATH)
    case_reports = _load_json(CASE_REPORTS_PATH)

    checks = []
    checks += section_a(decision_policy)
    checks += section_b(misses, decision_policy, case_reports)
    checks += section_c(CACHE_PATH, decision_policy)
    checks += section_d(decision_policy, misses_meta, case_reports)
    checks += section_e()
    checks += section_f()
    checks += section_g(misses)
    checks += section_h()

    any_fail = any(c["passed"] is False for c in checks)
    any_run = any(c["passed"] is not None for c in checks)
    verdict = "FAIL" if any_fail else ("PASS" if any_run else "NO_CHECKS_RAN")

    return {
        "verdict": verdict,
        "freeze_procedure_note": FREEZE_PROCEDURE_NOTE,
        "checks": checks,
    }


def main() -> int:
    result = run_all_checks()

    print("=" * 72)
    print("PHASE 4D -- BLUE TEAM INTEGRITY / FREEZE AUDIT")
    print("=" * 72)
    for c in result["checks"]:
        status = "SKIPPED" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
        print(f"\n[{status}] {c['name']}")
        for line in c["details"]:
            print(f"    {line}")

    print("\n" + "=" * 72)
    print(f"VERDICT: {result['verdict']}")
    if result["verdict"] == "FAIL":
        print("One or more invariants failed. See FAIL entries above.")
    elif result["verdict"] == "NO_CHECKS_RAN":
        print("No checks could run -- generate the pipeline artifacts first "
              "(decision_policy.py, miss_collector.py, explainability.py).")
    print("=" * 72)
    print(f"\nFREEZE PROCEDURE:\n{result['freeze_procedure_note']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull results written to {RESULTS_PATH}")

    return 1 if result["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
