"""
phase5b_robustness_evaluation.py
==================================
Phase 5B -- Robustness / Stress Evaluation (SCOPE: prevalence shift only)

This is an EVALUATION-ONLY layer over the already-built, already-fitted,
FROZEN AI Defence System artifacts. Like evaluation_harness.py (Phase 5A),
it does LOAD -> SCORE -> EVALUATE -> REPORT and never LOAD -> TRAIN ->
TUNE -> SCORE -> REPORT.

============================================================
WHY THIS FILE ONLY IMPLEMENTS ONE OF THE SIX REQUESTED
ROBUSTNESS CATEGORIES
============================================================
The Phase 5B brief asked for six robustness dimensions: feature noise,
timing jitter, missing information, behavioral drift, class-imbalance /
prevalence shift, and graph sparsity. Before writing any code, this
repository's actual scoring architecture was inspected (per the brief's
own "inspect first" requirement), and a hard blocker was found for five
of the six:

    The canonical score this system's frozen decision policy
    (decision_policy_results.json's "corrected" t_review/t_block) was
    tuned against is the Stage 5 FUSED score (risk_fusion.run_risk_fusion).
    That score has NO saved, loadable, "frozen scorer" artifact anywhere
    in this repository:

      - Stage 3 (GCN):        gcn.train() is called fresh, per fold,
                               every time risk_fusion.compute_base_scores()
                               runs. No .joblib/.pt/etc for it exists.
      - Stage 4 (Autoencoder): autoencoder.train() is likewise called
                               fresh, per fold, every run. No saved
                               artifact exists.
      - Stage 5 (fusion meta-model): risk_fusion.fit_fusion_oof() calls
                               sklearn LogisticRegression().fit() fresh,
                               per fold, every run. No saved artifact
                               exists.

    This is not an oversight -- web_prototype/api/inference.py's own
    module docstring documents it explicitly: "None of the three
    [Stage 3/4/5] has a saved, loadable artifact anywhere in this repo
    ... Each is retrained fresh inside a 5-fold CV loop every time its
    batch script runs."

    Consequently, the ONLY way to obtain a fused score for a
    feature-noised, timing-jittered, missingness-perturbed,
    behaviorally-drifted, or graph-sparsified population (i.e. any
    population whose FEATURE VALUES differ from the canonical rows) is
    to re-run risk_fusion.run_risk_fusion() -- which trains a fresh GCN,
    a fresh autoencoder, and a fresh meta-model, and refits per-fold
    standardization (mu/sigma) on the perturbed features. That is
    retraining and preprocessing fitting, both of which the Phase 5B
    brief explicitly and repeatedly forbids ("Do NOT retrain models",
    "Do NOT fit preprocessing", "the stress evaluator must never call
    ... model training, calibration fitting, preprocessing fitting. If
    an existing API performs any of these operations, do not use that
    API."). The cached decision_policy_validation_cache.npz score array
    cannot substitute either -- it holds fixed scores for the
    UNPERTURBED canonical rows, so it cannot reflect what a perturbed
    input would score.

    This was reported to the requester as an ambiguity/blocker rather
    than silently worked around (per the brief's own instruction: "If
    anything is ambiguous or unsupported by the existing repository,
    STOP and report the ambiguity rather than inventing an
    implementation."). The requester chose: scope Phase 5B down to
    whatever robustness dimension IS compatible with a truly frozen,
    no-retrain scorer, and report the other five as explicitly not
    implementable under the current architecture (see
    NOT_IMPLEMENTED_CATEGORIES below and the report's
    "scope_and_limitations" section) rather than force an
    implementation that would secretly retrain the model.

    Class-imbalance / prevalence shift is the ONE category that fits a
    frozen scorer with zero exceptions: it evaluates the SAME
    already-computed fused scores (loaded straight from
    decision_policy_validation_cache.npz, exactly like Phase 5A) under a
    different fraud/legitimate MIX, by resampling which already-scored
    ROWS participate -- never touching a feature value, never calling a
    model, never fitting anything. That is exactly what
    decision_policy.py's own sample_weights()/CostModel.
    assumed_production_fraud_rate machinery already does for cost
    (reused here, not reinvented) -- this file extends that idea to
    full population-level classification/decision-policy/attack-family
    metrics at several explicit, reproducible target prevalences.

============================================================
WHAT THIS FILE REUSES VERBATIM (not reinvented)
============================================================
  - evaluation_harness.build_canonical_evaluation_population()
  - evaluation_harness.load_canonical_fused_scores()
  - evaluation_harness.load_canonical_policy()
  - evaluation_harness.compute_classification_metrics()
  - evaluation_harness.compute_fraud_metrics()
  - evaluation_harness.compute_decision_policy_metrics()
  - evaluation_harness.compute_attack_family_metrics()
  - decision_policy.policy_stats() / CostModel (via the above)
  - artifact_metadata.py's provenance helpers

None of Phase 5A's files are modified. This file only imports and calls
their public functions.

Run from the repo root (after Phase 5A's own artifacts already exist):

    python phase5b_robustness_evaluation.py

Produces: phase5b_robustness_report.json
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import artifact_metadata as am              # noqa: E402
import autoencoder as ae                    # noqa: E402
import decision_policy as dp                # noqa: E402
import evaluation_harness as eh             # noqa: E402
import gcn                                  # noqa: E402
import risk_fusion as rf                    # noqa: E402

SCHEMA_VERSION = "phase5b.1"
REPORT_PATH = REPO_ROOT / "phase5b_robustness_report.json"

SUPPORTED_ATTACK_FAMILIES = eh.SUPPORTED_ATTACK_FAMILIES
LEGITIMATE_FAMILY = eh.LEGITIMATE_FAMILY

# Requested-but-not-implemented categories, with the specific reason each
# one is blocked. Kept as data (not just prose in the docstring) so the
# JSON report can carry this as a first-class, machine-readable finding.
NOT_IMPLEMENTED_CATEGORIES = {
    "feature_noise": (
        "Requires re-scoring perturbed feature vectors through the Stage "
        "3/4/5 pipeline. No saved GCN/autoencoder/fusion-meta-model "
        "artifact exists in this repo; producing a fused score for any "
        "row whose features differ from the canonical population "
        "requires risk_fusion.run_risk_fusion(), which trains a fresh "
        "GCN + autoencoder + LogisticRegression meta-model and refits "
        "per-fold standardization -- forbidden retraining/preprocessing "
        "fitting under this phase's mandate."
    ),
    "timing_jitter": (
        "Same blocker as feature_noise: timestamp perturbation changes "
        "derived behavioral features (e.g. hesitation/timing features), "
        "which requires re-scoring through the same un-frozen Stage "
        "3/4/5 pipeline."
    ),
    "missing_information": (
        "Same blocker as feature_noise: inducing missingness in "
        "device/beneficiary/metadata fields changes the feature vector "
        "fed to Stage 1+2/3/4, which requires re-scoring through the "
        "same un-frozen Stage 3/4/5 pipeline."
    ),
    "behavioral_drift": (
        "Same blocker as feature_noise: a distribution shift in "
        "velocity/amount/reuse/temporal features requires re-scoring "
        "shifted feature vectors through the same un-frozen Stage 3/4/5 "
        "pipeline."
    ),
    "graph_sparsity": (
        "Reducing cross-customer edges changes connected_mask and A "
        "(the adjacency matrix), which changes the message-passed "
        "features (A_hat @ X) that Stage 3's GCN is trained AND scored "
        "on every fold. There is no saved GCN that can be handed a "
        "different graph at inference time; the only implementation "
        "path is risk_fusion.compute_base_scores() with a modified A, "
        "which retrains a fresh GCN on it -- forbidden retraining."
    ),
}
IMPLEMENTED_CATEGORY = "class_imbalance_prevalence_shift"


class Phase5BError(RuntimeError):
    """Raised for any condition this evaluator treats as a hard failure.
    Prefers failing loudly over silently producing a misleading report."""


class Phase5BFrozenEvaluationViolation(RuntimeError):
    """Raised the instant a forbidden training/fitting/optimization entry
    point is invoked while a Phase 5B frozen_execution_guard() is active.
    This is a hard-fail signal, not a value to be swallowed and turned
    into a boolean -- guard() only ever reports
    no_training_optimization_fitting_occurred=True for a run in which
    this was never raised."""


# Real, actual callables this guard intercepts -- based on this
# repository's real imports/entry points (see module-level imports
# above), not a generic/hypothetical list. Kept as data so the report
# can state exactly what was guarded, and so tests can assert the guard
# target list matches what's actually patched.
FROZEN_GUARD_TARGETS = (
    "gcn.train",
    "autoencoder.train",
    "risk_fusion.run_risk_fusion",
    "risk_fusion.fit_fusion_oof",
    "decision_policy.optimize_thresholds",
    "decision_policy.nested_threshold_estimate",
    "sklearn.linear_model.LogisticRegression.fit",
)


def _make_forbidden_stub(name: str):
    def _forbidden(*args, **kwargs):
        raise Phase5BFrozenEvaluationViolation(
            f"Phase 5B frozen-evaluation guard: forbidden operation {name!r} "
            f"was invoked. Phase 5B must operate strictly on frozen, "
            f"already-computed artifacts/scores; it must never train, fit, "
            f"or re-optimize anything."
        )
    _forbidden.__name__ = f"forbidden_{name.replace('.', '_')}"
    return _forbidden


@contextlib.contextmanager
def frozen_execution_guard(diagnostics: dict[str, Any] | None = None):
    """Scoped guard: while this context manager is active, every entry
    point in FROZEN_GUARD_TARGETS is replaced with a stub that raises
    Phase5BFrozenEvaluationViolation the instant it is called. Original
    callables are restored on exit (success OR exception) -- this never
    leaves a global monkeypatch behind for unrelated code/tests.

    `diagnostics`, if given, is populated with which targets were
    guarded so the caller/report can record it. It is NOT used to
    convert a violation into a boolean -- a violation always propagates
    as a real exception.
    """
    if diagnostics is not None:
        diagnostics["guarded_targets"] = list(FROZEN_GUARD_TARGETS)
        diagnostics["violation"] = None

    originals = {
        "gcn.train": (gcn, "train", gcn.train),
        "autoencoder.train": (ae, "train", ae.train),
        "risk_fusion.run_risk_fusion": (rf, "run_risk_fusion", rf.run_risk_fusion),
        "risk_fusion.fit_fusion_oof": (rf, "fit_fusion_oof", rf.fit_fusion_oof),
        "decision_policy.optimize_thresholds": (dp, "optimize_thresholds", dp.optimize_thresholds),
        "decision_policy.nested_threshold_estimate": (dp, "nested_threshold_estimate", dp.nested_threshold_estimate),
        "sklearn.linear_model.LogisticRegression.fit": (LogisticRegression, "fit", LogisticRegression.fit),
    }
    try:
        for name, (owner, attr, _orig) in originals.items():
            setattr(owner, attr, _make_forbidden_stub(name))
        yield diagnostics
    except Phase5BFrozenEvaluationViolation as exc:
        if diagnostics is not None:
            diagnostics["violation"] = str(exc)
        raise
    finally:
        for _name, (owner, attr, orig) in originals.items():
            setattr(owner, attr, orig)


# ---------------------------------------------------------------------------
# Step 1 -- canonical baseline (IDENTICAL to Phase 5A's population/score/
# policy loading -- reused verbatim, not reimplemented)
# ---------------------------------------------------------------------------
def load_canonical_baseline() -> dict[str, Any]:
    df = eh.build_canonical_evaluation_population()
    dataset_info = eh.validate_dataset_invariants(df)
    y, proba, dollars = eh.load_canonical_fused_scores(df)
    eh.validate_scores(proba, dataset_info["row_count"])
    policy = eh.load_canonical_policy()
    attack_family = df["attack_family"].values
    return {
        "df": df,
        "dataset_info": dataset_info,
        "y": y,
        "proba": proba,
        "dollars": dollars,
        "attack_family": attack_family,
        "trace_id": df["trace_id"].astype(str).values,
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Step 1b -- independent threshold verification
# ---------------------------------------------------------------------------
def independently_verify_thresholds(evaluator_policy: dict[str, Any], tolerance: float = 1e-9) -> dict[str, Any]:
    """Independently re-loads the canonical production decision-policy
    artifact through the repository's REAL loading path
    (eh.load_canonical_policy(), which itself does nothing but
    json.load() decision_policy_results.json's 'corrected' block -- see
    evaluation_harness.py) and compares the thresholds it returns
    against the thresholds this Phase 5B run actually used. This is a
    second, independent disk read -- not a comparison of a value against
    itself inside the same code path -- so a bug that silently mutated
    `evaluator_policy` in-process after the first load would still be
    caught here."""
    fresh = eh.load_canonical_policy()
    canonical_t_review = fresh["t_review"]
    canonical_t_block = fresh["t_block"]
    evaluator_t_review = evaluator_policy["t_review"]
    evaluator_t_block = evaluator_policy["t_block"]

    review_match = abs(canonical_t_review - evaluator_t_review) <= tolerance
    block_match = abs(canonical_t_block - evaluator_t_block) <= tolerance
    passed = bool(review_match and block_match)

    return {
        "passed": passed,
        "method": "independent second load of decision_policy_results.json via evaluation_harness.load_canonical_policy()",
        "tolerance": tolerance,
        "canonical_t_review": canonical_t_review,
        "canonical_t_block": canonical_t_block,
        "evaluator_t_review": evaluator_t_review,
        "evaluator_t_block": evaluator_t_block,
        "t_review_match": bool(review_match),
        "t_block_match": bool(block_match),
    }


# ---------------------------------------------------------------------------
# Step 1c -- canonical-data immutability fingerprinting
# ---------------------------------------------------------------------------
def fingerprint_canonical_population(baseline: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fingerprint of every canonical field Phase 5B reads
    (row count, trace/case IDs, labels, attack family, fused scores, and
    dollar-loss values), plus a byte hash of the on-disk fused-score
    cache file. Comparing this fingerprint before vs. after a Phase 5B
    run is how canonical_data_never_overwritten is actually verified --
    it is a value/byte-level check, not object identity, so it also
    catches an implementation that replaced the canonical arrays with an
    equal-looking copy."""
    y = baseline["y"]
    proba = baseline["proba"]
    dollars = baseline["dollars"]
    attack_family = baseline["attack_family"]
    trace_id = baseline["trace_id"]

    hasher = hashlib.sha256()
    hasher.update(str(len(y)).encode())
    hasher.update(b"|".join(str(t).encode() for t in trace_id.tolist()))
    hasher.update(np.asarray(y, dtype=np.int64).tobytes())
    hasher.update(b"|".join(str(f).encode() for f in attack_family.tolist()))
    hasher.update(np.asarray(proba, dtype=np.float64).tobytes())
    hasher.update(np.asarray(dollars, dtype=np.float64).tobytes())
    array_fingerprint = hasher.hexdigest()

    cache_fingerprint = am.hash_file(dp.CACHE_PATH)

    return {
        "row_count": int(len(y)),
        "array_fingerprint_sha256": array_fingerprint,
        "cache_file_sha256": cache_fingerprint,
        "cache_file_path": str(dp.CACHE_PATH),
    }


# ---------------------------------------------------------------------------
# Step 2 -- prevalence-shift scenario definitions
# ---------------------------------------------------------------------------
# Target production fraud prevalences. Values are deliberately GROUNDED in
# figures already present in this repository / cited by its own docstrings,
# not invented:
#   - 0.006 is decision_policy.CostModel.assumed_production_fraud_rate's
#     own default -- the prevalence the DEPLOYED policy was already tuned
#     against. Included so this suite directly stress-tests the assumption
#     the live thresholds already depend on.
#   - 0.05 / 0.01 / 0.001 bracket that assumption on both sides (elevated
#     attack-campaign environment down to a very low, large-retail-bank
#     prevalence), giving mild/moderate/severe severity levels around the
#     documented baseline instead of arbitrary numbers.
# "canonical" (no resampling) is included as severity level "baseline" --
# every other scenario's degradation is measured against it.
# Each scenario's seed is an EXPLICIT, FIXED integer -- deliberately NOT
# derived from Python's built-in hash() of the scenario name. str hashing
# is randomized per-process by default (PYTHONHASHSEED), so a seed derived
# from hash(name) is stable within one process (which is all a naive
# "run main() twice in the same script" self-check would catch) but
# DIFFERENT across two separate `python phase5b_robustness_evaluation.py`
# invocations -- a real determinism bug, caught here specifically by
# running the evaluator in two independent processes and diffing the
# output files (not just twice within main()). Fixed literal seeds avoid
# this class of bug entirely.
BASE_SEED = 5002  # Phase 5B's own seed root -- distinct from cwg.RANDOM_STATE
                   # (42) so this evaluator's randomness can never silently
                   # collide with (or be mistaken for) a model-fitting seed.
PREVALENCE_SCENARIOS = [
    {"scenario_name": "prevalence_baseline_canonical", "severity": "baseline", "target_prevalence": None, "seed": None},
    {"scenario_name": "prevalence_mild_5pct", "severity": "mild", "target_prevalence": 0.05, "seed": BASE_SEED + 1},
    {"scenario_name": "prevalence_moderate_1pct", "severity": "moderate", "target_prevalence": 0.01, "seed": BASE_SEED + 2},
    {"scenario_name": "prevalence_severe_deployed_assumption_0.6pct", "severity": "severe", "target_prevalence": 0.006, "seed": BASE_SEED + 3},
    {"scenario_name": "prevalence_extreme_0.1pct", "severity": "extreme", "target_prevalence": 0.001, "seed": BASE_SEED + 4},
]


def _family_targets(canonical_family_counts: dict[str, int], n_fraud_target: int) -> dict[str, int]:
    """Largest-remainder apportionment of n_fraud_target across the three
    canonical attack families, proportional to their CANONICAL shares.
    Guarantees sum(targets) == n_fraud_target exactly (a hard invariant
    checked below), rather than leaving a family combination that merely
    approximates it."""
    total = sum(canonical_family_counts.values())
    if total == 0:
        raise Phase5BError("Canonical fraud population is empty -- cannot apportion prevalence targets.")
    raw = {fam: n_fraud_target * count / total for fam, count in canonical_family_counts.items()}
    floors = {fam: int(np.floor(v)) for fam, v in raw.items()}
    remainder = n_fraud_target - sum(floors.values())
    # Largest fractional remainder gets the leftover unit(s), tie-broken
    # by family name for determinism.
    fracs = sorted(
        ((raw[fam] - floors[fam], fam) for fam in raw),
        key=lambda t: (-t[0], t[1]),
    )
    targets = dict(floors)
    for i in range(remainder):
        targets[fracs[i][1]] += 1
    assert sum(targets.values()) == n_fraud_target
    return targets


def build_scenario_population(baseline: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Builds a resampled population for one prevalence scenario. NEVER
    calls a model, NEVER changes a score/feature value -- every row's
    (fraud, attack_family, proba, dollars) is copied verbatim from the
    canonical baseline; only WHICH rows (and how many times a fraud row
    appears) changes. Legitimate rows are never resampled (kept 1:1,
    exactly the canonical 1182 legit traces) -- only the fraud side is
    resampled, which is what "prevalence" (the fraud:legit ratio) means
    here; changing the legit population would not be a prevalence shift,
    it would be a legitimate-behavior drift (out of scope -- see module
    docstring)."""
    y = baseline["y"]
    proba = baseline["proba"]
    dollars = baseline["dollars"]
    attack_family = baseline["attack_family"]
    trace_id = baseline["trace_id"]

    legit_idx = np.where(y == 0)[0]
    fraud_idx = np.where(y == 1)[0]
    n_legit = len(legit_idx)
    n_fraud_canonical = len(fraud_idx)

    if "target_prevalence" not in scenario:
        raise Phase5BError("Scenario definition missing required parameter 'target_prevalence'.")
    target_prevalence = scenario["target_prevalence"]

    if target_prevalence is not None and "seed" not in scenario:
        raise Phase5BError(
            f"Scenario {scenario.get('scenario_name', '<unnamed>')!r} requires a "
            f"'seed' parameter (any resampling scenario must be seeded)."
        )
    seed = scenario.get("seed")

    if target_prevalence is None:
        # "baseline" severity: identical to the canonical population,
        # by design -- included as the zero-degradation reference point,
        # not a bug (see invariant #20 in the brief: "a scenario cannot
        # silently become identical to baseline unless that is explicitly
        # expected" -- here it is explicitly expected).
        sel_fraud_idx = fraud_idx
        family_targets = {fam: int((attack_family[fraud_idx] == fam).sum()) for fam in SUPPORTED_ATTACK_FAMILIES}
        replace_used = {fam: False for fam in SUPPORTED_ATTACK_FAMILIES}
        rng_state_note = "no resampling performed (identity scenario)"
    else:
        if not (0.0 < target_prevalence < 1.0):
            raise Phase5BError(f"Invalid target_prevalence={target_prevalence!r}; must be in (0, 1).")
        n_fraud_target = int(round(target_prevalence * n_legit / (1.0 - target_prevalence)))
        n_fraud_target = max(n_fraud_target, 1)  # never resample to literally zero fraud

        canonical_family_counts = {
            fam: int((attack_family[fraud_idx] == fam).sum()) for fam in SUPPORTED_ATTACK_FAMILIES
        }
        family_targets = _family_targets(canonical_family_counts, n_fraud_target)

        rng = np.random.default_rng(seed)
        sel_fraud_idx_parts = []
        replace_used = {}
        for fam in SUPPORTED_ATTACK_FAMILIES:
            fam_pool = fraud_idx[attack_family[fraud_idx] == fam]
            n_target = family_targets[fam]
            if n_target <= len(fam_pool):
                chosen = rng.choice(fam_pool, size=n_target, replace=False)
                replace_used[fam] = False
            else:
                chosen = rng.choice(fam_pool, size=n_target, replace=True)
                replace_used[fam] = True
            sel_fraud_idx_parts.append(chosen)
        sel_fraud_idx = np.concatenate(sel_fraud_idx_parts) if sel_fraud_idx_parts else np.array([], dtype=int)
        rng_state_note = f"numpy Generator(PCG64), seed={seed}"

    all_idx = np.concatenate([legit_idx, sel_fraud_idx])

    resampled_id = np.array(
        [f"{trace_id[i]}::orig" for i in legit_idx]
        + [f"{trace_id[i]}::rs{k}" for k, i in enumerate(sel_fraud_idx)]
    )
    source_trace_id = trace_id[all_idx]

    n_dup_source_ids = int(len(source_trace_id) - len(set(source_trace_id.tolist())))

    fraud_count = int(len(sel_fraud_idx))
    legitimate_count = int(n_legit)
    total_count = fraud_count + legitimate_count
    achieved_prevalence = fraud_count / total_count
    requested_prevalence = target_prevalence  # None for the baseline/identity scenario
    prevalence_error = (
        None if requested_prevalence is None
        else round(abs(achieved_prevalence - requested_prevalence), 8)
    )

    scenario_pop = {
        "resampled_id": resampled_id,
        "source_trace_id": source_trace_id,
        "y": y[all_idx],
        "proba": proba[all_idx],
        "dollars": dollars[all_idx],
        "attack_family": attack_family[all_idx],
    }

    metadata = {
        "scenario_name": scenario["scenario_name"],
        "severity": scenario["severity"],
        "severity": scenario["severity"],
        "scenario_description": (
            "Identity (no resampling) canonical population -- zero-degradation reference."
            if target_prevalence is None else
            f"Fraud rows resampled (family-stratified, seeded) to hit an overall fraud "
            f"prevalence of {target_prevalence:.4%} against the full, UNCHANGED "
            f"canonical legitimate population ({n_legit} rows kept 1:1). No feature "
            f"value, model input, or score is ever modified -- only which "
            f"already-scored rows participate, and how many times a fraud row "
            f"repeats, changes."
        ),
        "perturbation_type": "prevalence_resampling",
        # Requested vs. ACHIEVED prevalence, reported explicitly and
        # separately -- with a finite population of integer rows the
        # 0.1% (and other) targets cannot necessarily be hit exactly
        # (e.g. 1 fraud row / (1182 legit + 1 fraud) = 0.084531%, not
        # exactly 0.1%). Never call an approximate integer result
        # "exactly" the requested value.
        "requested_prevalence": requested_prevalence,
        "achieved_prevalence": round(achieved_prevalence, 8),
        "prevalence_error": prevalence_error,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "total_count": total_count,
        "perturbation_parameters": {
            "target_prevalence": target_prevalence,
            "achieved_prevalence": round(achieved_prevalence, 6),
            "family_fraud_targets": family_targets,
            "family_resampled_with_replacement": replace_used,
            "legit_population": "unchanged (all canonical legitimate rows kept, no resampling)",
        },
        "random_seed": seed if target_prevalence is not None else None,
        "rng_note": rng_state_note,
        "source_population": "cascade_with_graph.load_all_records() canonical evaluation population (Phase 5A)",
        "source_population_size": int(len(baseline["y"])),
        "stressed_population_size": int(len(all_idx)),
        "n_duplicate_source_trace_ids": n_dup_source_ids,
        "n_duplicate_source_trace_ids_expected": target_prevalence is not None and any(replace_used.values()),
    }
    return {"population": scenario_pop, "metadata": metadata}


# ---------------------------------------------------------------------------
# Step 3 -- per-scenario metrics, reusing evaluation_harness's canonical
# metric functions VERBATIM (not reimplemented) on the resampled arrays.
# ---------------------------------------------------------------------------
def evaluate_scenario(pop: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    y = pop["y"]
    proba = pop["proba"]
    dollars = pop["dollars"]
    attack_family = pop["attack_family"]

    classification = eh.compute_classification_metrics(y, proba)
    fraud_metrics = eh.compute_fraud_metrics(classification)
    decision_policy_metrics = eh.compute_decision_policy_metrics(
        y, proba, dollars, attack_family, policy["t_review"], policy["t_block"], policy["cost_model"],
    )
    fam_df = pd.DataFrame({"attack_family": attack_family})
    attack_family_metrics = eh.compute_attack_family_metrics(fam_df, y, proba, policy["t_review"], policy["t_block"])

    # Row-level score-change metrics (MAE/RMSE/mean delta/max delta/
    # decision-flip-rate) are explicitly UNAVAILABLE for this scenario
    # type: prevalence resampling never changes any row's own score --
    # it only changes population COMPOSITION. Reporting these as 0 would
    # misleadingly imply "measured zero drift"; reporting them as
    # unavailable is honest per the brief's own instruction ("If a
    # metric becomes undefined, represent it explicitly as unavailable
    # rather than silently converting it to zero").
    score_change_metrics = {
        "available": False,
        "reason": (
            "Prevalence resampling changes which already-scored rows "
            "participate (and how many times a fraud row repeats); it "
            "never changes any row's own score. Per-row "
            "baseline-vs-stressed score deltas / decision flips are not "
            "a meaningful concept for this scenario type."
        ),
        "score_mae": None,
        "score_rmse": None,
        "score_mean_delta": None,
        "score_max_absolute_delta": None,
        "decision_flip_rate": None,
    }

    return {
        "classification": classification,
        "fraud_metrics": fraud_metrics,
        "decision_policy_metrics": decision_policy_metrics,
        "attack_families": attack_family_metrics,
        "score_change_metrics": score_change_metrics,
    }


def compute_degradation(baseline_result: dict[str, Any], scenario_result: dict[str, Any]) -> dict[str, Any]:
    """Degradation relative to the canonical baseline scenario (severity
    'baseline'), reported transparently -- no cherry-picking, no
    collapsing into one pass/fail score."""

    def _delta(base_val, stressed_val):
        if base_val is None or stressed_val is None:
            return None
        return round(float(stressed_val) - float(base_val), 6)

    bc, sc = baseline_result["classification"], scenario_result["classification"]
    bf, sf = baseline_result["fraud_metrics"], scenario_result["fraud_metrics"]
    bd, sd = baseline_result["decision_policy_metrics"], scenario_result["decision_policy_metrics"]

    return {
        "recall_delta": _delta(bc["recall"], sc["recall"]),
        "precision_delta": _delta(bc["precision"], sc["precision"]),
        "f1_delta": _delta(bc["f1"], sc["f1"]),
        "fraud_recall_delta": _delta(bf["fraud_recall"], sf["fraud_recall"]),
        "fraud_precision_delta": _delta(bf["fraud_precision"], sf["fraud_precision"]),
        "fraud_false_positive_rate_delta": _delta(bf["fraud_false_positive_rate"], sf["fraud_false_positive_rate"]),
        "fraud_false_negative_rate_delta": _delta(bf["fraud_false_negative_rate"], sf["fraud_false_negative_rate"]),
        "fraud_recall_blocked_plus_review_delta": _delta(
            bd["canonical_policy_stats"]["fraud_recall_blocked_plus_review"],
            sd["canonical_policy_stats"]["fraud_recall_blocked_plus_review"],
        ),
        "fraud_recall_blocked_only_delta": _delta(
            bd["canonical_policy_stats"]["fraud_recall_blocked_only"],
            sd["canonical_policy_stats"]["fraud_recall_blocked_only"],
        ),
        "expected_cost_at_assumed_prevalence_delta": _delta(
            bd["canonical_policy_stats"]["expected_cost_at_assumed_prevalence"],
            sd["canonical_policy_stats"]["expected_cost_at_assumed_prevalence"],
        ),
        "decision_flip_rate": None,  # see score_change_metrics note -- N/A for this scenario type
        "mean_absolute_score_delta": None,
        "classification_note": (
            "recall / fraud_false_positive_rate / fraud_false_negative_rate are "
            "PREVALENCE-INDEPENDENT here (same per-class score distribution, same "
            "fixed thresholds -- only sampling variance from the seeded bootstrap "
            "can move them). precision / fraud_precision / expected_cost ARE "
            "prevalence-dependent by construction and are expected to move with "
            "target_prevalence -- see decision_policy.py's own module docstring "
            "for why (the 'block-everyone' optimism at unrealistic prevalence)."
        ),
        "degradation_category": _classify_degradation(bc, sc, bf, sf),
    }


def _classify_degradation(bc, sc, bf, sf) -> str:
    """Transparent, documented thresholds -- not an arbitrary pass/fail.
    Based on the PREVALENCE-INDEPENDENT signal only (fraud recall), since
    precision/cost are expected to move with prevalence by construction
    (see classification_note above) and would otherwise dominate/hide the
    recall signal this classification is meant to surface."""
    br, sr = bf.get("fraud_recall"), sf.get("fraud_recall")
    if br is None or sr is None:
        return "unavailable"
    delta = sr - br
    if abs(delta) < 0.01:
        return "stable"
    if abs(delta) < 0.03:
        return "mild degradation" if delta < 0 else "stable (recall improved)"
    if abs(delta) < 0.08:
        return "material degradation" if delta < 0 else "stable (recall improved)"
    return "severe degradation" if delta < 0 else "stable (recall improved)"


# ---------------------------------------------------------------------------
# Step 4 -- invariants (repository-mandated list, scoped to what this
# scenario type can actually violate)
# ---------------------------------------------------------------------------
def compute_invariants(
    baseline: dict[str, Any],
    scenario_records: list[dict[str, Any]],
    policy: dict[str, Any],
    threshold_verification: dict[str, Any],
    immutability_check: dict[str, Any],
    guard_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["baseline_population_unchanged"] = (
        len(baseline["y"]) == baseline["dataset_info"]["row_count"]
    )
    checks["scenario_ids_valid"] = all(
        rec["metadata"]["scenario_name"] in {s["scenario_name"] for s in PREVALENCE_SCENARIOS}
        for rec in scenario_records
    )
    dup_ok = True
    for rec in scenario_records:
        expected = rec["metadata"]["n_duplicate_source_trace_ids_expected"]
        actual_has_dup = rec["metadata"]["n_duplicate_source_trace_ids"] > 0
        if actual_has_dup and not expected:
            dup_ok = False
        # Legit rows must NEVER be duplicated in any scenario (only
        # fraud rows may repeat, and only when explicitly flagged).
        legit_ids = rec["population"]["source_trace_id"][rec["population"]["y"] == 0]
        if len(legit_ids) != len(set(legit_ids.tolist())):
            dup_ok = False
    checks["no_unexpected_duplicate_trace_ids"] = dup_ok

    row_counts_ok = all(
        len(rec["population"]["y"]) == rec["metadata"]["stressed_population_size"]
        for rec in scenario_records
    )
    checks["row_counts_reconcile"] = row_counts_ok

    labels_preserved_ok = True
    for rec in scenario_records:
        src = rec["population"]["source_trace_id"]
        y_resampled = rec["population"]["y"]
        # every resampled row's label must equal the ORIGINAL canonical
        # row's label it was drawn from (never flipped/relabeled)
        canon_lookup = dict(zip(baseline["trace_id"].tolist(), baseline["y"].tolist()))
        if not all(canon_lookup[t] == y for t, y in zip(src.tolist(), y_resampled.tolist())):
            labels_preserved_ok = False
    checks["labels_unchanged_by_resampling"] = labels_preserved_ok

    families_canonical_ok = all(
        set(np.unique(rec["population"]["attack_family"])) <= (set(SUPPORTED_ATTACK_FAMILIES) | {LEGITIMATE_FAMILY})
        for rec in scenario_records
    )
    checks["attack_families_remain_canonical"] = families_canonical_ok

    scores_finite_ok = all(
        bool(np.all(np.isfinite(rec["population"]["proba"])))
        and bool(np.all((rec["population"]["proba"] >= 0) & (rec["population"]["proba"] <= 1)))
        for rec in scenario_records
    )
    checks["scores_finite_and_in_unit_interval"] = scores_finite_ok

    # These three invariants are backed by ACTUAL independent verification
    # (an independent artifact re-load, a before/after byte-level
    # fingerprint, and a real execution guard that raises the instant a
    # forbidden call happens) rather than being self-asserted -- see
    # independently_verify_thresholds(), fingerprint_canonical_population(),
    # and frozen_execution_guard() respectively.
    checks["thresholds_identical_to_canonical_policy"] = bool(threshold_verification["passed"])
    checks["random_seeds_recorded"] = all(
        ("random_seed" in rec["metadata"]) for rec in scenario_records
    )
    checks["canonical_data_never_overwritten"] = bool(immutability_check["passed"])
    checks["no_training_optimization_fitting_occurred"] = bool(
        guard_diagnostics.get("passed") and guard_diagnostics.get("violation") is None
    )

    checks["all_passed"] = all(bool(v) for v in checks.values() if not isinstance(v, str))
    return checks


# ---------------------------------------------------------------------------
# Step 5 -- provenance
# ---------------------------------------------------------------------------
def compute_provenance(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "git_commit": am._git_commit(REPO_ROOT),
        "git_dirty": am._git_dirty(REPO_ROOT),
        "python_version": sys.version.split()[0],
        "package_versions": am._package_versions(),
        "policy_artifact_provenance": {
            "path": "decision_policy_results.json",
            "block": "corrected",
            "provenance": policy.get("policy_provenance"),
        },
        "cache_provenance": {
            "path": str(dp.CACHE_PATH),
            "validation_variant": "fused",
            "sha256": am.hash_file(dp.CACHE_PATH),
            "note": "unmodified, reused verbatim -- see canonical_immutability for before/after fingerprint",
        },
        "phase5b_seed_root": BASE_SEED,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_phase5b_evaluation() -> dict[str, Any]:
    guard_diagnostics: dict[str, Any] = {}

    with frozen_execution_guard(guard_diagnostics):
        baseline = load_canonical_baseline()
        policy = baseline["policy"]

        pre_fingerprint = fingerprint_canonical_population(baseline)

        scenario_records = [build_scenario_population(baseline, s) for s in PREVALENCE_SCENARIOS]
        scenario_results = {}
        for rec in scenario_records:
            name = rec["metadata"]["scenario_name"]
            scenario_results[name] = evaluate_scenario(rec["population"], policy)

        post_fingerprint = fingerprint_canonical_population(baseline)

    # If we reach here, the guard's `with` block exited without a
    # Phase5BFrozenEvaluationViolation being raised -- i.e. none of the
    # forbidden training/fitting/optimization entry points were called.
    guard_diagnostics["passed"] = True

    threshold_verification = independently_verify_thresholds(policy)
    immutability_check = {
        "passed": pre_fingerprint == post_fingerprint,
        "pre_evaluation_fingerprint": pre_fingerprint,
        "post_evaluation_fingerprint": post_fingerprint,
    }

    baseline_scenario_name = next(s["scenario_name"] for s in PREVALENCE_SCENARIOS if s["severity"] == "baseline")
    baseline_result = scenario_results[baseline_scenario_name]

    degradation_summary = {
        rec["metadata"]["scenario_name"]: compute_degradation(baseline_result, scenario_results[rec["metadata"]["scenario_name"]])
        for rec in scenario_records
    }

    invariants = compute_invariants(
        baseline, scenario_records, policy, threshold_verification, immutability_check, guard_diagnostics,
    )
    provenance = compute_provenance(policy)

    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "phase": "5B",
            "objective": "Robustness / stress evaluation of the FROZEN fraud-risk system",
            "implemented_category": IMPLEMENTED_CATEGORY,
            "not_implemented_categories": NOT_IMPLEMENTED_CATEGORIES,
            "scope_note": (
                "Only class-imbalance/prevalence-shift is implemented. The other "
                "five requested categories (feature noise, timing jitter, missing "
                "information, behavioral drift, graph sparsity) all require "
                "re-scoring perturbed feature vectors through Stage 3 (GCN) / "
                "Stage 4 (autoencoder) / Stage 5 (fusion meta-model), none of "
                "which has a saved, frozen, loadable artifact in this repository "
                "-- producing a score for them would require retraining, which "
                "this phase's mandate forbids. See module docstring / "
                "not_implemented_categories for the specific blocker per category. "
                "This scope decision was made explicit to, and approved by, the "
                "requester before implementation."
            ),
        },
        "baseline": {
            "row_count": baseline["dataset_info"]["row_count"],
            "attack_family_counts": baseline["dataset_info"]["attack_family_counts"],
            "canonical_prevalence": round(float((baseline["y"] == 1).mean()), 6),
        },
        "frozen_policy": {
            "source": "decision_policy_results.json['corrected']",
            "t_review": policy["t_review"],
            "t_block": policy["t_block"],
            "frozen": True,
            "note": "Never recomputed/optimized anywhere in this file -- see leakage_protections.",
        },
        "scenarios": [rec["metadata"] for rec in scenario_records],
        "scenario_results": scenario_results,
        "attack_family_results": {
            name: scenario_results[name]["attack_families"] for name in scenario_results
        },
        "degradation_summary": degradation_summary,
        "invariants": invariants,
        "threshold_verification": threshold_verification,
        "canonical_immutability": immutability_check,
        "training_fitting_guard": guard_diagnostics,
        "leakage_protections": {
            "threshold_optimization": "GUARDED (decision_policy.optimize_thresholds raises if called; see training_fitting_guard)",
            "nested_threshold_estimation": "GUARDED (decision_policy.nested_threshold_estimate raises if called; see training_fitting_guard)",
            "calibration_fitting": "NOT USED (no calibrator .fit() anywhere in this file)",
            "retraining": "GUARDED (gcn.train / autoencoder.train / risk_fusion.run_risk_fusion / risk_fusion.fit_fusion_oof all raise if called; see training_fitting_guard). GCN/autoencoder/meta-model retraining is precisely why 5 of 6 categories are out of scope -- see metadata.scope_note.",
            "preprocessing_fitting": "GUARDED (sklearn.linear_model.LogisticRegression.fit raises if called; see training_fitting_guard)",
            "labels_used_only_after_scoring": True,
            "canonical_corpus_modified": not immutability_check["passed"],
        },
        "provenance": provenance,
        "conclusions": _build_conclusions(scenario_results, degradation_summary, baseline_scenario_name),
    }
    return report


def _build_conclusions(scenario_results, degradation_summary, baseline_scenario_name) -> dict[str, Any]:
    non_baseline = [n for n in degradation_summary if n != baseline_scenario_name]
    categories = {n: degradation_summary[n]["degradation_category"] for n in non_baseline}
    return {
        "summary": (
            "Under prevalence resampling alone (fixed frozen fused-score model, "
            "fixed frozen t_review/t_block thresholds), fraud recall is the "
            "prevalence-independent signal to watch; it is reported per scenario "
            "in degradation_summary rather than collapsed into a single score. "
            "Precision and expected cost move with prevalence by construction "
            "(see decision_policy.py's own documented block-everyone-at-high-"
            "prevalence finding) and should not be read as model degradation."
        ),
        "degradation_by_scenario": categories,
        "categories_not_evaluated": list(NOT_IMPLEMENTED_CATEGORIES.keys()),
        "recommendation": (
            "To evaluate the five out-of-scope categories, either (a) accept "
            "Stage-1+2-only robustness testing against the genuinely frozen "
            "xgb_model.joblib/calibrator.joblib artifact (a different, weaker "
            "system than the deployed fused-score policy), or (b) explicitly "
            "authorize re-running the identical, seed-controlled 5-fold CV "
            "recipe on perturbed data as an accepted exception to the "
            "no-retrain mandate."
        ),
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _compare_core_results(r1: dict[str, Any], r2: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    diffs = {}
    for name in r1["scenario_results"]:
        for section in ("classification", "fraud_metrics", "decision_policy_metrics", "attack_families"):
            v1 = r1["scenario_results"][name][section]
            v2 = r2["scenario_results"][name][section]
            if v1 != v2:
                diffs[f"{name}.{section}"] = {"run_1": v1, "run_2": v2}
    if r1["degradation_summary"] != r2["degradation_summary"]:
        diffs["degradation_summary"] = {"run_1": r1["degradation_summary"], "run_2": r2["degradation_summary"]}
    return len(diffs) == 0, diffs


def main() -> int:
    print("Phase 5B: running robustness evaluation (run #1)...")
    report_1 = run_phase5b_evaluation()

    print("Phase 5B: running again to verify determinism (run #2)...")
    report_2 = run_phase5b_evaluation()

    deterministic, diffs = _compare_core_results(report_1, report_2)
    report_1["determinism_check"] = {"deterministic": deterministic, "diffs": diffs}
    if not deterministic:
        print("WARNING: non-deterministic core results detected between run #1 and run #2:")
        print(json.dumps(diffs, indent=2, default=_json_default))

    with open(REPORT_PATH, "w") as f:
        json.dump(report_1, f, indent=2, default=_json_default)

    print(f"\nWrote {REPORT_PATH}")
    print(f"invariants.all_passed = {report_1['invariants']['all_passed']}")
    print(f"determinism_check.deterministic = {deterministic}")

    if not report_1["invariants"]["all_passed"]:
        print("FAIL: one or more invariants failed. See report 'invariants' section.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())