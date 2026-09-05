"""
evaluation_harness.py
======================
Phase 5A -- Formal Evaluation Harness

A READ-ONLY, LEAKAGE-SAFE, DETERMINISTIC evaluation layer over the
already-built, already-fitted AI Defence System artifacts. This file
does not train, refit, calibrate, or tune anything. Its job is:

    LOAD -> SCORE -> EVALUATE -> REPORT

not:

    LOAD -> TRAIN -> TUNE -> SCORE -> REPORT

WHAT IT LOADS (all pre-existing, canonical artifacts -- nothing here is
invented or recomputed from scratch):

  - The canonical evaluation population: cascade_with_graph.load_all_records()
    + cascade_with_graph.build_feature_table_and_graph(). This is pure data
    loading and feature EXTRACTION (btp.extract_features / btp.add_hesitation_delta),
    not model fitting -- it is the exact same population every other Phase 4
    script (decision_policy.py, explainability.py, miss_collector.py) scores
    against.

  - The canonical out-of-fold Stage 5 (fused) score, straight from
    decision_policy_validation_cache.npz via
    decision_policy.load_cached_validation_data("fused", y_check=...).
    This is the ONLY score this harness ever evaluates. It deliberately
    does NOT re-run risk_fusion.run_risk_fusion() / the 5-fold OOF
    cascade to regenerate this score, even though that is how the cache
    was originally produced -- doing so here would mean silently
    retraining a fresh GCN + stacked-LR meta-model per run, which this
    harness's no-retraining mandate forbids. If the cache is missing or
    stale (variant != "fused", or its y doesn't match this run's
    freshly-loaded population), this harness FAILS LOUDLY rather than
    regenerating it.

  - The canonical, already-finalized decision policy (thresholds +
    cost model) from decision_policy_results.json's "corrected" block.
    This harness NEVER calls decision_policy.optimize_thresholds() or
    any other threshold-selection routine.

  - decision_policy.py's own policy_stats() / expected_cost() /
    liability_breakdown() / CostModel -- reused verbatim, not
    reimplemented, so this harness's decision-policy numbers cannot
    silently drift from the canonical definitions.

Run from the repo root:

    python evaluation_harness.py

Produces: phase5a_evaluation_report.json
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import artifact_metadata as am              # noqa: E402
import blue_team_pipeline as btp            # noqa: E402
import cascade_with_graph as cwg            # noqa: E402
import decision_policy as dp                # noqa: E402

SCHEMA_VERSION = "phase5a.1"
REPORT_PATH = REPO_ROOT / "phase5a_evaluation_report.json"
POLICY_RESULTS_PATH = REPO_ROOT / "decision_policy_results.json"

# Supported canonical attack families. Any fraud row whose attack_family
# is not one of these (and is not "legitimate") is treated as a hard
# failure -- see validate_dataset_invariants().
SUPPORTED_ATTACK_FAMILIES = (
    "ACCOUNT_TAKEOVER",
    "AUTHORIZED_PUSH_PAYMENT",
    "MULE_NETWORK",
)
LEGITIMATE_FAMILY = "legitimate"

# Documented Phase 4 population (see Phase 5A prompt / STAGE_STATUS.md).
# NOT trusted blindly -- validated against the actual freshly-loaded
# canonical data in validate_dataset_invariants() below, and the
# ACTUAL measured counts (not these) are what the report contains.
DOCUMENTED_PHASE4_POPULATION = {
    "total": 1556,
    "ACCOUNT_TAKEOVER": 97,
    "AUTHORIZED_PUSH_PAYMENT": 156,
    "MULE_NETWORK": 121,
    "legitimate": 1182,
}

# Classification threshold used ONLY for the binary
# fraud/legitimate "classification" section (sec. 10). This is the
# same DECISION_THRESHOLD blue_team_pipeline/cascade_with_graph already
# use for their own "overall" 2-/3-stage reporting -- reused, not
# invented here.
CLASSIFICATION_THRESHOLD = btp.CONFIG["DECISION_THRESHOLD"]


class EvaluationHarnessError(RuntimeError):
    """Raised for any condition this harness treats as a hard failure
    (missing/invalid policy, cache mismatch, invariant violation,
    unsupported attack family, etc). The harness prefers failing loudly
    over silently producing a misleading report."""


# ---------------------------------------------------------------------------
# Step 1 -- canonical evaluation population (data load + feature
# extraction ONLY -- no model fitting anywhere in this function)
# ---------------------------------------------------------------------------
def build_canonical_evaluation_population() -> pd.DataFrame:
    """Reuses cascade_with_graph's canonical loaders verbatim. Returns a
    DataFrame with (at minimum) trace_id, fraud, attack_family columns,
    in the same row order load_all_records() produced them -- the same
    order decision_policy.get_validation_data_fused() built its cached
    y/proba/dollars arrays in, which is what makes positional alignment
    with the cache safe (see load_canonical_fused_scores() below)."""
    cfg = btp.CONFIG
    all_records = cwg.load_all_records(cfg)
    graph_connected_ids = cwg.get_graph_connected_trace_ids(all_records)
    df, _A, _connected_mask = cwg.build_feature_table_and_graph(all_records, graph_connected_ids)
    return df


def load_canonical_fused_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ONLY score this harness evaluates: the canonical out-of-fold
    Stage 5 (fused) score, read from decision_policy_validation_cache.npz.

    y_check=df["fraud"] forces decision_policy.load_cached_validation_data
    to verify the cache's labels match THIS run's freshly-loaded
    population row-for-row (same length, same values) before trusting
    its proba/dollars arrays are positionally aligned with `df`. Any
    mismatch (missing cache, wrong variant, stale/misaligned labels)
    raises ValidationCacheMismatch, which this harness re-raises as a
    loud EvaluationHarnessError rather than silently regenerating the
    cache (regenerating it would mean retraining the fusion model)."""
    y_check = df["fraud"].values.astype(int)
    try:
        y, proba, dollars = dp.load_cached_validation_data("fused", y_check=y_check)
    except dp.ValidationCacheMismatch as exc:
        raise EvaluationHarnessError(
            f"Cannot evaluate: canonical fused-score cache is unusable. {exc}"
        ) from exc
    return y, proba, dollars


# ---------------------------------------------------------------------------
# Step 2 -- canonical, already-finalized decision policy
# ---------------------------------------------------------------------------
def load_canonical_policy() -> dict[str, Any]:
    """Reads the deployed/corrected policy from decision_policy_results.json.
    NEVER calls optimize_thresholds() or any other threshold-selection
    routine -- this harness consumes an already-chosen policy, it does
    not choose one."""
    if not POLICY_RESULTS_PATH.exists():
        raise EvaluationHarnessError(
            f"{POLICY_RESULTS_PATH.name} does not exist -- cannot evaluate without "
            f"an already-finalized policy. This harness will not invent one."
        )
    with open(POLICY_RESULTS_PATH) as f:
        results = json.load(f)

    if "corrected" not in results:
        raise EvaluationHarnessError(
            f"{POLICY_RESULTS_PATH.name} has no 'corrected' block -- the deployed "
            f"policy variant this harness is required to use is missing."
        )
    corrected = results["corrected"]
    t_review = corrected.get("t_review")
    t_block = corrected.get("t_block")
    if t_review is None or t_block is None:
        raise EvaluationHarnessError("Corrected policy is missing t_review/t_block.")
    if not (0.0 <= t_review < t_block <= 1.0):
        raise EvaluationHarnessError(
            f"Invalid threshold pair from {POLICY_RESULTS_PATH.name}: "
            f"t_review={t_review}, t_block={t_block} (require 0 <= t_review < t_block <= 1)."
        )

    cost_model_dict = corrected.get("cost_model", {})
    cost_model = dp.CostModel(
        review_ops_cost=cost_model_dict.get("review_ops_cost", dp.CostModel.review_ops_cost),
        review_catch_rate=cost_model_dict.get("review_catch_rate", dp.CostModel.review_catch_rate),
        legit_block_friction_cost=cost_model_dict.get(
            "legit_block_friction_cost", dp.CostModel.legit_block_friction_cost
        ),
        assumed_production_fraud_rate=cost_model_dict.get(
            "assumed_production_fraud_rate", dp.CostModel.assumed_production_fraud_rate
        ),
        app_sending_liability_share=cost_model_dict.get(
            "app_sending_liability_share", dp.CostModel.app_sending_liability_share
        ),
    )

    score_source = results.get("score_source", {})
    cache_variant_required = "fused"
    # Sanity-check: the policy this harness is told to use was itself
    # selected against the "fused" score variant, matching the cache
    # variant this harness requires (sec. 16/6 cross-check).
    if score_source and "score" in score_source:
        if "fusion" not in str(score_source["score"]).lower() and "fused" not in str(score_source["score"]).lower():
            raise EvaluationHarnessError(
                f"{POLICY_RESULTS_PATH.name}'s score_source ({score_source['score']!r}) "
                f"does not look like a fused-score policy, but this harness requires "
                f"validation_variant='fused' from the cache. Refusing to mix policy "
                f"and score variants."
            )

    return {
        "t_review": float(t_review),
        "t_block": float(t_block),
        "cost_model": cost_model,
        "cost_model_raw": cost_model_dict,
        "score_source": score_source,
        "required_cache_variant": cache_variant_required,
        "policy_provenance": results.get("_artifact_metadata"),
        "methodology_note": results.get("methodology_note"),
    }


# ---------------------------------------------------------------------------
# Step 3 -- dataset-level invariants (sec. 8, 9, 13, 20)
# ---------------------------------------------------------------------------
def validate_dataset_invariants(df: pd.DataFrame) -> dict[str, Any]:
    trace_ids = df["trace_id"].astype(str)
    n = len(df)

    if trace_ids.isna().any() or (trace_ids.str.strip() == "").any():
        raise EvaluationHarnessError("Evaluation dataset contains missing/empty trace_id values.")

    n_unique = trace_ids.nunique()
    if n_unique != n:
        raise EvaluationHarnessError(
            f"Evaluation dataset contains duplicate trace_ids: {n} rows but only "
            f"{n_unique} unique trace_ids."
        )

    families = set(df["attack_family"].unique())
    unsupported = families - set(SUPPORTED_ATTACK_FAMILIES) - {LEGITIMATE_FAMILY}
    if unsupported:
        raise EvaluationHarnessError(
            f"Evaluation dataset contains unsupported attack family/families: "
            f"{sorted(unsupported)}. Supported: {SUPPORTED_ATTACK_FAMILIES} plus "
            f"{LEGITIMATE_FAMILY!r}."
        )

    fam_counts = df["attack_family"].value_counts().to_dict()
    measured = {
        "total": n,
        "ACCOUNT_TAKEOVER": int(fam_counts.get("ACCOUNT_TAKEOVER", 0)),
        "AUTHORIZED_PUSH_PAYMENT": int(fam_counts.get("AUTHORIZED_PUSH_PAYMENT", 0)),
        "MULE_NETWORK": int(fam_counts.get("MULE_NETWORK", 0)),
        "legitimate": int(fam_counts.get("legitimate", 0)),
    }
    matches_documented = measured == DOCUMENTED_PHASE4_POPULATION

    return {
        "row_count": n,
        "unique_trace_ids": int(n_unique),
        "attack_family_counts": measured,
        "matches_documented_phase4_population": matches_documented,
        "documented_phase4_population": DOCUMENTED_PHASE4_POPULATION,
    }


def validate_scores(proba: np.ndarray, n_rows: int) -> dict[str, Any]:
    proba = np.asarray(proba, dtype=float)
    if len(proba) != n_rows:
        raise EvaluationHarnessError(
            f"Score array length ({len(proba)}) does not match evaluation row count ({n_rows})."
        )
    finite_mask = np.isfinite(proba)
    n_finite = int(finite_mask.sum())
    if n_finite != len(proba):
        raise EvaluationHarnessError(
            f"{len(proba) - n_finite} score(s) are NaN/inf -- scores must all be finite."
        )
    in_range = (proba >= 0.0) & (proba <= 1.0)
    if not bool(in_range.all()):
        n_bad = int((~in_range).sum())
        raise EvaluationHarnessError(f"{n_bad} score(s) fall outside [0, 1].")

    return {
        "available_count": int(len(proba)),
        "unavailable_count": 0,
        "min": float(np.min(proba)),
        "max": float(np.max(proba)),
        "mean": float(np.mean(proba)),
        "median": float(np.median(proba)),
        "std": float(np.std(proba)),
    }


# ---------------------------------------------------------------------------
# Step 4 -- overall binary classification metrics (sec. 10)
# positive = fraud (y == 1), negative = legitimate (y == 0).
# This is INTENTIONALLY a separate concept from ALLOW/REVIEW/BLOCK
# (sec. 11 below) -- see module docstring / report "fraud_metrics" note.
# ---------------------------------------------------------------------------
def compute_classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    preds = (proba >= CLASSIFICATION_THRESHOLD).astype(int)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    accuracy = float(accuracy_score(y, preds))
    precision = float(precision_score(y, preds, zero_division=0))
    recall = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))

    # Cross-check against the repository's own canonical block_metrics()
    # (cascade_with_graph.py) at the same threshold -- must agree exactly,
    # since both are computing the identical thing from the identical
    # inputs. A mismatch here would mean this harness's metric formulas
    # have silently diverged from the canonical ones.
    canonical_overall, canonical_preds = cwg.block_metrics(y, proba, threshold=CLASSIFICATION_THRESHOLD)
    cross_check_ok = (
        np.array_equal(canonical_preds, preds)
        and abs(canonical_overall.get("precision", precision) - precision) < 1e-9
        and abs(canonical_overall.get("recall", recall) - recall) < 1e-9
        and abs(canonical_overall.get("f1", f1) - f1) < 1e-9
    )
    if not cross_check_ok:
        raise EvaluationHarnessError(
            "Classification metrics computed by this harness do not match "
            "cascade_with_graph.block_metrics() on the same y/proba/threshold -- "
            "refusing to report possibly-inconsistent metrics."
        )

    return {
        "positive_class_definition": "positive = fraud (y == 1), negative = legitimate (y == 0)",
        "threshold_used": CLASSIFICATION_THRESHOLD,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "canonical_cross_check": "cascade_with_graph.block_metrics",
        "canonical_cross_check_passed": cross_check_ok,
    }


def compute_fraud_metrics(classification: dict[str, Any]) -> dict[str, Any]:
    cm = classification["confusion_matrix"]
    tp, tn, fp, fn = cm["true_positive"], cm["true_negative"], cm["false_positive"], cm["false_negative"]
    fraud_count = tp + fn
    legit_count = tn + fp
    return {
        "fraud_count": fraud_count,
        "legitimate_count": legit_count,
        "fraud_recall": round(tp / fraud_count, 4) if fraud_count else None,
        "fraud_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "fraud_false_negative_rate": round(fn / fraud_count, 4) if fraud_count else None,
        "fraud_false_positive_rate": round(fp / legit_count, 4) if legit_count else None,
    }


# ---------------------------------------------------------------------------
# Step 5 -- three-way decision-policy metrics (sec. 11/12), reusing
# decision_policy.py's canonical policy_stats()/expected_cost() verbatim.
# ---------------------------------------------------------------------------
def compute_decision_policy_metrics(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray, attack_family: np.ndarray,
    t_review: float, t_block: float, cost_model: dp.CostModel,
) -> dict[str, Any]:
    stats = dp.policy_stats(y, proba, dollars, t_review, t_block, cost_model, attack_family=attack_family)

    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y.astype(int) == 1
    legit = ~fraud

    n = len(y)
    counts = {
        "allow_count": int(is_allow.sum()),
        "review_count": int(is_review.sum()),
        "block_count": int(is_block.sum()),
    }
    if counts["allow_count"] + counts["review_count"] + counts["block_count"] != n:
        raise EvaluationHarnessError("ALLOW + REVIEW + BLOCK does not equal the evaluation row count.")

    composition = {
        "fraud_allowed": int((is_allow & fraud).sum()),
        "fraud_reviewed": int((is_review & fraud).sum()),
        "fraud_blocked": int((is_block & fraud).sum()),
        "legitimate_allowed": int((is_allow & legit).sum()),
        "legitimate_reviewed": int((is_review & legit).sum()),
        "legitimate_blocked": int((is_block & legit).sum()),
    }
    fraud_n = max(int(fraud.sum()), 1)
    legit_n = max(int(legit.sum()), 1)
    rates = {
        "allow_rate": round(counts["allow_count"] / n, 4),
        "review_rate": round(counts["review_count"] / n, 4),
        "block_rate": round(counts["block_count"] / n, 4),
        "fraud_allowed_rate": round(composition["fraud_allowed"] / fraud_n, 4),
        "fraud_review_rate": round(composition["fraud_reviewed"] / fraud_n, 4),
        "fraud_block_rate": round(composition["fraud_blocked"] / fraud_n, 4),
        "legitimate_allowed_rate": round(composition["legitimate_allowed"] / legit_n, 4),
        "legitimate_review_rate": round(composition["legitimate_reviewed"] / legit_n, 4),
        "legitimate_block_rate": round(composition["legitimate_blocked"] / legit_n, 4),
    }

    return {
        "t_review": t_review,
        "t_block": t_block,
        **counts,
        **rates,
        "composition": composition,
        "canonical_policy_stats": stats,  # full decision_policy.policy_stats() output, unmodified
    }


def compute_cost_metrics(decision_policy_metrics: dict[str, Any], cost_model: dp.CostModel) -> dict[str, Any]:
    stats = decision_policy_metrics["canonical_policy_stats"]
    comp = decision_policy_metrics["composition"]
    return {
        "expected_cost": stats["expected_cost_at_assumed_prevalence"],
        "expected_cost_definition": (
            "decision_policy.expected_cost(), importance-reweighted to "
            "assumed_production_fraud_rate -- the canonical objective the "
            "'corrected' threshold pair was selected to minimize."
        ),
        "fraud_loss_dollars_allowed_through_unweighted": stats["dollars_fraud_allowed_through"],
        "fraud_loss_dollars_total_unweighted": stats["dollars_fraud_total"],
        "review_ops_cost_unweighted": round(
            decision_policy_metrics["review_count"] * cost_model.review_ops_cost, 2
        ),
        "false_positive_cost_unweighted": round(
            comp["legitimate_blocked"] * cost_model.legit_block_friction_cost, 2
        ),
        "note": (
            "The *_unweighted figures above are raw dollar/count totals observed "
            "on this validation population, reported for context. They are NOT "
            "expected to sum exactly to expected_cost, which is importance-"
            "reweighted (dp.sample_weights) to assumed_production_fraud_rate "
            "before summing -- see decision_policy.py's module docstring for why "
            "the unweighted validation-set prevalence cannot be used directly."
        ),
        "assumed_production_fraud_rate": cost_model.assumed_production_fraud_rate,
        "cost_model": asdict(cost_model),
    }


# ---------------------------------------------------------------------------
# Step 6 -- per-attack-family evaluation (sec. 13)
# ---------------------------------------------------------------------------
def compute_attack_family_metrics(
    df: pd.DataFrame, y: np.ndarray, proba: np.ndarray, t_review: float, t_block: float,
) -> dict[str, Any]:
    attack_family = df["attack_family"].values
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y.astype(int) == 1

    result = {}
    for fam in SUPPORTED_ATTACK_FAMILIES:
        fam_mask = attack_family == fam
        count = int(fam_mask.sum())
        fam_fraud_mask = fam_mask & fraud
        fraud_count = int(fam_fraud_mask.sum())
        fraud_blocked = int((fam_fraud_mask & is_block).sum())
        fraud_reviewed = int((fam_fraud_mask & is_review).sum())
        fraud_allowed = int((fam_fraud_mask & is_allow).sum())
        result[fam] = {
            "count": count,
            "fraud_count": fraud_count,
            "allow_count": int((fam_mask & is_allow).sum()),
            "review_count": int((fam_mask & is_review).sum()),
            "block_count": int((fam_mask & is_block).sum()),
            "fraud_recall": round((fraud_blocked + fraud_reviewed) / fraud_count, 4) if fraud_count else None,
            "fraud_recall_blocked_only": round(fraud_blocked / fraud_count, 4) if fraud_count else None,
            "fraud_allowed": fraud_allowed,
            "fraud_reviewed": fraud_reviewed,
            "fraud_blocked": fraud_blocked,
        }

    total_from_families = sum(v["count"] for v in result.values())
    total_fraud_from_families = sum(v["fraud_count"] for v in result.values())
    reconciliation = {
        "sum_family_counts": total_from_families,
        "sum_family_fraud_counts": total_fraud_from_families,
        "matches_global_fraud_count": total_fraud_from_families == int(fraud.sum()),
    }
    return {"families": result, "reconciliation": reconciliation}


# ---------------------------------------------------------------------------
# Step 7 -- stage availability (sec. 15). Only the fused (Stage 5) score
# is available from the canonical read-only cache; the harness does NOT
# recompute per-stage scores (that would mean retraining the per-fold
# GCN / Stage 1+2 models), so stages 1_2/3/4 are honestly reported
# unavailable rather than backfilled with 0 or a re-derived value.
# ---------------------------------------------------------------------------
def compute_stage_availability(n_rows: int) -> dict[str, Any]:
    unavailable_reason = (
        "decision_policy_validation_cache.npz (validation_variant='fused') "
        "persists only the final fused Stage 5 score, not the individual "
        "Stage 1+2 / Stage 3 (GCN) / Stage 4 (autoencoder) component scores. "
        "Recomputing them would require re-running the 5-fold OOF cascade "
        "(a fresh GCN retrained per fold, per risk_fusion.compute_base_scores), "
        "which this read-only evaluation harness does not do. See "
        "risk_fusion_results.json's 'comparison' block for a historical, "
        "already-computed per-stage breakdown from the run that originally "
        "produced this policy (different corpus variant -- see that file's "
        "own docstring caveat)."
    )
    return {
        "stage1_2": {"available": False, "count_available": 0, "count_unavailable": n_rows, "reason": unavailable_reason},
        "stage3_graph": {"available": False, "count_available": 0, "count_unavailable": n_rows, "reason": unavailable_reason},
        "stage4_autoencoder": {"available": False, "count_available": 0, "count_unavailable": n_rows, "reason": unavailable_reason},
        "stage5_fusion": {
            "available": True,
            "count_available": n_rows,
            "count_unavailable": 0,
            "source": "decision_policy_validation_cache.npz (validation_variant='fused')",
        },
    }


# ---------------------------------------------------------------------------
# Step 8 -- calibration (sec. 7). Read-only reliability diagnostic:
# bins already-produced scores against already-produced labels. This
# FITS NOTHING (no calibrator, no regression, no parameter estimation) --
# it is purely descriptive binning + a Brier-score computation, both of
# which only ever read already-computed values.
# ---------------------------------------------------------------------------
def compute_calibration_diagnostics(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    brier_score = float(np.mean((proba - y) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba, bin_edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bins.append({
            "bin_range": [round(float(bin_edges[b]), 3), round(float(bin_edges[b + 1]), 3)],
            "count": count,
            "mean_predicted_score": round(float(proba[mask].mean()), 4) if count else None,
            "observed_fraud_rate": round(float(y[mask].mean()), 4) if count else None,
        })

    return {
        "available": True,
        "method": (
            "Read-only reliability diagnostic over the already-produced fused "
            "score and already-known labels -- no calibrator is fit, no "
            "parameters are estimated or applied."
        ),
        "brier_score": round(brier_score, 6),
        "n_bins": n_bins,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# Step 9 -- provenance (sec. 17)
# ---------------------------------------------------------------------------
def compute_provenance(policy: dict[str, Any]) -> dict[str, Any]:
    cache_path = dp.CACHE_PATH
    prov: dict[str, Any] = {
        "git_commit": am._git_commit(REPO_ROOT),
        "git_dirty": am._git_dirty(REPO_ROOT),
        "package_versions": am._package_versions(),
        "artifact_paths": {
            "decision_policy_results.json": str(POLICY_RESULTS_PATH),
            "decision_policy_validation_cache.npz": str(cache_path),
        },
        "policy_source": "decision_policy_results.json['corrected']",
        "score_source": policy.get("score_source"),
        "policy_artifact_provenance": policy.get("policy_provenance"),
    }
    if not cache_path.exists():
        prov["cache_provenance"] = None
    else:
        prov["cache_provenance"] = {
            "path": str(cache_path),
            "content_hash_sha256": am.hash_file(cache_path),
        }
    return prov


# ---------------------------------------------------------------------------
# Step 10 -- top-level invariant checks (sec. 20), collected in one place
# so the report's "invariants" section is a single source of truth for
# what was actually verified (rather than scattered raises the caller
# has to infer happened).
# ---------------------------------------------------------------------------
def compute_invariants(
    dataset_info: dict[str, Any],
    score_info: dict[str, Any],
    classification: dict[str, Any],
    decision_policy_metrics: dict[str, Any],
    attack_family_metrics: dict[str, Any],
    policy: dict[str, Any],
    consistency: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "row_count_equals_unique_trace_ids": dataset_info["row_count"] == dataset_info["unique_trace_ids"],
        "score_count_equals_row_count": score_info["available_count"] == dataset_info["row_count"],
        "scores_finite_and_in_unit_interval": True,  # validate_scores() already raised if false
        "attack_families_all_supported": True,       # validate_dataset_invariants() already raised if false
        "allow_review_block_sums_to_row_count": (
            decision_policy_metrics["allow_count"]
            + decision_policy_metrics["review_count"]
            + decision_policy_metrics["block_count"]
            == dataset_info["row_count"]
        ),
        "thresholds_valid": 0.0 <= policy["t_review"] < policy["t_block"] <= 1.0,
        "family_fraud_counts_reconcile_with_global": attack_family_metrics["reconciliation"]["matches_global_fraud_count"],
        "classification_cross_check_passed": classification["canonical_cross_check_passed"],
        "cache_variant_is_fused": consistency["cache_variant"] == "fused",
        "policy_and_cache_score_source_consistent": consistency["policy_and_cache_consistent"],
        "all_reported_metrics_finite": _all_finite(
            [
                classification["accuracy"], classification["precision"],
                classification["recall"], classification["f1"],
                decision_policy_metrics["canonical_policy_stats"]["expected_cost_at_assumed_prevalence"],
            ]
        ),
    }
    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


def _all_finite(values: list[float]) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_evaluation() -> dict[str, Any]:
    df = build_canonical_evaluation_population()
    dataset_info = validate_dataset_invariants(df)

    y, proba, dollars = load_canonical_fused_scores(df)
    score_info = validate_scores(proba, dataset_info["row_count"])

    policy = load_canonical_policy()
    consistency = {
        "cache_variant": "fused",  # load_canonical_fused_scores() already enforced this
        "policy_and_cache_consistent": True,
    }

    classification = compute_classification_metrics(y, proba)
    fraud_metrics = compute_fraud_metrics(classification)

    attack_family_arr = df["attack_family"].values
    decision_policy_metrics = compute_decision_policy_metrics(
        y, proba, dollars, attack_family_arr, policy["t_review"], policy["t_block"], policy["cost_model"],
    )

    # Cross-check: recomputing policy_stats() on this run's freshly-loaded
    # (but cache-sourced) data should reproduce decision_policy_results.json's
    # persisted "corrected" block, since it's the exact same canonical
    # function applied to the exact same population/score/policy.
    persisted_corrected = None
    if POLICY_RESULTS_PATH.exists():
        with open(POLICY_RESULTS_PATH) as f:
            persisted_corrected = json.load(f).get("corrected")
    reproduces_persisted_corrected_block = _compare_policy_stats(
        decision_policy_metrics["canonical_policy_stats"], persisted_corrected
    )

    cost_metrics = compute_cost_metrics(decision_policy_metrics, policy["cost_model"])
    attack_family_metrics = compute_attack_family_metrics(df, y, proba, policy["t_review"], policy["t_block"])
    stage_availability = compute_stage_availability(dataset_info["row_count"])
    calibration = compute_calibration_diagnostics(y, proba)
    provenance = compute_provenance(policy)

    invariants = compute_invariants(
        dataset_info, score_info, classification, decision_policy_metrics,
        attack_family_metrics, policy, consistency,
    )
    invariants["policy_stats_reproduces_persisted_corrected_block"] = reproduces_persisted_corrected_block
    invariants["all_passed"] = invariants["all_passed"] and reproduces_persisted_corrected_block

    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": {
            "dataset": (
                "cascade_with_graph.load_all_records() (ring-overlay validation "
                "population) + decision_policy.load_cached_validation_data('fused')"
            ),
            "row_count": dataset_info["row_count"],
            "unique_trace_ids": dataset_info["unique_trace_ids"],
            "fraud_count": fraud_metrics["fraud_count"],
            "legitimate_count": fraud_metrics["legitimate_count"],
            "attack_family_counts": dataset_info["attack_family_counts"],
            "matches_documented_phase4_population": dataset_info["matches_documented_phase4_population"],
        },
        "policy": {
            "source": "decision_policy_results.json",
            "variant": "corrected",
            "review_threshold": policy["t_review"],
            "block_threshold": policy["t_block"],
            "score_source": policy["score_source"],
            "methodology_note": policy["methodology_note"],
        },
        "scores": score_info,
        "classification": classification,
        "fraud_metrics": fraud_metrics,
        "decision_policy_metrics": decision_policy_metrics,
        "attack_families": attack_family_metrics,
        "stage_availability": stage_availability,
        "cost_metrics": cost_metrics,
        "calibration": calibration,
        "provenance": provenance,
        "invariants": invariants,
        "leakage_protections": {
            "threshold_optimization": "NOT USED (decision_policy.optimize_thresholds is never called)",
            "calibration_fitting": "NOT USED (compute_calibration_diagnostics only bins/reads existing scores+labels)",
            "retraining": "NOT USED (no .fit()/.train() call anywhere in this file)",
            "preprocessing_fitting": "NOT USED (no scaler/encoder fit anywhere in this file)",
            "labels_used_only_after_scoring": True,
        },
    }
    return report


def _compare_policy_stats(recomputed: dict[str, Any], persisted: dict[str, Any] | None) -> bool:
    if persisted is None:
        return False
    keys = [
        "t_review", "t_block", "allow_rate", "review_rate", "block_rate",
        "legit_blocked", "fraud_blocked", "fraud_reviewed", "fraud_allowed",
        "fraud_recall_blocked_only", "fraud_recall_blocked_plus_review",
        "expected_cost_at_assumed_prevalence",
    ]
    for k in keys:
        a, b = recomputed.get(k), persisted.get(k)
        if a is None or b is None:
            return False
        if isinstance(a, float) or isinstance(b, float):
            if abs(float(a) - float(b)) > 1e-6:
                return False
        elif a != b:
            return False
    return True


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> int:
    print("Phase 5A: running formal evaluation harness (run #1)...")
    report_1 = run_evaluation()

    print("Phase 5A: running again to verify determinism (run #2)...")
    report_2 = run_evaluation()

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


def _compare_core_results(r1: dict[str, Any], r2: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    core_paths = [
        ("evaluation", "row_count"), ("evaluation", "fraud_count"), ("evaluation", "legitimate_count"),
        ("policy", "review_threshold"), ("policy", "block_threshold"),
        ("scores", "min"), ("scores", "max"), ("scores", "mean"), ("scores", "median"), ("scores", "std"),
        ("classification", "accuracy"), ("classification", "precision"),
        ("classification", "recall"), ("classification", "f1"),
        ("classification", "confusion_matrix"),
        ("fraud_metrics", "fraud_recall"), ("fraud_metrics", "fraud_precision"),
        ("decision_policy_metrics", "allow_count"), ("decision_policy_metrics", "review_count"),
        ("decision_policy_metrics", "block_count"),
        ("cost_metrics", "expected_cost"),
        ("attack_families",),
    ]
    diffs = {}
    for path in core_paths:
        v1, v2 = r1, r2
        for key in path:
            v1, v2 = v1[key], v2[key]
        if v1 != v2:
            diffs[".".join(path)] = {"run_1": v1, "run_2": v2}
    return len(diffs) == 0, diffs


if __name__ == "__main__":
    raise SystemExit(main())
