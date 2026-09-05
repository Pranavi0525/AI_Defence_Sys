"""
api/inference.py
==================
Real, load-once, inference-only Stage 1+2 scoring path for POST /api/score.

WHAT THIS DOES
--------------
- Loads blue_team_output/calibrator.joblib (the calibrated XGBoost saved by
  blue_team_pipeline.py's main()) and blue_team_output/xgb_model.joblib
  (the raw booster, needed for SHAP) exactly ONCE, at process startup --
  see ModelRegistry.load(). Nothing here ever calls .fit() on anything, and
  nothing here shells out to blue_team_pipeline.py, cascade_with_graph.py,
  or any other batch script. If that's what you want, use
  POST /api/pipeline/run instead (see pipeline_runner.py).
- Reuses blue_team_pipeline.extract_features() and .stage1_rule_filter()
  UNCHANGED -- the online feature contract is byte-for-byte the same
  function the training/eval code uses, imported directly, not
  reimplemented. See _load_pipeline_module() below for how that root-level
  script is imported as a module without turning it into a package.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
Stage 3 (GCN graph escalation), Stage 4 (autoencoder novelty), and Stage 5
(risk fusion) are NOT computed here, because:
  - None of the three has a saved, loadable artifact anywhere in this repo
    (verified: only xgb_model.joblib and calibrator.joblib exist under
    blue_team_output/). Each is retrained fresh inside a 5-fold CV loop
    every time its batch script runs.
  - Stage 3 specifically needs a CROSS-CUSTOMER graph
    (build_cross_customer_graph() in cascade_with_graph.py, keyed on
    shared beneficiary_id / device_id across the whole population) to mean
    anything at all. A single incoming transaction, scored in isolation,
    has no other nodes to connect to -- there is no persistent
    entity-resolution/graph store in this repo to query instead. Faking a
    graph score for an isolated request would be exactly the kind of
    fabrication this endpoint is required not to do.
See score_trace()'s returned StageUnavailable entries for how this is
surfaced honestly to API callers instead of silently omitted or faked.

DECISION RULE (Stage 1+2 only -- see module docstring for why the existing
decision_policy_results.json thresholds are NOT reused here)
-----------------------------------------------------------
  - Stage 1 auto-clears (stage1_rule_filter() -> False): decision = ALLOW,
    risk_score = 0.0, model never invoked. This exactly mirrors
    compute_stage_1_2_cascade()'s own semantics for an auto-cleared row.
  - Stage 1 escalates and calibrated score < DECISION_THRESHOLD (0.5,
    the same constant blue_team_pipeline.CONFIG already uses to report its
    own Stage-1+2-only precision/recall/F1 numbers): decision = REVIEW.
    This is intentionally more conservative than silently ALLOWing --
    Stage 1 already found the trace behaviorally anomalous, and Stage
    3/4/5 (which could otherwise clear or confirm that suspicion) simply
    aren't available online, so the safe default is a human review queue,
    not a silent pass.
  - score >= DECISION_THRESHOLD: decision = BLOCK.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import OUT_DIR, REPO_ROOT

logger = logging.getLogger("ai_defense.inference")

_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


class ModelNotReady(RuntimeError):
    """Raised by score_trace() if called before/without a successfully
    loaded ModelRegistry. The API layer turns this into HTTP 503, never a
    raw traceback."""


def _load_pipeline_module() -> types.ModuleType:
    """Import repo-root blue_team_pipeline.py as a module WITHOUT requiring
    the repo root to be a package or permanently polluting sys.path for
    every other import in the process. We need this module's
    extract_features(), stage1_rule_filter(), and FEATURE_COLS verbatim --
    reimplementing them here would be exactly the kind of "invented feature
    semantics" the online endpoint must avoid.
    """
    module_path = REPO_ROOT / "blue_team_pipeline.py"
    if not module_path.exists():
        raise ModelNotReady(f"blue_team_pipeline.py not found at {module_path}")
    spec = importlib.util.spec_from_file_location("blue_team_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class HesitationBaseline:
    """Cold-start baseline for HESITATION_DELTA, computed from the same
    training feature table blue_team_pipeline.py already wrote to disk
    (blue_team_output/feature_table.csv) -- NOT recomputed by retraining
    anything. A single online request has no per-customer transaction
    history to build a customer-specific baseline from (the service is
    stateless), so it always falls into add_hesitation_delta()'s own
    documented cold-start branch: population-wide mean/std instead of a
    per-customer one."""

    global_mean: float
    global_std: float


class ModelRegistry:
    """Holds everything score_trace() needs, loaded exactly once. A single
    process-wide instance lives on app.state (see app.py's lifespan
    handler) so every request reuses it -- no per-request disk I/O beyond
    reading the request body itself."""

    def __init__(self) -> None:
        self.ready: bool = False
        self.load_error: str | None = None
        self.calibrator = None
        self.base_model = None
        self.shap_explainer = None
        self.feature_cols: list[str] = []
        self.decision_threshold: float = 0.5
        self.hesitation_baseline: HesitationBaseline | None = None
        self.reference_thresholds: dict[str, float] | None = None
        self.model_version: str = "unloaded"
        self._pipeline_module: types.ModuleType | None = None

    def load(self) -> None:
        """Load all artifacts. Never raises on missing/broken artifacts --
        records the failure on self.load_error and leaves self.ready=False
        instead, so the process can still start, answer /healthz, and
        report /readyz=false rather than crashing outright (per requirement
        A: the app must not CLAIM readiness it doesn't have, but a failed
        model load should not take down the whole API, including the
        pipeline-dashboard endpoints that don't need this registry)."""
        try:
            calibrator_path = OUT_DIR / "calibrator.joblib"
            base_model_path = OUT_DIR / "xgb_model.joblib"
            feature_table_path = OUT_DIR / "feature_table.csv"

            missing = [p for p in (calibrator_path, base_model_path, feature_table_path) if not p.exists()]
            if missing:
                raise ModelNotReady(
                    "Missing required artifact(s): "
                    + ", ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
                    + ". Run `PYTHONPATH=src python3 blue_team_pipeline.py` from the repo root first."
                )

            pipeline = _load_pipeline_module()
            self._pipeline_module = pipeline
            self.feature_cols = list(pipeline.FEATURE_COLS)
            self.decision_threshold = float(pipeline.CONFIG["DECISION_THRESHOLD"])

            self.calibrator = joblib.load(calibrator_path)
            self.base_model = joblib.load(base_model_path)

            import shap  # deferred import: heavy, only needed once artifacts exist

            self.shap_explainer = shap.TreeExplainer(self.base_model)

            feat_df = pd.read_csv(feature_table_path)
            global_mean = float(feat_df["mean_time_between_transactions"].mean())
            global_std = float(feat_df["mean_time_between_transactions"].std())
            if not global_std or np.isnan(global_std):
                global_std = 1.0
            self.hesitation_baseline = HesitationBaseline(global_mean=global_mean, global_std=global_std)

            self.reference_thresholds = self._load_reference_thresholds()
            self.model_version = self._compute_model_version(calibrator_path, base_model_path)

            self.ready = True
            self.load_error = None
            logger.info("model_registry_loaded", extra={"model_version": self.model_version})
        except Exception as exc:  # noqa: BLE001 -- record, don't crash the process
            self.ready = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("model_registry_load_failed", extra={"error": self.load_error})

    def _load_reference_thresholds(self) -> dict[str, float] | None:
        path = REPO_ROOT / "decision_policy_results.json"
        if not path.exists():
            return None
        import json

        with open(path) as f:
            data = json.load(f)
        corrected = data.get("corrected")
        if not corrected:
            return None
        return {"t_review": float(corrected["t_review"]), "t_block": float(corrected["t_block"])}

    def _compute_model_version(self, calibrator_path: Path, base_model_path: Path) -> str:
        import json

        results_path = OUT_DIR / "results.json"
        trained_at = "unknown"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    meta = json.load(f).get("_artifact_metadata", {})
                trained_at = meta.get("generated_at_utc", "unknown")
            except Exception:  # noqa: BLE001 -- version string is best-effort
                pass
        artifact_hash = _sha256_file(calibrator_path) + _sha256_file(base_model_path)
        short_hash = hashlib.sha256(artifact_hash.encode()).hexdigest()[:12]
        return f"stage1_2-calibrated-xgb@{trained_at}+{short_hash}"


def _build_hesitation_delta(feats: dict, baseline: HesitationBaseline) -> float:
    """Online counterpart of blue_team_pipeline.add_hesitation_delta(),
    restricted to the cold-start branch that function already defines --
    see HesitationBaseline's docstring for why cold-start is the only
    branch a stateless single-request service can ever take."""
    if feats.get("beneficiary_added_before_transaction") != 1:
        return 0.0
    raw = feats["time_from_beneficiary_add_to_transaction"]
    return float((raw - baseline.global_mean) / baseline.global_std)


def score_trace(trace_dict: dict, registry: ModelRegistry) -> dict[str, Any]:
    """trace_dict: the validated ObservableAttackTrace, already
    .model_dump(mode="json")'d by the caller (see app.py). Returns a plain
    dict matching schemas.ScoreResponse's fields (minus request_id/
    inference_latency_ms, which the caller fills in)."""
    if not registry.ready:
        raise ModelNotReady(registry.load_error or "Model artifacts are not loaded.")

    t0 = time.perf_counter()
    pipeline = registry._pipeline_module
    assert pipeline is not None

    feats = pipeline.extract_features(trace_dict)
    feats["HESITATION_DELTA"] = _build_hesitation_delta(feats, registry.hesitation_baseline)

    escalate = bool(pipeline.stage1_rule_filter(feats))

    top_features: list[dict[str, Any]] = []
    if not escalate:
        risk_score = 0.0
        stage1_2_detail = "Stage 1 rule filter auto-cleared this trace; Stage 2 model was not invoked (mirrors compute_stage_1_2_cascade()'s own auto-clear semantics)."
        ran_model = False
    else:
        X = np.array([[feats[c] for c in registry.feature_cols]], dtype=float)
        risk_score = float(registry.calibrator.predict_proba(X)[0, 1])
        ran_model = True
        stage1_2_detail = "Stage 1 escalated this trace; score is the calibrated Stage 2 XGBoost probability."

        shap_values = registry.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):  # older shap API: list per class
            shap_values = shap_values[1]
        row_shap = shap_values[0]
        order = np.argsort(-np.abs(row_shap))[:5]
        top_features = [
            {
                "feature": registry.feature_cols[i],
                "value": feats[registry.feature_cols[i]],
                "shap_contribution": float(row_shap[i]),
            }
            for i in order
        ]

    if not escalate:
        decision = "ALLOW"
        decision_basis = "stage1_auto_clear"
    elif risk_score < registry.decision_threshold:
        decision = "REVIEW"
        decision_basis = f"stage1_2_score < DECISION_THRESHOLD ({registry.decision_threshold}); Stage 1 flagged it but Stage 2 score didn't cross the fraud cutoff -- routed to review since Stage 3/4/5 aren't available online to resolve the ambiguity further."
    else:
        decision = "BLOCK"
        decision_basis = f"stage1_2_score >= DECISION_THRESHOLD ({registry.decision_threshold})"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "decision": decision,
        "risk_score": risk_score,
        "decision_basis": decision_basis,
        "stage1_2": {
            "available": True,
            "score": risk_score,
            "ran_model": ran_model,
            "detail": stage1_2_detail,
        },
        "stage3_graph": {
            "available": False,
            "reason": (
                "Stage 3 requires a cross-customer graph built from shared beneficiary_id/"
                "device_id across the whole traffic population (see "
                "cascade_with_graph.build_cross_customer_graph()). A single incoming "
                "transaction has no other entities to connect to online, and this repo has "
                "no persistent graph/entity-resolution store to query instead."
            ),
        },
        "stage4_autoencoder": {
            "available": False,
            "reason": (
                "Stage 4's autoencoder has no saved artifact -- cascade_with_autoencoder.py "
                "retrains a fresh autoencoder inside each cross-validation fold every time "
                "it runs, rather than persisting reusable weights."
            ),
        },
        "stage5_fusion": {
            "available": False,
            "reason": (
                "Stage 5's fusion logistic regression combines stage_1_2/gcn/ae scores and "
                "has no saved artifact either (risk_fusion.py refits it per fold); it also "
                "structurally depends on Stage 3 and Stage 4, both unavailable above."
            ),
        },
        "top_contributing_features": top_features,
        "reference_full_pipeline_thresholds": registry.reference_thresholds,
        "model_version": registry.model_version,
        "inference_latency_ms": round(latency_ms, 3),
    }
