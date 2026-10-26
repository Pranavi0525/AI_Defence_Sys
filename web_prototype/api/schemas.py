"""
api/schemas.py
================
Request/response contracts for POST /api/score.

The REQUEST schema is not reinvented here -- it IS
`red_team.schemas.observable.ObservableAttackTrace`, the same contract the
Red Team corpus generator already produces and blue_team_pipeline.py already
consumes (see reports/ato_corpus_raw.json's "observable_trace" field, and
inference.py's score_trace()). Importing it directly means the online path
and the batch/training path can never silently drift onto two different
ideas of what a "trace" looks like.

Only the RESPONSE shape is new -- there's no existing precedent for it
because nothing in this repo previously scored a single transaction online.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from red_team.schemas.observable import ObservableAttackTrace  # noqa: E402

# Re-exported so callers only need to import from api.schemas.
__all__ = ["ObservableAttackTrace", "StageScore", "StageUnavailable", "ScoreResponse"]


class StageScore(BaseModel):
    """A stage that genuinely ran online, with a real number behind it."""

    available: bool = True
    score: float | None = Field(
        None, description="Calibrated fraud probability in [0, 1], or null if the stage auto-cleared the trace without running."
    )
    ran_model: bool = Field(
        ..., description="Whether the trained model was actually invoked (false when Stage 1 auto-cleared this trace)."
    )
    detail: str


class StageUnavailable(BaseModel):
    """A stage that is honestly reported as NOT computable online, with why."""

    available: bool = False
    reason: str


class ScoreResponse(BaseModel):
    trace_id: str
    request_id: str
    decision: str = Field(..., description="ALLOW | REVIEW | BLOCK")
    risk_score: float = Field(..., description="The Stage 1+2 calibrated probability this decision was based on. 0.0 if Stage 1 auto-cleared the trace.")
    decision_basis: str = Field(..., description="Which score/threshold rule actually produced `decision`.")

    stage1_2: StageScore
    stage3_graph: StageUnavailable
    stage4_autoencoder: StageUnavailable
    stage5_fusion: StageUnavailable

    top_contributing_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description="SHAP-based feature attributions for the Stage 2 score, empty if Stage 1 auto-cleared (model never ran).",
    )

    reference_full_pipeline_thresholds: dict[str, float] | None = Field(
        None,
        description=(
            "The (t_review, t_block) thresholds decision_policy.py optimized against the "
            "full 5-stage FUSED score, shown for context only -- NOT applied to risk_score "
            "above, since risk_score is a weaker Stage-1+2-only signal and applying "
            "fused-score thresholds to it would misrepresent precision/recall."
        ),
    )

    model_version: str
    inference_latency_ms: float
