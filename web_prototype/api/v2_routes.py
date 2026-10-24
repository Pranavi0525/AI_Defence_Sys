"""
api/v2_routes.py -- /api/v2/* : the FULL v2 cascade online (XGBoost + Isolation Forest + autoencoder + stacker + cost-based policy,
history-only graph features, drift monitoring). The v1 endpoint POST /api/score is untouched.

Runtime needs only numpy, scipy and xgboost for scoring (no scikit-learn/pandas): artifacts are portable files in models/v2/
(see v2/serving/bundle.py). Set V2_MODEL_DIR to override the location.

Safety: `commit=true` writes the trace into the live entity store (which changes later graph features). It is DISABLED unless
V2_ALLOW_COMMIT=true, so a public demo cannot be used to poison the store.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.dossier import LIABILITY_V2, build_dossier, render_markdown  # noqa: E402
from v2.serving.scorer import V2Scorer  # noqa: E402

logger = logging.getLogger("ai_defense.api.v2")
ALLOWED_EVENTS = {"SESSION_LOGIN", "TRANSACTION", "BENEFICIARY_ADDITION", "DEVICE_REGISTRATION", "ACCOUNT_OPENING",
                  "KYC_VERIFICATION", "SESSION_LOGOUT", "ACCOUNT_CONTEXT_CHANGE"}
MAX_EVENTS = 200


class TraceIn(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=120)
    customer_id: str = Field(..., min_length=1, max_length=120)
    events: list[dict[str, Any]] = Field(..., min_length=1, max_length=MAX_EVENTS)
    observation_window: list[str] | None = None

    @field_validator("events")
    @classmethod
    def _check_events(cls, events):
        for i, e in enumerate(events):
            if e.get("event_type") not in ALLOWED_EVENTS:
                raise ValueError(f"events[{i}].event_type must be one of {sorted(ALLOWED_EVENTS)}")
            try:
                datetime.fromisoformat(str(e.get("timestamp")))
            except Exception as exc:
                raise ValueError(f"events[{i}].timestamp must be an ISO-8601 string") from exc
            if e["event_type"] == "TRANSACTION":
                try:
                    if float(e["amount"]) <= 0:
                        raise ValueError
                except Exception as exc:
                    raise ValueError(f"events[{i}].amount must be a positive number") from exc
        return events


class V2ScoreRequest(BaseModel):
    trace: TraceIn
    commit: bool = Field(False, description="Add this trace to the live entity store after scoring (requires V2_ALLOW_COMMIT=true).")


class V2DossierRequest(BaseModel):
    trace: TraceIn
    family: str | None = Field(None, description="Fraud family, if an analyst has triaged the case. Unknown at scoring time, so optional.")
    include_allow: bool = Field(False, description="Also build a dossier for an ALLOW decision (e.g. to study near-misses).")

    @field_validator("family")
    @classmethod
    def _check_family(cls, v):
        if v is not None and v not in LIABILITY_V2:
            raise ValueError(f"family must be one of {sorted(LIABILITY_V2)}")
        return v


class V2Registry:
    def __init__(self):
        self.scorer: V2Scorer | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        directory = Path(os.environ.get("V2_MODEL_DIR", _ROOT / "models" / "v2"))
        try:
            self.scorer = V2Scorer(directory)
            self.load_error = None
        except Exception as exc:          # never leak paths/tracebacks to clients
            self.scorer, self.load_error = None, f"v2 model artifacts could not be loaded ({type(exc).__name__})"
            logger.warning("v2_model_load_failed", extra={"error": type(exc).__name__})

    @property
    def ready(self) -> bool:
        return self.scorer is not None


registry = V2Registry()
router = APIRouter(prefix="/api/v2", tags=["v2 full cascade"])


def _require_ready() -> V2Scorer:
    if not registry.ready:
        raise HTTPException(503, registry.load_error or "v2 model is not loaded. Check /api/v2/status.")
    return registry.scorer


@router.get("/status")
def v2_status():
    body = {"ready": registry.ready, "commit_enabled": os.environ.get("V2_ALLOW_COMMIT", "false").lower() == "true"}
    if registry.ready:
        body.update({"model_version": registry.scorer.model_version, "entity_store": registry.scorer.store.stats()})
    else:
        body["error"] = registry.load_error
    return body


@router.get("/model-card")
def v2_model_card():
    return _require_ready().model_card()


@router.post("/score")
def v2_score(req: V2ScoreRequest, request: Request):
    scorer = _require_ready()
    if req.commit and os.environ.get("V2_ALLOW_COMMIT", "false").lower() != "true":
        raise HTTPException(403, "commit=true is disabled on this deployment (set V2_ALLOW_COMMIT=true to enable).")
    rec = req.trace.model_dump(mode="json")
    try:
        with registry.lock:                     # store reads/writes + drift buffer are not thread-safe
            result = scorer.score(rec, commit=req.commit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info("v2_score_computed", extra={"request_id": request_id, "trace_id": req.trace.trace_id, "decision": result["decision"],
                                            "risk_score": round(result["stacked_probability"], 4), "model_version": result["model_version"]})
    return {"trace_id": req.trace.trace_id, "request_id": request_id, "decision": result["decision"],
            "risk_score": result["stacked_probability"],
            "stages": {"supervised_xgboost": {"probability": result["xgb_probability"]},
                       "novelty_isolation_forest": {"percentile_vs_normal": result["iforest_percentile"], "raw": result["iforest_raw"]},
                       "novelty_autoencoder": {"percentile_vs_normal": result["autoencoder_percentile"], "raw": result["autoencoder_raw"]},
                       "graph_history_only": result["graph_features"],
                       "fusion_stacked": {"probability": result["stacked_probability"]}},
            "thresholds": result["thresholds"], "top_contributing_features": result["top_features"],
            "novelty_note": result["novelty_note"], "committed": result["committed"],
            "model_version": result["model_version"], "inference_latency_ms": result["latency_ms"],
            "disclaimer": "Synthetic-data prototype; the score is not a real-world fraud probability."}


@router.get("/drift")
def v2_drift():
    with registry.lock:
        return _require_ready().drift.report()


@router.post("/dossier")
def v2_dossier(req: V2DossierRequest):
    """Analyst dossier for a REVIEW/BLOCK decision (model drivers, readable evidence, graph evidence, assumed acting party).
    Read-only: never writes to the entity store. `dossier` is null for ALLOW unless include_allow=true."""
    scorer = _require_ready()
    rec = req.trace.model_dump(mode="json")
    try:
        with registry.lock:
            d = build_dossier(scorer, rec, family=req.family, include_allow=req.include_allow)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"trace_id": req.trace.trace_id, "dossier": d, "markdown": render_markdown(d) if d else None,
            "note": None if d else "Decision was ALLOW, so no dossier was built (set include_allow=true to force one)."}
