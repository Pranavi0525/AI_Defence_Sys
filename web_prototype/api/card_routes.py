"""
api/card_routes.py -- /api/card/* : the CREDIT CARD fraud detection module (independent of the digital-payment module).

Own model (models/card/, override with CARD_MODEL_DIR), own registry, own thresholds, own router. It imports only v2.card.*; it
never touches the payment model registries (inference.py, v2_routes.py). A missing or broken card bundle makes only /api/card/*
answer 503; payment endpoints are unaffected.

The model is trained on SYNTHETIC simulator data. `model_score` is NOT a real-world fraud probability -- every response says so.

Single and batch prediction share one pipeline (v2/card/serving.py): /predict is /predict-batch of (history + transaction).
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.card.serving import CHANNELS, MAX_ROWS, CardInputError, CardModelError, CardScorer  # noqa: E402

logger = logging.getLogger("ai_defense.api.card")
MAX_CSV_BYTES = 10_000_000
DISCLAIMER = ("SYNTHETIC simulator-trained model. model_score is a ranking score, not a real-world fraud probability; "
              "do not use for real authorisation decisions.")


class CardTxn(BaseModel):
    """One card authorisation. Field meanings: see v2/card/serving.py."""
    model_config = ConfigDict(extra="forbid")
    card: int | str
    ts: float = Field(..., ge=0, allow_inf_nan=False, description="event time, seconds (hour-of-day features use ts mod 86400, UTC)")
    amount: float = Field(..., gt=0, allow_inf_nan=False, description="INR, > 0 and <= the configured business limit (CARD_MAX_TXN_AMOUNT); see /api/card/model-card")
    mcc: int | str
    channel: str = Field(..., description=f"one of {list(CHANNELS)}")
    lat: float = Field(..., ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(..., ge=-180, le=180, allow_inf_nan=False)
    merchant: str = Field(..., min_length=1)
    declined: int | None = Field(0, description="0/1; this card's authorisation outcome. Prior declines are features; the row's own is not.")


class CardPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: CardTxn
    history: list[CardTxn] = Field(default_factory=list, max_length=MAX_ROWS,
                                   description="Earlier transactions of the SAME card (optional but strongly recommended: velocity/novelty features need them).")


class CardBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[CardTxn] | None = Field(None, max_length=MAX_ROWS)
    csv_text: str | None = Field(None, max_length=MAX_CSV_BYTES, description="CSV with a header row: card,ts,amount,mcc,channel,lat,lon,merchant[,declined]")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.transactions is None) == (self.csv_text is None):
            raise ValueError("supply exactly one of `transactions` or `csv_text`")
        return self


class CardRegistry:
    def __init__(self):
        self.scorer: CardScorer | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        directory = Path(os.environ.get("CARD_MODEL_DIR", _ROOT / "models" / "card"))
        try:
            self.scorer = CardScorer(directory)
            self.load_error = None
        except CardModelError as exc:               # message is client-safe (no paths / tracebacks)
            self.scorer, self.load_error = None, str(exc)
            logger.warning("card_model_load_failed", extra={"error": "CardModelError"})
        except Exception as exc:                    # never let a card failure escape into the payment app's startup
            self.scorer, self.load_error = None, f"card model could not be loaded ({type(exc).__name__})"
            logger.warning("card_model_load_failed", extra={"error": type(exc).__name__})

    @property
    def ready(self) -> bool:
        return self.scorer is not None


registry = CardRegistry()
router = APIRouter(prefix="/api/card", tags=["credit card fraud (synthetic-trained)"])


def _require_ready() -> CardScorer:
    if not registry.ready:
        raise HTTPException(503, registry.load_error or "card model is not loaded. Check /api/card/status.")
    return registry.scorer


def _invalid(exc: CardInputError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"error": "invalid_card_input", "detail": exc.errors, "total_problems": exc.total})


def _dump(t: CardTxn) -> dict[str, Any]:
    return t.model_dump()


def _parse_csv(text: str) -> list[dict]:
    import pandas as pd
    if len(text.encode("utf-8", "ignore")) > MAX_CSV_BYTES:
        raise CardInputError([{"row": None, "field": None, "problem": "csv_text is too large"}])
    try:
        df = pd.read_csv(io.StringIO(text), dtype=object, keep_default_na=True)
    except Exception as exc:
        raise CardInputError([{"row": None, "field": None, "problem": f"CSV could not be parsed ({type(exc).__name__})"}]) from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    dup_cols = sorted({c for c in df.columns if list(df.columns).count(c) > 1 or c.rsplit(".", 1)[0] in df.columns and c.rsplit(".", 1)[-1].isdigit() and c != c.rsplit(".", 1)[0]})
    if dup_cols:
        raise CardInputError([{"row": None, "field": c.rsplit(".", 1)[0], "problem": "column appears more than once in the CSV header"} for c in dup_cols])
    recs = []
    for r in df.to_dict("records"):
        out = {}
        for k, v in r.items():
            if isinstance(v, str):
                v = v.strip()
                if k in ("ts", "amount", "lat", "lon", "declined"):
                    try:
                        v = float(v)
                    except ValueError:
                        pass            # left as text -> the validator reports it by row and field
                elif k in ("card", "mcc") and v.lstrip("-").isdigit():
                    v = int(v)
            elif v != v:                # NaN from an empty cell -> missing
                v = None
            out[k] = v
        recs.append(out)
    return recs


@router.get("/status")
def card_status():
    body = {"ready": registry.ready, "synthetic_trained": True, "disclaimer": DISCLAIMER}
    if registry.ready:
        s = registry.scorer
        from v2.card.limits import CURRENCY
        body.update({
            "model_version": s.model_version,
            "threshold": s.threshold,
            "feature_count": len(s.features),
            "currency": CURRENCY,
            "max_transaction_amount": s.amount_limit,
            "configured_serving_input_limit": s.amount_limit,
            "training_amount_support": s.amount_support,
            "serving_limit_exceeds_training_support": s.serving_limit_exceeds_training_support,
        })
    else:
        body["error"] = registry.load_error
    return body


@router.get("/model-card")
def card_model_card():
    return _require_ready().model_card()


@router.post("/predict")
def card_predict(req: CardPredictRequest, request: Request):
    scorer = _require_ready()
    try:
        with registry.lock:
            res = scorer.score_one(_dump(req.transaction), [_dump(h) for h in req.history])
            latency = scorer._last_latency_ms
    except CardInputError as exc:
        return _invalid(exc)
    except CardModelError as exc:
        raise HTTPException(503, str(exc)) from exc
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info("card_predict_computed", extra={"request_id": request_id, "classification": res["classification"],
                                                "model_version": scorer.model_version})
    res.pop("index", None)
    return {"request_id": request_id, **res, "threshold": scorer.threshold, "model_version": scorer.model_version,
            "inference_latency_ms": round(latency, 3), "disclaimer": DISCLAIMER}


@router.post("/predict-batch")
def card_predict_batch(req: CardBatchRequest, request: Request):
    scorer = _require_ready()
    try:
        rows = [_dump(t) for t in req.transactions] if req.transactions is not None else _parse_csv(req.csv_text)
        with registry.lock:
            results = scorer.score_batch(rows)
            latency = scorer._last_latency_ms
    except CardInputError as exc:
        return _invalid(exc)
    except CardModelError as exc:
        raise HTTPException(503, str(exc)) from exc
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    summary = CardScorer.summarize(results)
    logger.info("card_predict_batch_computed", extra={"request_id": request_id, "transactions": summary["transactions"],
                                                      "flagged_fraud": summary["flagged_fraud"], "model_version": scorer.model_version})
    return {"request_id": request_id, "summary": summary, "results": results, "threshold": scorer.threshold,
            "model_version": scorer.model_version, "inference_latency_ms": round(latency, 3), "disclaimer": DISCLAIMER}
