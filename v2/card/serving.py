"""v2/card/serving.py -- online + batch scoring with the served card-fraud model.

ONE pipeline for everything:  validate -> v2.card.features.build_features (the 27 causal features, unchanged) -> booster ->
threshold.  `CardScorer.score_batch(rows)` is that pipeline; `CardScorer.score_one(transaction, history)` is literally
score_batch(history + [transaction]) returning the last row, so single and batch results cannot diverge.

Independence: this module imports only from v2.card.* (never from the payment cascade: v2.serving, v2.fusion, red_team, ...).
Runtime needs numpy, pandas and xgboost (no scikit-learn).

Failure policy: a missing / corrupted / mismatching artifact raises CardModelError. There is NO fallback model.
Bad input raises CardInputError carrying per-row problems. Nothing is silently imputed or dropped.

Input schema (the columns v2.card.features.build_features reads):
    card      id of the card (int or non-empty string)
    ts        event time in seconds (hour-of-day features use ts mod 86400, i.e. UTC)
    amount    > 0
    mcc       merchant category code (int or non-empty string)
    channel   "POS" | "ECOM" | "ATM"
    lat, lon  location of the transaction (for ECOM: the merchant's location, as in the simulator)
    merchant  non-empty string
    declined  0/1, whether this card's authorisation was declined. OPTIONAL, defaults to 0. NOTE: the CURRENT row's own flag is
              never used as a feature; only PRIOR declines on the same card are (see v2/card/features.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, build_features

REQUIRED = ("card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant")
OPTIONAL = ("declined",)
CHANNELS = ("POS", "ECOM", "ATM")
MAX_ROWS = 50_000
BUNDLE_FORMAT = "card-bundle-1"
MAX_REPORTED_ERRORS = 25


class CardModelError(RuntimeError):
    """The card model artifacts are missing, corrupted, or inconsistent with the feature code."""


class CardInputError(ValueError):
    """The input transactions failed validation. `.errors` is a list of {row, field, problem} dicts."""

    def __init__(self, errors: list[dict], total: int | None = None):
        self.errors = errors[:MAX_REPORTED_ERRORS]
        self.total = total if total is not None else len(errors)
        super().__init__(f"{self.total} input problem(s); first: {errors[0] if errors else None}")


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x))


def _is_id(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    return isinstance(x, str) and x.strip() != ""


def validate_rows(rows) -> pd.DataFrame:
    """Strictly validate a list of dicts (or a DataFrame) and return a clean frame in the simulator's column layout, with a `row_id`
    column holding each row's original position. Raises CardInputError listing every problem found (up to a cap)."""
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
        cols = set(rows.columns)
    else:
        records = list(rows)
        cols = set().union(*[r.keys() for r in records if isinstance(r, dict)]) if records else set()
    if not records:
        raise CardInputError([{"row": None, "field": None, "problem": "no transactions supplied"}])
    if len(records) > MAX_ROWS:
        raise CardInputError([{"row": None, "field": None, "problem": f"too many transactions ({len(records)} > {MAX_ROWS})"}])
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise CardInputError([{"row": None, "field": c, "problem": "required column is missing"} for c in missing])

    errs: list[dict] = []
    out = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            errs.append({"row": i, "field": None, "problem": "row is not an object"}); continue
        # NaN / None in a required field is "missing", not a value to impute
        def bad(field, problem):
            errs.append({"row": i, "field": field, "problem": problem})
        row = {}
        for k in ("card", "mcc", "merchant"):
            v = r.get(k)
            if isinstance(v, float) and v.is_integer() and k != "merchant":
                v = int(v)
            if not _is_id(v):
                bad(k, "missing or not an id (int or non-empty string)")
            row[k] = v.strip() if isinstance(v, str) else v
        ts = r.get("ts"); amt = r.get("amount"); la = r.get("lat"); lo = r.get("lon")
        if not _is_num(ts) or float(ts) < 0:
            bad("ts", "must be a finite number >= 0 (seconds)")
        if not _is_num(amt) or float(amt) <= 0:
            bad("amount", "must be a finite number > 0")
        if not _is_num(la) or not -90 <= float(la) <= 90:
            bad("lat", "must be a finite number in [-90, 90]")
        if not _is_num(lo) or not -180 <= float(lo) <= 180:
            bad("lon", "must be a finite number in [-180, 180]")
        ch = r.get("channel")
        ch = ch.strip().upper() if isinstance(ch, str) else ch
        if ch not in CHANNELS:
            bad("channel", f"must be one of {list(CHANNELS)}")
        dec = r.get("declined", 0)
        if dec is None or (isinstance(dec, float) and math.isnan(dec)):
            dec = 0
        if isinstance(dec, (bool, np.bool_)):
            dec = int(dec)
        if not _is_num(dec) or float(dec) not in (0.0, 1.0):
            bad("declined", "must be 0 or 1 when supplied")
        row.update(ts=ts, amount=amt, lat=la, lon=lo, channel=ch, declined=dec)
        out.append(row)
    if errs:
        raise CardInputError(errs, total=len(errs))

    df = pd.DataFrame(out)
    df["row_id"] = np.arange(len(df))
    for c in ("ts", "amount", "lat", "lon"):
        df[c] = df[c].astype(float)
    df["declined"] = df["declined"].astype(int)
    # build_features groups by `card`; a card that is 7 (int) in one row and "7" (str) in another is the same card to a human
    df["card"] = df["card"].astype(str)
    df["mcc"] = df["mcc"].astype(str)
    return df


class CardScorer:
    def __init__(self, directory):
        import xgboost as xgb
        self.dir = Path(directory)
        pol_path, model_path = self.dir / "policy.json", self.dir / "xgb.json"
        for p in (pol_path, model_path):
            if not p.is_file():
                raise CardModelError(f"card model artifact missing: {p.name} (expected in the card model directory; "
                                     f"run scripts/train_card_model.py). No fallback model is used.")
        try:
            self.policy = json.loads(pol_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CardModelError(f"card policy.json is unreadable ({type(exc).__name__})") from exc
        pol = self.policy
        if pol.get("format") != BUNDLE_FORMAT:
            raise CardModelError(f"unsupported card bundle format {pol.get('format')!r} (expected {BUNDLE_FORMAT!r})")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != pol.get("xgb_sha256"):
            raise CardModelError("card xgb.json does not match the checksum recorded in policy.json (corrupted or mismatched artifacts)")
        if list(pol.get("features", [])) != list(FEATURES):
            raise CardModelError("feature list in the card bundle differs from v2.card.features.FEATURES; retrain with scripts/train_card_model.py")
        try:
            self.booster = xgb.Booster()
            self.booster.load_model(str(model_path))
        except Exception as exc:
            raise CardModelError(f"card model could not be loaded ({type(exc).__name__})") from exc
        self.features = list(pol["features"])
        self.threshold = float(pol["threshold"])
        self.ref = np.asarray(pol["legit_score_reference"], float)
        self.model_version = pol["model_version"]
        self._last_latency_ms = 0.0

    # ------------------------------------------------------------------ the single pipeline
    def score_batch(self, rows) -> list[dict]:
        """Score every transaction. Features for a row use only that card's EARLIER rows in the same request, so a card's history
        should be included in the request. Results are returned in the input order."""
        import xgboost as xgb
        t0 = time.perf_counter()
        df = validate_rows(rows)
        f = build_features(df.sort_values(["card", "ts", "row_id"], kind="stable"))
        X = f[self.features].to_numpy(dtype=float)
        scores = self.booster.predict(xgb.DMatrix(X, feature_names=self.features, missing=np.nan))
        pct = np.searchsorted(self.ref, scores, side="right") / len(self.ref)
        res = pd.DataFrame({"row_id": f["row_id"].to_numpy(), "model_score": scores, "legit_percentile": pct,
                            "hist_len": f["hist_len"].to_numpy(int)}).sort_values("row_id")
        ms = float(1000 * (time.perf_counter() - t0))
        out = []
        for r in res.itertuples(index=False):
            fraud = bool(r.model_score >= self.threshold)
            out.append({"index": int(r.row_id), "classification": "FRAUD" if fraud else "LEGITIMATE", "is_fraud": fraud,
                        "model_score": float(r.model_score), "legit_percentile": float(r.legit_percentile),
                        "prior_transactions_on_card": int(r.hist_len)})
        self._last_latency_ms = ms
        return out

    def score_one(self, transaction: dict, history: list[dict] | None = None) -> dict:
        """Score `transaction` given optional prior `history` of the same card. Same pipeline as score_batch."""
        rows = list(history or []) + [transaction]
        res = self.score_batch(rows)
        mine = res[-1]
        mine["index"] = 0
        # the target row is the last one supplied; features only look backwards in time, so rows timed later than it in `history`
        # would not influence it -- but warn the caller because they are almost certainly a mistake
        later = [h for h in (history or []) if isinstance(h, dict) and _is_num(h.get("ts")) and _is_num(transaction.get("ts"))
                 and float(h["ts"]) > float(transaction["ts"])]
        mine["warnings"] = ([f"{len(later)} history row(s) are timed after the transaction and were ignored for its features"] if later else [])
        if not history:
            mine["warnings"].append("no card history supplied: velocity / novelty features are computed as for the card's first transaction")
        return mine

    @staticmethod
    def summarize(results: list[dict]) -> dict:
        n = len(results)
        fr = sum(r["is_fraud"] for r in results)
        scores = np.array([r["model_score"] for r in results], float)
        return {"transactions": n, "flagged_fraud": int(fr), "legitimate": int(n - fr),
                "flagged_rate": (fr / n) if n else None,
                "model_score": {"min": float(scores.min()), "mean": float(scores.mean()), "max": float(scores.max())} if n else None}

    def model_card(self) -> dict:
        p = self.policy
        return {"model_version": self.model_version, "synthetic": p["synthetic"], "data_source": p["data_source"], "seed": p["seed"],
                "trained_utc": p["trained_utc"], "features": self.features, "threshold": self.threshold,
                "operating_point": p["operating_point"], "score_meaning": p["score_meaning"], "evaluation": p["evaluation"],
                "input_schema": {"required": list(REQUIRED), "optional": list(OPTIONAL), "channels": list(CHANNELS), "max_rows": MAX_ROWS},
                "limitations": p["limitations"]}
