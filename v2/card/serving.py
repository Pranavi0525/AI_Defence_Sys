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
    ts        event time in seconds, 0 <= ts <= limits.MAX_TS_SECONDS (hour-of-day features use ts mod 86400, i.e. UTC)
    amount    INR, > 0 and <= the configured business limit (env CARD_MAX_TXN_AMOUNT, default limits.DEFAULT_MAX_TXN_AMOUNT).
              The limit is INPUT VALIDATION: a request above it is rejected (HTTP 422), never scored.
    mcc       merchant category code: integer 0-9999 or a string of 1-4 digits (normalised to 4 digits, so 5411 == "5411")
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
from .limits import CURRENCY, MAX_TS_SECONDS, MCC_MAX, LimitsConfigError, max_txn_amount

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


def validate_rows(rows, amount_limit: float | None = None) -> pd.DataFrame:
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

    if amount_limit is None:
        try:
            amount_limit = max_txn_amount()
        except LimitsConfigError as exc:
            raise CardModelError(str(exc)) from exc
    else:
        if not _is_num(amount_limit) or float(amount_limit) <= 0:
            raise LimitsConfigError(f"amount_limit={amount_limit!r} must be a positive finite number (INR)")
        amount_limit = float(amount_limit)
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
            elif k == "mcc":
                m = str(v).strip()
                if not (m.isascii() and m.isdigit() and len(m) <= 4 and 0 <= int(m) <= MCC_MAX):
                    bad("mcc", "must be an integer 0-9999 or a string of 1-4 digits")
                else:
                    v = m.zfill(4)
            row[k] = v.strip() if isinstance(v, str) else v
        ts = r.get("ts"); amt = r.get("amount"); la = r.get("lat"); lo = r.get("lon")
        if not _is_num(ts) or not 0 <= float(ts) <= MAX_TS_SECONDS:
            bad("ts", f"must be a finite number of seconds in [0, {MAX_TS_SECONDS:.0f}]")
        if not _is_num(amt) or float(amt) <= 0:
            bad("amount", "must be a finite number > 0")
        elif float(amt) > amount_limit:
            bad("amount", f"exceeds the configured maximum transaction amount of {amount_limit:,.2f} {CURRENCY} "
                          f"(business limit, env CARD_MAX_TXN_AMOUNT); not scored")
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
    df.attrs["amount_limit"] = amount_limit
    return df


class CardScorer:
    def __init__(self, directory, max_amount: float | None = None):
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
        self.feature_support = pol.get("feature_support") or {}
        self.amount_support = pol.get("amount_support") or {}
        try:
            self.amount_limit = max_txn_amount() if max_amount is None else float(max_amount)
            if not math.isfinite(self.amount_limit) or self.amount_limit <= 0:
                raise LimitsConfigError(f"max_amount={self.amount_limit!r} must be a positive finite number (INR)")
        except LimitsConfigError as exc:
            raise CardModelError(str(exc)) from exc
        train_max = self.amount_support.get("train_max_amount")
        self.train_max_amount = float(train_max) if train_max is not None else None
        self.serving_limit_exceeds_training_support = bool(
            self.train_max_amount is not None and self.amount_limit > self.train_max_amount
        )
        if self.serving_limit_exceeds_training_support:
            import warnings
            warnings.warn(
                f"Configured serving limit ({self.amount_limit:,.2f} {CURRENCY}) exceeds recorded "
                f"training support ({self.train_max_amount:,.2f} {CURRENCY}). Model extrapolation warnings "
                "will be emitted for amounts above training support.",
                UserWarning,
                stacklevel=2,
            )
        self._last_latency_ms = 0.0

    # ------------------------------------------------------------------ the single pipeline
    def _score_df(self, df: pd.DataFrame, with_features: bool = False):
        import xgboost as xgb
        t0 = time.perf_counter()
        f = build_features(df.sort_values(["card", "ts", "row_id"], kind="stable"))
        X = f[self.features].to_numpy(dtype=float)
        scores = self.booster.predict(xgb.DMatrix(X, feature_names=self.features, missing=np.nan))
        pct = np.searchsorted(self.ref, scores, side="right") / len(self.ref)
        outside_lists: list[list[str]] = [[] for _ in range(len(f))]
        for feat, sup in self.feature_support.items():
            if feat in f.columns:
                arr = f[feat].to_numpy(dtype=float)
                bad_mask = np.isfinite(arr) & ((arr < sup["min"]) | (arr > sup["max"]))
                for idx in np.flatnonzero(bad_mask):
                    outside_lists[idx].append(feat)
        res = pd.DataFrame({"row_id": f["row_id"].to_numpy(), "model_score": scores, "legit_percentile": pct,
                            "hist_len": f["hist_len"].to_numpy(int), "features_outside": outside_lists})
        if with_features:
            res = pd.concat([res, f[self.features].add_prefix("f_").reset_index(drop=True)], axis=1)
        res = res.sort_values("row_id")
        self._last_latency_ms = float(1000 * (time.perf_counter() - t0))
        return res

    def _rows_out(self, res: pd.DataFrame) -> list[dict]:
        out = []
        for r in res.itertuples(index=False):
            fraud = bool(r.model_score >= self.threshold)
            outside = list(getattr(r, "features_outside", []))
            outside_range = bool(outside)
            score_reliable = not outside_range
            warnings = []
            if outside_range:
                warnings.append(
                    "feature(s) " + ", ".join(outside) + " lie outside the range the model was trained on; "
                    "a tree model cannot extrapolate, so treat this score with extra caution"
                )
            out.append({"index": int(r.row_id), "classification": "FRAUD" if fraud else "LEGITIMATE", "is_fraud": fraud,
                        "model_score": float(r.model_score), "legit_percentile": float(r.legit_percentile),
                        "prior_transactions_on_card": int(r.hist_len),
                        "outside_training_range": outside_range,
                        "score_reliable": score_reliable,
                        "features_outside_training_range": outside,
                        "warnings": warnings})
        return out

    def score_batch(self, rows) -> list[dict]:
        """Score every transaction. Features for a row use only that card's EARLIER rows in the same request, so a card's history
        should be included in the request. Results are returned in the input order."""
        df = validate_rows(rows, amount_limit=self.amount_limit)
        return self._rows_out(self._score_df(df))

    def score_one(self, transaction: dict, history: list[dict] | None = None) -> dict:
        """Score `transaction` given optional prior `history` of the same card. Same pipeline as score_batch, after the history is
        cleaned: rows of OTHER cards, rows timed after the transaction, exact duplicates of another history row and copies of the
        transaction itself are dropped (and counted in `history_dropped`). Nothing is imputed. Validation covers history and
        transaction together, so error row numbers refer to history rows 0..n-1 and the transaction is row n."""
        hist_in = list(history or [])
        try:
            df = validate_rows(hist_in + [transaction], amount_limit=self.amount_limit)
        except CardInputError as exc:
            n_h = len(hist_in)
            errs = []
            for e in exc.errors:
                e = dict(e)
                if e.get("row") == n_h:
                    e["row"], e["source"] = None, "transaction"       # the transaction itself, not a history row
                elif e.get("row") is not None:
                    e["source"] = "history"
                errs.append(e)
            raise CardInputError(errs, total=exc.total) from exc
        n = len(df)
        tx = df.iloc[n - 1]
        key_cols = ["card", "ts", "amount", "mcc", "channel", "lat", "lon", "merchant", "declined"]
        h = df.iloc[:n - 1]
        other = (h["card"] != tx["card"]).to_numpy()
        later = ((h["ts"] > tx["ts"]) & ~pd.Series(other, index=h.index)).to_numpy()
        same_as_tx = np.logical_and.reduce([h[c].to_numpy() == tx[c] for c in key_cols]) if len(h) else np.zeros(0, bool)
        same_as_tx = same_as_tx & ~other & ~later
        cand = ~(other | later | same_as_tx)
        dup = np.zeros(len(h), bool)
        if len(h):
            dup = (h[key_cols].duplicated(keep="first")).to_numpy() & cand
        keep = cand & ~dup
        kept = pd.concat([h[keep], df.iloc[[n - 1]]], ignore_index=True)
        kept["row_id"] = np.arange(len(kept))
        res = self._score_df(kept, with_features=True)
        mine = self._rows_out(res)[-1]
        mine["index"] = 0
        last = res.iloc[-1]
        feats = {k: (None if not math.isfinite(float(last["f_" + k])) else float(last["f_" + k])) for k in self.features}
        dropped = {"other_card": int(other.sum()), "later_than_transaction": int(later.sum()),
                   "same_as_transaction": int(same_as_tx.sum()), "duplicate": int(dup.sum())}
        mine["history_rows_supplied"] = len(hist_in)
        mine["history_rows_used"] = int(keep.sum())
        mine["history_dropped"] = dropped
        mine["features"] = feats
        warnings = []
        if dropped["other_card"]:
            warnings.append(f"{dropped['other_card']} history row(s) belong to a different card and were ignored")
        if dropped["later_than_transaction"]:
            warnings.append(f"{dropped['later_than_transaction']} history row(s) are timed after the transaction and were ignored for its features")
        if dropped["same_as_transaction"]:
            warnings.append(f"{dropped['same_as_transaction']} history row(s) are identical to the transaction itself and were ignored")
        if dropped["duplicate"]:
            warnings.append(f"{dropped['duplicate']} duplicate history row(s) were ignored (first occurrence kept)")
        if not history:
            warnings.append("no card history supplied: velocity / novelty features are computed as for the card's first transaction")
        if mine["outside_training_range"]:
            warnings.append("feature(s) " + ", ".join(mine["features_outside_training_range"]) + " lie outside the range the model was trained on; a tree model cannot "
                            "extrapolate, so treat this score with extra caution")
        mine["warnings"] = warnings
        return mine

    @staticmethod
    def summarize(results: list[dict]) -> dict:
        n = len(results)
        fr = sum(r["is_fraud"] for r in results)
        outside = sum(bool(r.get("outside_training_range")) for r in results)
        scores = np.array([r["model_score"] for r in results], float)
        return {"transactions": n, "flagged_fraud": int(fr), "legitimate": int(n - fr),
                "flagged_rate": (fr / n) if n else None,
                "outside_training_range": int(outside),
                "model_score": {"min": float(scores.min()), "mean": float(scores.mean()), "max": float(scores.max())} if n else None}

    def model_card(self) -> dict:
        p = self.policy
        return {"model_version": self.model_version, "synthetic": p["synthetic"], "data_source": p["data_source"], "seed": p["seed"],
                "trained_utc": p["trained_utc"], "features": self.features, "threshold": self.threshold,
                "operating_point": p["operating_point"], "score_meaning": p["score_meaning"], "evaluation": p["evaluation"],
                "input_schema": {"required": list(REQUIRED), "optional": list(OPTIONAL), "channels": list(CHANNELS), "max_rows": MAX_ROWS,
                         "currency": CURRENCY, "max_transaction_amount": self.amount_limit, "max_transaction_amount_env": "CARD_MAX_TXN_AMOUNT"},
                "amount_support": self.amount_support, "feature_support": self.feature_support,
                "serving_limit_exceeds_training_support": self.serving_limit_exceeds_training_support,
                "limitations": p["limitations"]}
