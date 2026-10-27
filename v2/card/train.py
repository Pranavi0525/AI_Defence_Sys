"""v2/card/train.py -- TRAINING-side export of the served card-fraud model (needs scikit-learn only for the shared helpers).

Everything here reuses the existing Phase 10 pieces; nothing is re-implemented:
  * data      : v2.card.sim.simulate   (SYNTHETIC simulator, fixed seed)
  * features  : v2.card.features.build_features / FEATURES   (the 27 causal features, unchanged)
  * split     : v2.card.evaluate.split  (train first 30 days, validate 30-42, test 42-60)
  * threshold : v2.card.evaluate.thr_at_fpr  (score at a fixed false-positive rate on validation LEGIT traffic)
  * XGBoost   : same hyper-parameters as v2.card.evaluate.fit_models (kept in XGB_PARAMS below)

The served model is fit on the TRAIN block only; the threshold is chosen on the VALIDATION block; the reported metrics
are measured on the held-out TEST block. Nothing is fit on test rows.

Artifacts (all portable, no pickles) are written to a card-specific directory (default: models/card/), never to the payment
model directories:
    xgb.json     the XGBoost booster
    policy.json  feature order, threshold, score reference distribution, provenance, evaluation, limitations, sha256 of xgb.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .evaluate import PREV, split, thr_at_fpr
from .features import FEATURES, build_features
from .limits import CURRENCY, DEFAULT_SIM_MAX_AMOUNT
from .sim import DAY, simulate

BUNDLE_FORMAT = "card-bundle-1"
DEFAULT_SEED = 0
DEFAULT_FPR = 0.01
# Same as v2.card.evaluate.fit_models (xgb branch); random_state is supplied from the seed.
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, n_jobs=1,
                  tree_method="hist", eval_metric="aucpr")
SWEEP_FPRS = (0.05, 0.02, 0.01, 0.005, 0.001)
# features whose TRAINING range is recorded in policy.json so serving can say "this value is outside anything the model was fitted on"
SUPPORT_FEATURES = ("amount", "log_amount", "amt_z", "amt_ratio_max", "sum_24h", "cnt_10m", "cnt_1h", "cnt_24h", "hist_len")
HIGH_VALUE_AMOUNT = 20_000.0      # INR; used ONLY to define a reporting slice ("high-value"), never as a model input or rule

LIMITATIONS = [
    "Trained and evaluated on SYNTHETIC simulator data only (v2.card.sim); all fraud patterns and rates are the author's assumptions.",
    "The score is a model score, NOT a calibrated or real-world fraud probability. Its meaning is: how this transaction ranks against "
    "legitimate validation traffic (see score_meaning).",
    "Very high leave-one-pattern-out results in the Phase 10 report may reflect signals shared across simulated patterns; they do not "
    "demonstrate detection of novel real-world fraud.",
    "Precision depends on fraud prevalence. 'precision_at_assumed_0p6pct' uses an ASSUMED 0.6% prevalence; it is not a measured production precision.",
    "Not validated on any real transaction dataset. Do not use for real authorisation decisions.",
    "Amounts are INR per authorisation. The model has only seen amounts up to policy.amount_support.train_max_amount; a tree model cannot "
    "extrapolate, so beyond the largest split it learned every amount is scored like the same large amount. The API therefore rejects "
    "amounts above the configured CARD_MAX_TXN_AMOUNT (a business limit, not a model output) and reports features outside the training range.",
    "How large legitimate purchases, spending spikes and the other scenarios look, and how common they are, are the author's "
    "ASSUMPTIONS in v2/card/sim.py (extended mode). Performance on them says nothing about how real fraud behaves.",
]

SCORE_MEANING = ("model_score is the raw output of an XGBoost classifier trained on synthetic simulator data. It is only meaningful as a "
                 "ranking: 'legit_percentile' is the share of legitimate VALIDATION transactions with a lower score. A transaction is "
                 "labelled FRAUD when model_score >= threshold, where the threshold is the score exceeded by ~1% of legitimate "
                 "validation transactions. It is not a probability of fraud in the real world.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int(((y == 1) & flag).sum()); fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum()); tn = int(((y == 0) & ~flag).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    fpr = fp / (fp + tn) if (fp + tn) else None
    prec_assumed = (rec * PREV / (rec * PREV + fpr * (1 - PREV))) if rec is not None and fpr is not None and (rec * PREV + fpr * (1 - PREV)) > 0 else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, false_positive_rate=fpr,
                precision_at_assumed_0p6pct=prec_assumed)


def _slice(y: np.ndarray, flag: np.ndarray, mask: np.ndarray) -> dict:
    """Recall on fraud rows and false-positive rate on legit rows inside `mask` (None where the slice has no such rows)."""
    f, l = mask & (y == 1), mask & (y == 0)
    return {"fraud_rows": int(f.sum()), "recall": float(flag[f].mean()) if f.any() else None,
            "legit_rows": int(l.sum()), "false_positive_rate": float(flag[l].mean()) if l.any() else None}


def _by_pattern(te: pd.DataFrame, flag: np.ndarray) -> dict:
    fr = te.assign(fl=flag)[te.is_fraud.values == 1]
    out = {}
    for pat, g in fr.groupby("pattern"):
        out[str(pat)] = {"rows": int(len(g)), "events": int(g.event.nunique()), "row_recall": float(g.fl.mean()),
                         "event_recall": float(g.groupby("event").fl.max().mean()), "median_amount": float(g.amount.median())}
    return out


def train_bundle(out_dir, seed: int = DEFAULT_SEED, fpr: float = DEFAULT_FPR, n_cards: int = 1500, days: int = 60,
                 events_per_pattern: int = 40, frame: pd.DataFrame | None = None, extended: bool = True,
                 max_amount: float | None = None) -> dict:
    """Train on the simulator (or on `frame`, an already-featured frame with the simulator's columns, for tests) and write the bundle.
    `extended=True` (default) uses the extended simulator (large legitimate purchases, spending spikes, new cards, organic event
    timing ...); `extended=False` reproduces the original Phase 10 training data exactly. Returns the policy dict that was written."""
    max_amount = float(DEFAULT_SIM_MAX_AMOUNT if max_amount is None else max_amount)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if frame is None:
        f = build_features(simulate(n_cards=n_cards, days=days, seed=seed, events_per_pattern=events_per_pattern,
                                    extended=extended, max_amount=max_amount))
    else:
        f = frame
    tr, va, te = split(f)
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        if part.is_fraud.sum() == 0 or (part.is_fraud == 0).sum() == 0:
            raise ValueError(f"{name} block must contain both fraud and legitimate rows to train/evaluate the card model")

    clf = XGBClassifier(random_state=seed, **XGB_PARAMS).fit(tr[FEATURES], tr.is_fraud)
    score = lambda d: clf.predict_proba(d[FEATURES])[:, 1]
    vleg = score(va[va.is_fraud == 0])
    thr = thr_at_fpr(vleg, fpr)

    st = score(te); yt = te.is_fraud.values
    flag = st >= thr
    ev = te[yt == 1].assign(fl=flag[yt == 1]).groupby("event").fl.max()
    sweep = {}
    for r in SWEEP_FPRS:
        t = thr_at_fpr(vleg, r)
        sweep[f"{r:g}"] = {"threshold": t, **_confusion(yt, st >= t)}

    pr_auc = float(average_precision_score(yt, st)); roc_auc = float(roc_auc_score(yt, st))
    val_flag = score(va) >= thr; yv = va.is_fraud.values
    big = te.amount.values >= HIGH_VALUE_AMOUNT
    slices = {f"amount_ge_{int(HIGH_VALUE_AMOUNT)}": _slice(yt, flag, big), f"amount_lt_{int(HIGH_VALUE_AMOUNT)}": _slice(yt, flag, ~big),
              "hist_len_lt_5": _slice(yt, flag, te.hist_len.values < 5)}
    support = {k: {"min": float(np.nanmin(tr[k])), "max": float(np.nanmax(tr[k]))} for k in SUPPORT_FEATURES}

    booster_path = out / "xgb.json"
    clf.get_booster().save_model(str(booster_path))
    # reference distribution of LEGIT validation scores, 1001 quantiles (for legit_percentile at serving time)
    ref = np.quantile(vleg, np.linspace(0, 1, 1001))

    policy = {
        "format": BUNDLE_FORMAT,
        "model_version": f"card-xgb-synthetic-seed{seed}-{_sha256(booster_path)[:8]}",
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "SYNTHETIC: v2.card.sim.simulate" + ("(extended=True)" if (frame is None and extended) else ""),
        "synthetic": True,
        "seed": seed,
        "simulator": ({"n_cards": n_cards, "days": days, "events_per_pattern": events_per_pattern, "extended": bool(extended),
                       "max_amount": max_amount} if frame is None else {"custom_frame": True}),
        "amount_support": {"currency": CURRENCY, "train_max_amount": float(tr.amount.max()),
                           "train_max_amount_legit": float(tr.loc[tr.is_fraud == 0, "amount"].max()),
                           "train_max_amount_fraud": float(tr.loc[tr.is_fraud == 1, "amount"].max())},
        "feature_support": support,
        "split_days": {"train_end": 30, "validation_end": 42, "test_end": days},
        "features": list(FEATURES),
        "xgb_params": {**XGB_PARAMS, "random_state": seed},
        "operating_point": {"target_false_positive_rate_on_validation_legit": fpr},
        "threshold": float(thr),
        "legit_score_reference": ref.tolist(),
        "score_meaning": SCORE_MEANING,
        "xgb_sha256": _sha256(booster_path),
        "evaluation": {
            "label": "SYNTHETIC held-out test block of the simulator (days 42-60); not real-world evidence",
            "test_rows": int(len(te)), "test_fraud_rows": int(yt.sum()), "test_fraud_events": int(len(ev)),
            "event_recall": float(ev.mean()),
            "at_threshold": _confusion(yt, flag),
            "pr_auc": pr_auc, "roc_auc": roc_auc, "test_fraud_prevalence": float(yt.mean()),
            "validation": {"label": "threshold-selection block (days 30-42): disjoint from the fit block (days 0-30) and from the test block",
                           "legit_rows": int((yv == 0).sum()), "target_false_positive_rate": fpr,
                           "achieved_false_positive_rate": float(val_flag[yv == 0].mean()), "fraud_rows": int(yv.sum()),
                           "recall": float(val_flag[yv == 1].mean()) if yv.any() else None},
            "by_pattern": _by_pattern(te, flag),
            "slices": slices,
            "threshold_sweep_by_validation_fpr": sweep,
            "assumed_prevalence_for_precision_at_assumed_0p6pct": PREV,
        },
        "limitations": LIMITATIONS,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=1), encoding="utf-8")
    return policy
