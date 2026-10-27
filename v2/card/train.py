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
from xgboost import XGBClassifier

from .evaluate import PREV, split, thr_at_fpr
from .features import FEATURES, build_features
from .sim import DAY, simulate

BUNDLE_FORMAT = "card-bundle-1"
DEFAULT_SEED = 0
DEFAULT_FPR = 0.01
# Same as v2.card.evaluate.fit_models (xgb branch); random_state is supplied from the seed.
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, n_jobs=1,
                  tree_method="hist", eval_metric="aucpr")
SWEEP_FPRS = (0.05, 0.02, 0.01, 0.005, 0.001)

LIMITATIONS = [
    "Trained and evaluated on SYNTHETIC simulator data only (v2.card.sim); all fraud patterns and rates are the author's assumptions.",
    "The score is a model score, NOT a calibrated or real-world fraud probability. Its meaning is: how this transaction ranks against "
    "legitimate validation traffic (see score_meaning).",
    "Very high leave-one-pattern-out results in the Phase 10 report may reflect signals shared across simulated patterns; they do not "
    "demonstrate detection of novel real-world fraud.",
    "Precision depends on fraud prevalence. 'precision_at_assumed_0p6pct' uses an ASSUMED 0.6% prevalence; it is not a measured production precision.",
    "Not validated on any real transaction dataset. Do not use for real authorisation decisions.",
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


def train_bundle(out_dir, seed: int = DEFAULT_SEED, fpr: float = DEFAULT_FPR, n_cards: int = 1500, days: int = 60,
                 events_per_pattern: int = 40, frame: pd.DataFrame | None = None) -> dict:
    """Train on the simulator (or on `frame`, an already-featured frame with the simulator's columns, for tests) and write the bundle.
    Returns the policy dict that was written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if frame is None:
        f = build_features(simulate(n_cards=n_cards, days=days, seed=seed, events_per_pattern=events_per_pattern))
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

    booster_path = out / "xgb.json"
    clf.get_booster().save_model(str(booster_path))
    # reference distribution of LEGIT validation scores, 1001 quantiles (for legit_percentile at serving time)
    ref = np.quantile(vleg, np.linspace(0, 1, 1001))

    policy = {
        "format": BUNDLE_FORMAT,
        "model_version": f"card-xgb-synthetic-seed{seed}-{_sha256(booster_path)[:8]}",
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "SYNTHETIC: v2.card.sim.simulate",
        "synthetic": True,
        "seed": seed,
        "simulator": {"n_cards": n_cards, "days": days, "events_per_pattern": events_per_pattern} if frame is None else {"custom_frame": True},
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
            "threshold_sweep_by_validation_fpr": sweep,
            "assumed_prevalence_for_precision_at_assumed_0p6pct": PREV,
        },
        "limitations": LIMITATIONS,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=1), encoding="utf-8")
    return policy
