"""v2/evalkit.py -- shared evaluation helpers (models, folds, metrics, bootstrap, calibration, novelty detectors)."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from eval_v2.splits import customer_grouped_folds
from v2.features import BEHAVIOUR, GRAPH, NEW_SCHEMA

ALL_FEATURES = BEHAVIOUR + NEW_SCHEMA + GRAPH
NOVELTY_FEATURES = BEHAVIOUR + GRAPH           # no NaN-bearing schema fields
FAMILIES_V1 = ["ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"]
FAMILIES_V2 = ["SYNTHETIC_IDENTITY", "FAKE_MERCHANT", "DEEPFAKE_KYC", "AI_SCRIPTED_APP"]
FRAUD_FAMILIES = FAMILIES_V1 + FAMILIES_V2


def xgb(seed=42, **kw):
    p = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
             eval_metric="aucpr", random_state=seed, n_jobs=1, tree_method="exact")
    p.update(kw)
    return XGBClassifier(**p)


def folds_for(df, seed, n=5):
    return customer_grouped_folds(df.reset_index(drop=True), n, seed, strata_col="family")


# ------------------------------------------------------------------ novelty detectors (legit-only training)
class NoveltyModels:
    """IsolationForest + small autoencoder, both fit on LEGITIMATE rows of the training fold only."""
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X_legit: np.ndarray):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_legit)
        X_legit = self.imp.transform(X_legit)
        self.qt = QuantileTransformer(n_quantiles=min(200, len(X_legit)), output_distribution="normal",
                                      random_state=self.seed).fit(X_legit)
        Z = self.qt.transform(X_legit)
        self.iforest = IsolationForest(n_estimators=200, random_state=self.seed).fit(Z)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ae = MLPRegressor(hidden_layer_sizes=(12, 4, 12), max_iter=400, random_state=self.seed,
                                   early_stopping=False).fit(Z, Z)
        self.thr = {}
        s = self.score(X_legit)
        for k in s:
            self.thr[k] = float(np.quantile(s[k], 0.98))     # flags ~2% of training legit
        return self

    def score(self, X: np.ndarray) -> dict:
        Z = self.qt.transform(self.imp.transform(X))
        return {"iforest": -self.iforest.score_samples(Z), "autoencoder": ((self.ae.predict(Z) - Z) ** 2).mean(axis=1)}


# ------------------------------------------------------------------ metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def prevalence_precision(tpr, fpr, prev):
    d = tpr * prev + fpr * (1 - prev)
    return tpr * prev / d if d > 0 else np.nan


def cluster_bootstrap(df, pred, n_boot=500, seed=0, prev=0.006):
    """Percentile CIs, resampling CUSTOMERS (not rows) so repeated customers don't make CIs too tight."""
    rng = np.random.default_rng(seed)
    y = df["fraud"].values
    groups = df["customer_id"].values
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    res = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([idx_by[g] for g in pick])
        yy, pp = y[ix], pred[ix]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        tpr = ((pp == 1) & (yy == 1)).sum() / yy.sum()
        fpr = ((pp == 1) & (yy == 0)).sum() / (yy == 0).sum()
        prec = ((pp == 1) & (yy == 1)).sum() / max((pp == 1).sum(), 1)
        res.append((tpr, fpr, prec, prevalence_precision(tpr, fpr, prev)))
    a = np.array(res)
    out = {}
    for i, n in enumerate(["recall", "fpr", "precision_synthetic_mix", "precision_at_0.6pct_prevalence"]):
        out[n] = {"mean": float(np.nanmean(a[:, i])), "lo": float(np.nanpercentile(a[:, i], 2.5)),
                  "hi": float(np.nanpercentile(a[:, i], 97.5))}
    return out


def oof_xgb(df, feats, folds, calibrate=None, seed=42):
    X, y = df[feats].values, df["fraud"].values
    p = np.zeros(len(df))
    for tr, te in folds:
        if calibrate is None:
            m = xgb(seed).fit(X[tr], y[tr])
        else:
            m = CalibratedClassifierCV(xgb(seed), method=calibrate, cv=3).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def basic_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, p),
            "fp": int(((pred == 1) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum())}
