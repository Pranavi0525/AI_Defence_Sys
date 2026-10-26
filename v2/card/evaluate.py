"""v2/card/evaluate.py -- time-split and leave-one-pattern-out evaluation of: rules, XGBoost, MLP (neural net), per-card novelty (IsolationForest)."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from .features import FEATURES, RELATIVE
from .rules import rules_cp, rules_naive
from .sim import DAY, PATTERNS

warnings.filterwarnings("ignore")
T_TRAIN, T_VAL = 30 * DAY, 42 * DAY
PREV = 0.006


def split(f):
    return f[f.ts < T_TRAIN], f[(f.ts >= T_TRAIN) & (f.ts < T_VAL)], f[f.ts >= T_VAL]


def fit_models(train, seed):
    X, y = train[FEATURES], train.is_fraud
    xg = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=1,
                       tree_method="hist", eval_metric="aucpr").fit(X, y)
    mlp = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                        MLPClassifier((32, 16), max_iter=300, early_stopping=True, random_state=seed)).fit(X, y)
    leg = train[train.is_fraud == 0]
    imp = SimpleImputer(strategy="constant", fill_value=0).fit(leg[RELATIVE])
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(imp.transform(leg[RELATIVE]))
    return {"xgb": lambda d: xg.predict_proba(d[FEATURES])[:, 1], "mlp": lambda d: mlp.predict_proba(d[FEATURES])[:, 1],
            "percard_iforest": lambda d: -iso.score_samples(imp.transform(d[RELATIVE]))}


def thr_at_fpr(scores_legit, fpr):
    return float(np.quantile(scores_legit, 1 - fpr))


def summarise(test, flag):
    """flag: boolean array over `test` rows. Returns txn recall, event recall, legit FPR, precision at 0.6% prevalence."""
    fr = test.is_fraud.values == 1
    tpr = float(flag[fr].mean()) if fr.any() else np.nan
    fpr = float(flag[~fr].mean())
    ev = test[fr].assign(fl=flag[fr]).groupby("event").fl.max()
    prec = tpr * PREV / (tpr * PREV + fpr * (1 - PREV)) if (tpr * PREV + fpr * (1 - PREV)) > 0 else np.nan
    return dict(txn_recall=tpr, event_recall=float(ev.mean()) if len(ev) else np.nan, events=int(len(ev)), fpr=fpr, precision_at_0p6pct=prec)


def run_cases(f, seed, fpr=0.01, holdout=None):
    tr, va, te = split(f)
    if holdout:
        tr = tr[tr.pattern != holdout]
        te = pd.concat([te[te.is_fraud == 0], f[f.pattern == holdout]])
    fns = fit_models(tr, seed)
    vleg = va[va.is_fraud == 0]
    res, flags = {}, {}
    for nm, fn in fns.items():
        t = thr_at_fpr(fn(vleg), fpr if nm != "lane" else fpr / 2)
        flags[nm] = fn(te) >= t
    half = {nm: fns[nm](te) >= thr_at_fpr(fns[nm](vleg), fpr / 2) for nm in ("xgb", "percard_iforest")}
    flags["lane_xgb_or_percard"] = half["xgb"] | half["percard_iforest"]
    flags["rules_cp"] = rules_cp(te).values == 1
    flags["rules_naive"] = rules_naive(te).values == 1
    for nm, fl in flags.items():
        res[nm] = summarise(te, fl)
    return res


def evaluate(f, seed):
    out = {"time_split": run_cases(f, seed)}
    pats = [p for p in f.pattern.unique() if p != "LEGIT"]
    out["lopo"] = {p: run_cases(f, seed, holdout=p) for p in pats}
    return out
