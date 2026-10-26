"""
v2/public_check.py -- Phase 8: an external METHOD CHECK of the Stage 2 recipe on a real public transaction dataset.

This is NOT the AI Defence System. The public dataset has no sessions, devices, payees or graph, so Stage 1, the graph features,
the novelty stack and the dossiers do not apply. What it checks is only the modelling METHOD from Phases 1-5:
  (1) does a time-ordered split give a different answer from a random split?            (Phase 1 asked this for customers)
  (2) does calibration (sigmoid / isotonic, fitted on a LATER validation block) help or hurt?   (Phase 1: it hurt on synthetic data)
  (3) is precision at the natural fraud rate low, and what recall do we get at a fixed alert budget?   (Phase 5 style, no assumed costs)
  (4) is XGBoost better than a plain logistic regression?                               (a baseline, never skipped)
  (5) does a model trained on SHUFFLED labels score at the base rate?                   (leakage control; if not, the pipeline leaks)

Dataset (the standard one): Kaggle "Credit Card Fraud Detection" (ULB / Worldline), file creditcard.csv: Time, V1..V28 (PCA-anonymised),
Amount, Class. About 284,807 rows, 492 frauds, two days of European card transactions in 2013. Licence: Database Contents License (DbCL) v1.0.

Honest limits (also written into every report):
  * Only about two days of data: a time split tests short-term ordering, not long-term drift.
  * No card / customer id: the customer-leakage question from Phase 1 cannot be tested here, and the same cards may sit on both sides of any split.
  * Features are PCA components; nothing here says anything about interpretable features or about my simulator's realism.
  * Per-fold fraud counts are small (tens); intervals are wide. Bootstrap over rows ignores dependence between transactions.
  * A high score on this dataset is easy to get and proves little by itself; the comparisons (time vs random, raw vs calibrated, XGB vs LR) are the useful part.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V_COLS = [f"V{i}" for i in range(1, 29)]
EXPECTED = {"rows": 284807, "fraud": 492}
BUDGETS = (0.001, 0.005, 0.01)          # alert budget as a share of test transactions
SPLIT = (0.6, 0.2, 0.2)                 # train / validation (calibration) / test


# ------------------------------------------------------------------------------------------------ data
def load_creditcard(path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    missing = [c for c in ["Time", "Amount", "Class", *V_COLS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is not the expected creditcard.csv; missing columns {missing}")
    notes = []
    if len(df) != EXPECTED["rows"] or int(df.Class.sum()) != EXPECTED["fraud"]:
        notes.append(f"Row / fraud counts ({len(df)} / {int(df.Class.sum())}) differ from the standard file ({EXPECTED['rows']} / {EXPECTED['fraud']}); "
                     "check you downloaded the original dataset.")
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    return df, notes


def make_X(df: pd.DataFrame) -> np.ndarray:
    """V1..V28 and log1p(Amount). The raw Time column is NOT a feature (it would encode position in the sequence)."""
    return np.c_[df[V_COLS].values, np.log1p(df["Amount"].values)]


def time_split(n: int, fr=SPLIT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological blocks on a Time-sorted frame: train < validation < test, no overlap."""
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def random_split(y: np.ndarray, seed: int, fr=SPLIT):
    """Stratified random split with the SAME block sizes as time_split."""
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for cls in (0, 1):
        ii = rng.permutation(np.where(y == cls)[0])
        a, b = int(len(ii) * fr[0]), int(len(ii) * (fr[0] + fr[1]))
        parts[0].append(ii[:a]); parts[1].append(ii[a:b]); parts[2].append(ii[b:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


# ------------------------------------------------------------------------------------------------ models
def fit_xgb(X, y, seed=42):
    from xgboost import XGBClassifier
    # Same hyper-parameters as the project's Stage 2 recipe (v2.evalkit.xgb); tree_method "hist" instead of "exact" for speed on 170k rows.
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                         eval_metric="aucpr", random_state=seed, n_jobs=-1, tree_method="hist").fit(X, y)


def fit_logreg(X, y, seed=42):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)).fit(X, y)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrators(p_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Fit on the VALIDATION block (later than training, earlier than test). Implemented by hand: no dependence on sklearn's prefit API."""
    sig = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_val).reshape(-1, 1), y_val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val, y_val)
    return {"raw": lambda p: p, "sigmoid": lambda p: sig.predict_proba(_logit(p).reshape(-1, 1))[:, 1], "isotonic": lambda p: iso.predict(p)}


# ------------------------------------------------------------------------------------------------ metrics
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-FREQUENCY bins (equal-width bins are mostly empty at 0.2% prevalence)."""
    order = np.argsort(p, kind="mergesort")
    err, n = 0.0, len(y)
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / n * abs(float(y[chunk].mean()) - float(p[chunk].mean()))
    return err


def at_budget(y: np.ndarray, p: np.ndarray, rate: float) -> dict:
    """Flag the top ceil(rate * n) scores: recall and precision of that alert list (ties broken by position, stable)."""
    k = max(1, math.ceil(rate * len(y)))
    top = np.argsort(-p, kind="mergesort")[:k]
    caught = int(y[top].sum())
    return {"k": k, "recall": caught / max(int(y.sum()), 1), "precision": caught / k}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "n_fraud": int(y.sum()), "prevalence": float(y.mean()),
           "average_precision": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "ece_10_quantile_bins": float(ece(y, p))}
    for r in BUDGETS:
        b = at_budget(y, p, r)
        out[f"recall_at_{r * 100:g}pct_alerts"], out[f"precision_at_{r * 100:g}pct_alerts"] = b["recall"], b["precision"]
    return out


def bootstrap_ci(y, p, fn, B=200, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------------------------------------ experiments
def evaluate_split(X, y, split, seed=42, boot=200) -> dict:
    tr, va, te = split
    out = {"sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
           "fraud": {"train": int(y[tr].sum()), "validation": int(y[va].sum()), "test": int(y[te].sum())}, "models": {}}
    for name, fit in (("xgboost", fit_xgb), ("logreg", fit_logreg)):
        m = fit(X[tr], y[tr], seed)
        p_va, p_te = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
        cal = fit_calibrators(p_va, y[va])
        res = {k: metrics(y[te], f(p_te)) for k, f in cal.items()}
        res["raw"]["average_precision_ci95"] = bootstrap_ci(y[te], p_te, average_precision_score, boot, seed)
        res["raw"]["recall_at_0.5pct_alerts_ci95"] = bootstrap_ci(y[te], p_te, lambda a, b: at_budget(a, b, 0.005)["recall"], boot, seed)
        out["models"][name] = res
    return out


def label_shuffle_control(X, y, split, seed=42, repeats=3) -> dict:
    """Train on SHUFFLED labels: average precision on the true test labels must be close to the base rate. If not, the pipeline leaks."""
    tr, _, te = split
    rng = np.random.default_rng(seed)
    aps = []
    for r in range(repeats):
        m = fit_xgb(X[tr], rng.permutation(y[tr]), seed + r)
        aps.append(float(average_precision_score(y[te], m.predict_proba(X[te])[:, 1])))
    return {"mean_average_precision": float(np.mean(aps)), "per_repeat": aps, "test_prevalence": float(y[te].mean())}


def rolling_origin(X, y, n_blocks=6, seed=42) -> list[dict]:
    """Expanding-window folds: train on blocks 0..k-1, test on block k (k = 2..n_blocks-1). Shows variance across time."""
    blocks = np.array_split(np.arange(len(y)), n_blocks)
    out = []
    for k in range(2, n_blocks):
        tr, te = np.concatenate(blocks[:k]), blocks[k]
        m = fit_xgb(X[tr], y[tr], seed)
        p = m.predict_proba(X[te])[:, 1]
        r = {"fold": k, "train_rows": int(len(tr)), "test_rows": int(len(te)), "test_fraud": int(y[te].sum())}
        if y[te].sum() > 0:
            r.update(average_precision=float(average_precision_score(y[te], p)), recall_at_0p5pct_alerts=at_budget(y[te], p, 0.005)["recall"])
        out.append(r)
    return out
