"""
v2/phases.py -- Phase 1 (evaluation fixes), Phase 2 (unseen-family tests), Phase 4 (graph ablation).

TWO WORLDS, evaluated separately (why): pooling the v1 simulator and the v2 generator lets a model learn "which
simulator made this trace" (the new-schema fields are NaN for every v1 trace), which inflates scores for reasons that
have nothing to do with fraud. So:
    World A = v1 Red Team attacks (ATO, APP, MULE) vs Normal World legit.   features: BEHAVIOUR (+GRAPH)
    World B = v2 generator: hard-negative legit vs 4 GenAI-era families.    features: BEHAVIOUR + NEW_SCHEMA + GRAPH
Cross-world TRANSFER (train on A, test on B, common features only) is the 'novel fraud' test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from v2.evalkit import (BEHAVIOUR, GRAPH, NEW_SCHEMA, NoveltyModels, basic_metrics, cluster_bootstrap, ece,
                        folds_for, oof_xgb, xgb, FAMILIES_V1, FAMILIES_V2)

SEEDS = [11, 22, 33]
FEATS_A = BEHAVIOUR + GRAPH
FEATS_B = BEHAVIOUR + NEW_SCHEMA + GRAPH
NOV_A = BEHAVIOUR + GRAPH
NOV_B = BEHAVIOUR + NEW_SCHEMA + GRAPH


def worlds(df):
    A = df[df.source == "v1"].reset_index(drop=True)
    B = df[df.source == "v2_gen"].reset_index(drop=True)
    return A, B


def _fam_recall(dfw, p, fams, thr=0.5):
    return {f: float((p[(dfw.family == f).values] >= thr).mean()) for f in fams}


def _fpr_by_kind(dfw, p, thr=0.5):
    leg = dfw[dfw.fraud == 0]
    out = {}
    for k in sorted(leg.legit_kind.unique()):
        ix = leg.index[leg.legit_kind == k]
        label = k.replace("legitimate_v2_", "").replace("legitimate_v2", "routine_v2").replace("legitimate", "routine_v1")
        out[label] = {"n": int(len(ix)), "fpr": float((p[ix] >= thr).mean())}
    return out


# ------------------------------------------------------------------ Phase 1 (per world)
def phase1_world(dfw: pd.DataFrame, feats: list, fams: list) -> dict:
    y = dfw.fraud.values
    rows, fam_rows, kind_rows = [], [], []
    for s in SEEDS:
        p = oof_xgb(dfw, feats, folds_for(dfw, s))
        rows.append(basic_metrics(y, p)); fam_rows.append(_fam_recall(dfw, p, fams)); kind_rows.append(_fpr_by_kind(dfw, p))
    out = {"n": int(len(dfw)), "n_fraud": int(y.sum()),
           "metrics": {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1))}
                       for k in rows[0]},
           "per_family_recall": {f: float(np.mean([r[f] for r in fam_rows])) for f in fams},
           "fpr_by_legit_kind": {k: {"n": kind_rows[0][k]["n"], "fpr": float(np.mean([r[k]["fpr"] for r in kind_rows]))}
                                 for k in kind_rows[0]}}
    folds = folds_for(dfw, 11)
    # 1.2 Stage 1 learned IN-FOLD: shallow tree; threshold chosen on inner CV predictions of the train fold only.
    #     Reported for several "keep this share of fraud" targets so the trade-off is visible, not one cherry-picked point.
    X1 = dfw[BEHAVIOUR].values
    targets = (0.90, 0.95, 0.99)
    cleared = {t: np.zeros(len(dfw), bool) for t in targets}
    rules_text = None
    for tr, te in folds:
        tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0, min_samples_leaf=10)
        inner = cross_val_predict(tree, X1[tr], y[tr], cv=4, method="predict_proba")[:, 1]
        fs = np.sort(inner[y[tr] == 1])
        tree.fit(X1[tr], y[tr])
        pte = tree.predict_proba(X1[te])[:, 1]
        for t in targets:
            cleared[t][te] = pte < float(fs[int((1 - t) * len(fs))])
        if rules_text is None:
            rules_text = export_text(tree, feature_names=BEHAVIOUR, max_depth=3)
    p = oof_xgb(dfw, feats, folds)
    out["stage1_in_fold"] = {
        "xgb_alone_metrics": basic_metrics(y, p), "example_rules_fold0": rules_text,
        "by_fraud_keep_target": {str(t): {"legit_auto_cleared_share": float(cleared[t][y == 0].mean()),
                                           "fraud_wrongly_cleared": int(cleared[t][y == 1].sum()), "fraud_total": int(y.sum()),
                                           "cascade_metrics": basic_metrics(y, np.where(cleared[t], 0.0, p))} for t in targets}}
    # 1.3 calibration on the same outer folds
    cal = {}
    for name, method in [("raw_xgb", None), ("sigmoid", "sigmoid"), ("isotonic", "isotonic")]:
        pc = oof_xgb(dfw, feats, folds, calibrate=method)
        cal[name] = {"brier": float(brier_score_loss(y, pc)), "ece": ece(y, pc), **basic_metrics(y, pc)}
    out["calibration"] = cal
    # 1.5 cluster bootstrap + precision at 0.6% prevalence
    out["bootstrap_ci"] = cluster_bootstrap(dfw, (p >= 0.5).astype(int))
    return out


# ------------------------------------------------------------------ Phase 2a: leave-one-family-out inside a world
def lofo_world(dfw: pd.DataFrame, feats: list, nov_feats: list, fams: list, seed=11) -> dict:
    res = {}
    for held in fams:
        m_held = (dfw.family == held).values
        rest = dfw[~m_held].reset_index(drop=True)
        acc = {k: [] for k in ["xgb_rec", "iforest_rec", "ae_rec", "lane_rec", "xgb_fpr", "iforest_fpr", "ae_fpr", "lane_fpr"]}
        for tr, te in folds_for(rest, seed):
            tr_df, te_df = rest.iloc[tr], rest.iloc[te]
            test_df = pd.concat([dfw[m_held], te_df], ignore_index=True)
            held_m = np.r_[np.ones(m_held.sum(), bool), np.zeros(len(te_df), bool)]
            legit = test_df.fraud.values == 0
            pr = xgb(42).fit(tr_df[feats].values, tr_df.fraud.values).predict_proba(test_df[feats].values)[:, 1] >= 0.5
            nov = NoveltyModels(42).fit(tr_df[tr_df.fraud == 0][nov_feats].values)
            sc = nov.score(test_df[nov_feats].values)
            fi, fa = sc["iforest"] >= nov.thr["iforest"], sc["autoencoder"] >= nov.thr["autoencoder"]
            lane = pr | fi
            for k, v in [("xgb", pr), ("iforest", fi), ("ae", fa), ("lane", lane)]:
                acc[f"{k}_rec"].append(v[held_m].mean()); acc[f"{k}_fpr"].append(v[legit].mean())
        res[held] = {k: float(np.mean(v)) for k, v in acc.items()}
        res[held]["n_held_out"] = int(m_held.sum())
    return res


# ------------------------------------------------------------------ Phase 2b: transfer World A -> World B
def transfer_A_to_B(dfA, dfB) -> dict:
    feats = BEHAVIOUR            # the only features meaningful in both worlds
    nov_feats = BEHAVIOUR
    out = {}
    mdl = xgb(42).fit(dfA[feats].values, dfA.fraud.values)
    pB = mdl.predict_proba(dfB[feats].values)[:, 1] >= 0.5
    nov = NoveltyModels(42).fit(dfA[dfA.fraud == 0][nov_feats].values)
    sc = nov.score(dfB[nov_feats].values)
    fi = sc["iforest"] >= nov.thr["iforest"]
    legit = dfB.fraud.values == 0
    for name, flag in [("xgb_trained_on_A", pB), ("iforest_trained_on_A_legit", fi), ("xgb_or_iforest", pB | fi)]:
        out[name] = {"fpr_on_B_legit": float(flag[legit].mean()),
                     **{f: float(flag[(dfB.family == f).values].mean()) for f in FAMILIES_V2}}
    return out


# ------------------------------------------------------------------ Phase 4: graph feature ablation (World B and A)
def graph_ablation(dfw, feats_no_graph, feats_graph, fams):
    y = dfw.fraud.values
    out = {}
    for name, feats in [("no_graph", feats_no_graph), ("with_graph", feats_graph)]:
        ms, fr = [], []
        for s in SEEDS:
            p = oof_xgb(dfw, feats, folds_for(dfw, s))
            ms.append(basic_metrics(y, p)); fr.append(_fam_recall(dfw, p, fams))
        out[name] = {"f1": float(np.mean([m["f1"] for m in ms])), "recall": float(np.mean([m["recall"] for m in ms])),
                     "fp": float(np.mean([m["fp"] for m in ms])), "fn": float(np.mean([m["fn"] for m in ms])),
                     "per_family_recall": {f: float(np.mean([r[f] for r in fr])) for f in fams}}
    return out
