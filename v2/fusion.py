"""
v2/fusion.py -- Phase 5: nested fusion + cost-aware decision policy.

Everything that is LEARNED (stacker weights, ALLOW/REVIEW/BLOCK thresholds) is learned on inner out-of-fold scores of the
OUTER TRAINING fold only. The outer test fold is touched once, for reporting. (v1 chose thresholds and reported on the
same population.)

Cost model (ASSUMPTIONS, all adjustable, none estimated from real data):
    L   = loss from an ALLOWed fraud                          (default 500)
    q   = share of REVIEWed fraud that analysts actually stop (default 0.90)
    c_r = cost of one manual review                           (default 12)
    c_fp= friction/support cost of BLOCKing a legit customer  (default 25)
    prevalence = assumed production fraud rate                (default 0.6%); legit rows get weight (1-pi)/pi * n_f/n_l
    review_cap = max share of weighted traffic sent to REVIEW (default 5%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from v2.evalkit import NoveltyModels, folds_for, xgb

DEFAULTS = dict(L=500.0, q=0.90, c_r=12.0, c_fp=25.0, prevalence=0.006, review_cap=0.05)


def weights(y, prevalence):
    nf, nl = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    return np.where(y == 1, 1.0, (nf / nl) * (1 - prevalence) / prevalence)


def decide(score, t_r, t_b):
    return np.where(score >= t_b, 2, np.where(score >= t_r, 1, 0))     # 0 ALLOW, 1 REVIEW, 2 BLOCK


def expected_cost(dec, y, w, L, q, c_r, c_fp):
    fraud, legit = y == 1, y == 0
    c = np.zeros(len(y))
    c[fraud & (dec == 0)] = L
    c[fraud & (dec == 1)] = c_r + (1 - q) * L
    c[legit & (dec == 1)] = c_r
    c[legit & (dec == 2)] = c_fp
    return float((w * c).sum() / w.sum())


def choose_thresholds(score, y, **kw):
    P = {**DEFAULTS, **kw}
    w = weights(y, P["prevalence"])
    grid = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 81)))
    best = (np.inf, None, None)
    for i, t_r in enumerate(grid):
        for t_b in grid[i:]:
            d = decide(score, t_r, t_b)
            if (w * (d == 1)).sum() / w.sum() > P["review_cap"]:
                continue
            c = expected_cost(d, y, w, P["L"], P["q"], P["c_r"], P["c_fp"])
            if c < best[0]:
                best = (c, float(t_r), float(t_b))
    if best[1] is None:                                  # cap infeasible: block-only fallback
        best = (np.inf, float(grid[-1]), float(grid[-1]))
    return best[1], best[2]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Ecdf:
    def __init__(self, ref):
        self.ref = np.sort(ref)

    def __call__(self, x):
        return np.searchsorted(self.ref, x, side="right") / len(self.ref)


def base_scores(tr_df, te_df, feats, nov_feats, seed=42):
    m = xgb(seed).fit(tr_df[feats].values, tr_df.fraud.values)
    nov = NoveltyModels(seed).fit(tr_df[tr_df.fraud == 0][nov_feats].values)
    sc = nov.score(te_df[nov_feats].values)
    return np.c_[m.predict_proba(te_df[feats].values)[:, 1], sc["iforest"], sc["autoencoder"]]


def run_nested(dfw: pd.DataFrame, feats, nov_feats, seed=11, **policy):
    P = {**DEFAULTS, **policy}
    y_all = dfw.fraud.values
    strategies = ["xgb_only", "stacked", "naive_max"]
    out_scores = {s: np.zeros(len(dfw)) for s in strategies}
    out_dec = {s: np.zeros(len(dfw), int) for s in strategies}
    thr_log = {s: [] for s in strategies}
    sens_log = []
    for tr, te in folds_for(dfw, seed):
        tr_df, te_df = dfw.iloc[tr].reset_index(drop=True), dfw.iloc[te].reset_index(drop=True)
        ytr = tr_df.fraud.values
        S = np.zeros((len(tr_df), 3))
        for itr, iva in folds_for(tr_df, seed + 1, n=4):
            S[iva] = base_scores(tr_df.iloc[itr], tr_df.iloc[iva], feats, nov_feats)
        St = base_scores(tr_df, te_df, feats, nov_feats)
        ecdf_if, ecdf_ae = Ecdf(S[ytr == 0, 1]), Ecdf(S[ytr == 0, 2])

        def meta(M):
            return np.c_[_logit(M[:, 0]), ecdf_if(M[:, 1]), ecdf_ae(M[:, 2])]

        stack = LogisticRegression(C=1.0, max_iter=1000).fit(meta(S), ytr)
        fns = {"xgb_only": lambda M: M[:, 0],
               "stacked": lambda M: stack.predict_proba(meta(M))[:, 1],
               "naive_max": lambda M: np.maximum(M[:, 0], np.maximum(ecdf_if(M[:, 1]), ecdf_ae(M[:, 2])))}
        for s in strategies:
            sc_tr, sc_te = fns[s](S), fns[s](St)
            t_r, t_b = choose_thresholds(sc_tr, ytr, **policy)
            thr_log[s].append((t_r, t_b))
            out_scores[s][te] = sc_te
            out_dec[s][te] = decide(sc_te, t_r, t_b)
        # sensitivity of the stacked policy to assumptions (thresholds chosen on inner OOF of this fold)
        sc_tr = fns["stacked"](S)
        for prev in (0.002, 0.006, 0.02):
            sens_log.append({"assumption": f"prevalence={prev}", **dict(zip(("t_review", "t_block"), choose_thresholds(sc_tr, ytr, **{**policy, "prevalence": prev})))})
        for cr in (5, 12, 50):
            sens_log.append({"assumption": f"review_cost={cr}", **dict(zip(("t_review", "t_block"), choose_thresholds(sc_tr, ytr, **{**policy, "c_r": cr})))})
    w = weights(y_all, P["prevalence"])
    res = {}
    for s in strategies:
        d = out_dec[s]
        fraud, legit = y_all == 1, y_all == 0
        share = {n: float((w * (d == k)).sum() / w.sum()) for k, n in enumerate(["allow", "review", "block"])}
        flagged = d >= 1
        tpr_block = float((d[fraud] == 2).mean()); tpr_flag = float(flagged[fraud].mean())
        fpr_block = float((d[legit] == 2).mean()); fpr_flag = float(flagged[legit].mean())
        prec_block = tpr_block * P["prevalence"] / max(tpr_block * P["prevalence"] + fpr_block * (1 - P["prevalence"]), 1e-12)
        res[s] = {"traffic_share_at_assumed_prevalence": share, "fraud_recall_review_plus_block": tpr_flag,
                  "fraud_recall_block_only": tpr_block, "legit_flagged_rate": fpr_flag, "legit_blocked_rate": fpr_block,
                  "block_precision_at_assumed_prevalence": float(prec_block),
                  "expected_cost_per_10k_txn": expected_cost(d, y_all, w, P["L"], P["q"], P["c_r"], P["c_fp"]) * 1e4,
                  "thresholds_per_fold": [list(map(float, t)) for t in thr_log[s]]}
    res["allow_everything_cost_per_10k"] = expected_cost(np.zeros(len(y_all), int), y_all, w, P["L"], P["q"], P["c_r"], P["c_fp"]) * 1e4
    sens = pd.DataFrame(sens_log).groupby("assumption").agg(t_review_mean=("t_review", "mean"), t_review_std=("t_review", "std"),
                                                            t_block_mean=("t_block", "mean"), t_block_std=("t_block", "std"))
    res["policy_sensitivity_stacked"] = sens.reset_index().to_dict(orient="records")
    res["assumptions"] = P
    return res


# =====================================================================================================
# Unseen-family policy test: can a fusion layer + policy that is TRAINED to expect unseen fraud do better?
# =====================================================================================================
def _lofo_inner_scores(tr_df, feats, nov_feats, fams_in_train, seed):
    """Inner out-of-fold scores where each fraud family is scored by base models that never saw it.
    Legit rows are scored by grouped folds (k = number of fraud families, so every legit row is validated once)."""
    k = len(fams_in_train)
    S = np.full((len(tr_df), 3), np.nan)
    legit_ix = np.where(tr_df.fraud.values == 0)[0]
    legit_df = tr_df.iloc[legit_ix].reset_index(drop=True)
    lf = folds_for(legit_df.assign(family="legitimate"), seed, n=k)          # grouped by customer
    for i, fam in enumerate(fams_in_train):
        fam_ix = np.where((tr_df.family == fam).values)[0]
        val_legit = legit_ix[lf[i][1]]
        train_ix = np.setdiff1d(np.arange(len(tr_df)), np.r_[fam_ix, val_legit])
        val_ix = np.r_[fam_ix, val_legit]
        S[val_ix] = base_scores(tr_df.iloc[train_ix], tr_df.iloc[val_ix], feats, nov_feats)
    return S


def run_unseen_family_policy(dfw: pd.DataFrame, feats, nov_feats, fams, seed=11, **policy):
    P = {**DEFAULTS, **policy}
    rows = []
    for H in fams:
        held = (dfw.family == H).values
        rest = dfw[~held].reset_index(drop=True)
        acc = {}
        for tr, te in folds_for(rest, seed, n=3):
            tr_df = rest.iloc[tr].reset_index(drop=True)
            test_df = pd.concat([dfw[held], rest.iloc[te]], ignore_index=True)
            ytr, yte = tr_df.fraud.values, test_df.fraud.values
            fams_in = [f for f in fams if f != H]
            S_std = np.zeros((len(tr_df), 3))
            for itr, iva in folds_for(tr_df, seed + 1, n=4):
                S_std[iva] = base_scores(tr_df.iloc[itr], tr_df.iloc[iva], feats, nov_feats)
            S_lofo = _lofo_inner_scores(tr_df, feats, nov_feats, fams_in, seed + 2)
            St = base_scores(tr_df, test_df, feats, nov_feats)
            ecdf_if, ecdf_ae = Ecdf(S_std[ytr == 0, 1]), Ecdf(S_std[ytr == 0, 2])
            meta = lambda M: np.c_[_logit(M[:, 0]), ecdf_if(M[:, 1]), ecdf_ae(M[:, 2])]
            stack_std = LogisticRegression(C=1.0, max_iter=1000).fit(meta(S_std), ytr)
            stack_lofo = LogisticRegression(C=1.0, max_iter=1000).fit(meta(S_lofo), ytr)
            plans = {
                "xgb_only_std": (S_std[:, 0], St[:, 0]),
                "stacked_std": (stack_std.predict_proba(meta(S_std))[:, 1], stack_std.predict_proba(meta(St))[:, 1]),
                "stacked_lofo": (stack_lofo.predict_proba(meta(S_lofo))[:, 1], stack_lofo.predict_proba(meta(St))[:, 1]),
            }
            w = weights(yte, P["prevalence"])
            for name, (s_tr, s_te) in plans.items():
                t_r, t_b = choose_thresholds(s_tr, ytr, **policy)
                d = decide(s_te, t_r, t_b)
                a = acc.setdefault(name, {"recall_flag": [], "recall_block": [], "legit_flag": [], "legit_block": [], "cost": []})
                a["recall_flag"].append(float((d[yte == 1] >= 1).mean())); a["recall_block"].append(float((d[yte == 1] == 2).mean()))
                a["legit_flag"].append(float((d[yte == 0] >= 1).mean())); a["legit_block"].append(float((d[yte == 0] == 2).mean()))
                a["cost"].append(expected_cost(d, yte, w, P["L"], P["q"], P["c_r"], P["c_fp"]) * 1e4)
        for name, a in acc.items():
            rows.append({"held_out_family": H, "policy": name, **{k: float(np.mean(v)) for k, v in a.items()}})
    return rows
