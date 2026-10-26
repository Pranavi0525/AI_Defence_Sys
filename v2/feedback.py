"""
v2/feedback.py -- Phase 7b: analyst feedback loop for the v2 cascade, with an honest before/after comparison.

The loop (one repetition):
  1. Build a World B; hold out one customer-grouped, family-stratified quarter as an UNTOUCHED TEST set.
  2. Round 0: fit the served pipeline (XGBoost + novelty stages + stacker + cost-based thresholds) on the training part.
  3. "Analyst queue": using round-0 OUT-OF-FOLD decisions on the training part only,
        misses       = fraud that was ALLOWed         (found later, e.g. chargebacks; `found_share` of them are found)
        false alarms = legitimate traces that were REVIEWed or BLOCKed
  4. Regenerate FRESH look-alike cases (same fraud family / same legit kind as each mistake, new customers, new seed).
  5. Retrain on training + fresh hard examples ("targeted" arm).
  6. CONTROL arm: retrain on training + the SAME NUMBER of fresh fraud / legit traces drawn from the ordinary mix.
     Without it, any gain could be just "more data", not "learning from mistakes".
  7. Score baseline / targeted / control on the same test set. Repeat over several worlds; report paired differences.

Honest limits (also written into the report):
  * The fresh cases come from the SAME generator that made the test set, so a gain is an in-distribution best case. It says
    nothing about unseen fraud types (see Phase 2/5) or about real analysts, who find only some mistakes, late, and label noisily.
  * Fresh traces use new entity ids; their graph features are replayed within their own batch (no overlap with the original world),
    which gives the same values a shared replay would give.
  * The loop retrains the whole cascade; thresholds are re-learned from inner out-of-fold scores each time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from v2 import gen
from v2.evalkit import NoveltyModels, folds_for, xgb
from v2.features import WORLD_B_HARD, build_world_b, extract_features_v2, graph_features
from v2.fusion import DEFAULTS, Ecdf, _logit, base_scores, choose_thresholds, decide, expected_cost, weights
from v2.phases import FEATS_B, NOV_B

FAMILY_GENERATORS = {"SYNTHETIC_IDENTITY": gen.gen_synthetic_identity, "FAKE_MERCHANT": gen.gen_fake_merchant,
                     "DEEPFAKE_KYC": gen.gen_deepfake_kyc, "AI_SCRIPTED_APP": gen.gen_ai_scripted_app}


# ------------------------------------------------------------------------------------------------ frames
def frame_from_records(recs: list[dict]) -> pd.DataFrame:
    """Same columns as build_world_b: meta + features + history-only graph features (replayed within `recs`)."""
    feats = pd.DataFrame([extract_features_v2(r) for r in recs])
    meta = pd.DataFrame([{"trace_id": r["trace_id"], "customer_id": r["customer_id"], "fraud": r["fraud"],
                          "attack_family": r["attack_family"], "source": r["source"]} for r in recs])
    meta["family"] = np.where(meta.fraud == 0, "legitimate", meta.attack_family)
    meta["legit_kind"] = np.where(meta.fraud == 0, meta.attack_family, "")
    return meta.merge(feats, on="trace_id").merge(graph_features(recs), on="trace_id").reset_index(drop=True)


def _legit_kind_key(legit_kind: str) -> str:
    k = legit_kind.replace("legitimate_v2_", "")
    return "routine" if k in ("", "legitimate_v2") else k


# ------------------------------------------------------------------------------------------------ pipeline (same recipe as v2/serving/bundle.py)
class Pipeline:
    """XGBoost + novelty stages -> stacked logistic regression -> cost-based thresholds learned on inner OOF scores only."""

    def __init__(self, feats=FEATS_B, nov_feats=NOV_B, seed=11, **policy):
        self.feats, self.nov_feats, self.seed, self.policy = feats, nov_feats, seed, policy

    def fit(self, df: pd.DataFrame) -> "Pipeline":
        y = df.fraud.values
        S = np.zeros((len(df), 3))
        for itr, iva in folds_for(df, self.seed + 1, n=4):
            S[iva] = base_scores(df.iloc[itr], df.iloc[iva], self.feats, self.nov_feats)
        self.ecdf_if, self.ecdf_ae = Ecdf(S[y == 0, 1]), Ecdf(S[y == 0, 2])
        self.stack = LogisticRegression(C=1.0, max_iter=1000).fit(self._meta(S), y)
        self.oof_score = self.stack.predict_proba(self._meta(S))[:, 1]
        self.t_review, self.t_block = choose_thresholds(self.oof_score, y, **self.policy)
        self.oof_decision = decide(self.oof_score, self.t_review, self.t_block)
        self.model = xgb(42).fit(df[self.feats].values, y)
        self.nov = NoveltyModels(42).fit(df[df.fraud == 0][self.nov_feats].values)
        return self

    def _meta(self, M):
        return np.c_[_logit(M[:, 0]), self.ecdf_if(M[:, 1]), self.ecdf_ae(M[:, 2])]

    def decide(self, df: pd.DataFrame) -> np.ndarray:
        sc = self.nov.score(df[self.nov_feats].values)
        M = np.c_[self.model.predict_proba(df[self.feats].values)[:, 1], sc["iforest"], sc["autoencoder"]]
        return decide(self.stack.predict_proba(self._meta(M))[:, 1], self.t_review, self.t_block)


# ------------------------------------------------------------------------------------------------ analyst queue + hard examples
def analyst_queue(df: pd.DataFrame, oof_decision: np.ndarray, found_share=1.0, seed=0) -> dict:
    """Mistakes visible to an analyst. Misses are found only with probability `found_share` (late / partial discovery)."""
    rng = np.random.default_rng(seed)
    fraud, legit = df.fraud.values == 1, df.fraud.values == 0
    missed = fraud & (oof_decision == 0)
    found = missed & (rng.random(len(df)) < found_share)
    false_alarm = legit & (oof_decision >= 1)
    return {"misses": df[found], "false_alarms": df[false_alarm], "n_missed_total": int(missed.sum()), "n_misses_found": int(found.sum()),
            "n_false_alarms": int(false_alarm.sum())}


def generate_hard_examples(queue: dict, seed: int, per_case=2, world=WORLD_B_HARD) -> list[dict]:
    """Fresh look-alikes: per mistake, `per_case` new traces of the same fraud family / legit kind (new customers, new seed)."""
    g = gen.G(seed)
    pools = gen.Pools(g, world.get("young_share", 0.0))
    recs = []
    for fam, n in queue["misses"].family.value_counts().items():
        recs += FAMILY_GENERATORS[fam](g, pools, int(n) * per_case, world.get("stealth", 0.0))
    for kind, n in queue["false_alarms"].legit_kind.map(_legit_kind_key).value_counts().items():
        recs += [gen.legit_trace(g, pools, kind, 0) for _ in range(int(n) * per_case)]
    return recs


def generate_control(n_fraud: int, n_legit: int, seed: int, world=WORLD_B_HARD, hard_share=0.45) -> list[dict]:
    """Same NUMBER of fresh fraud / legit traces as the targeted set, but from the ordinary mix (no targeting)."""
    g = gen.G(seed)
    pools = gen.Pools(g, world.get("young_share", 0.0))
    recs = gen.gen_legit(g, pools, n_legit, hard_share, world.get("onboarding_share", 0.053)) if n_legit else []
    fams = list(FAMILY_GENERATORS)
    for i, fam in enumerate(fams):
        n = n_fraud // 4 + (1 if i < n_fraud % 4 else 0)
        if n:
            recs += FAMILY_GENERATORS[fam](g, pools, n, world.get("stealth", 0.0))
    return recs


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(pipe: Pipeline, test: pd.DataFrame, prevalence=DEFAULTS["prevalence"]) -> dict:
    d, y = pipe.decide(test), test.fraud.values
    P = {**DEFAULTS, **pipe.policy}
    w = weights(y, prevalence)
    fraud, legit = y == 1, y == 0
    tpr, fpr = float((d[fraud] >= 1).mean()), float((d[legit] >= 1).mean())
    prec = tpr * prevalence / max(tpr * prevalence + fpr * (1 - prevalence), 1e-12)
    out = {"recall_flag": tpr, "recall_block": float((d[fraud] == 2).mean()), "legit_flag_rate": fpr,
           "legit_block_rate": float((d[legit] == 2).mean()), "precision_flagged_at_0.6pct": float(prec),
           "cost_per_10k": expected_cost(d, y, w, P["L"], P["q"], P["c_r"], P["c_fp"]) * 1e4,
           "t_review": pipe.t_review, "t_block": pipe.t_block}
    for fam in test[test.fraud == 1].family.unique():
        out[f"recall_{fam}"] = float((d[(test.family == fam).values] >= 1).mean())
    for kind in sorted(test[test.fraud == 0].legit_kind.unique()):
        out[f"flag_{_legit_kind_key(kind)}"] = float((d[(test.legit_kind == kind).values] >= 1).mean())
    return out


def run_rep(rep: int, per_case=2, found_share=1.0, world=WORLD_B_HARD) -> dict:
    world_seed = 2026 + 101 * rep
    dfw = build_world_b(seed=world_seed, **world)
    tr, te = folds_for(dfw, 11 + rep, n=4)[0]
    train, test = dfw.iloc[tr].reset_index(drop=True), dfw.iloc[te].reset_index(drop=True)

    base = Pipeline(seed=11 + rep).fit(train)
    q = analyst_queue(train, base.oof_decision, found_share, seed=rep)
    hard = generate_hard_examples(q, seed=900_000 + rep, per_case=per_case, world=world)
    hard_df = frame_from_records(hard)
    ctrl_df = frame_from_records(generate_control(int(hard_df.fraud.sum()), int((1 - hard_df.fraud).sum()), seed=800_000 + rep, world=world))
    for extra in (hard_df, ctrl_df):                    # the fresh traces must be new customers
        assert not set(extra.customer_id) & (set(train.customer_id) | set(test.customer_id)), "customer leak into fresh examples"

    arms = {"baseline": base,
            "targeted": Pipeline(seed=11 + rep).fit(pd.concat([train, hard_df], ignore_index=True)),
            "control_same_size": Pipeline(seed=11 + rep).fit(pd.concat([train, ctrl_df], ignore_index=True))}
    return {"rep": rep, "world_seed": world_seed, "n_train": len(train), "n_test": len(test), "test_fraud": int(test.fraud.sum()),
            "queue": {k: q[k] for k in ("n_missed_total", "n_misses_found", "n_false_alarms")},
            "misses_by_family": q["misses"].family.value_counts().to_dict(),
            "false_alarms_by_kind": q["false_alarms"].legit_kind.map(_legit_kind_key).value_counts().to_dict(),
            "n_fresh": {"targeted_fraud": int(hard_df.fraud.sum()), "targeted_legit": int((1 - hard_df.fraud).sum()),
                        "control_fraud": int(ctrl_df.fraud.sum()), "control_legit": int((1 - ctrl_df.fraud).sum())},
            "arms": {name: evaluate(p, test) for name, p in arms.items()}}


# ------------------------------------------------------------------------------------------------ aggregation
def summarise(reps: list[dict]) -> dict:
    """Mean per arm and PAIRED differences (same test set each rep): targeted-baseline and targeted-control."""
    names = ["baseline", "targeted", "control_same_size"]
    keys = sorted({k for r in reps for a in r["arms"].values() for k in a})
    out = {"n_reps": len(reps), "arms": {n: {} for n in names}, "paired": {}}
    for k in keys:
        vals = {n: np.array([r["arms"][n].get(k, np.nan) for r in reps], float) for n in names}
        for n in names:
            out["arms"][n][k] = float(np.nanmean(vals[n]))
        for a, b in (("targeted", "baseline"), ("targeted", "control_same_size")):
            dlt = vals[a] - vals[b]
            ok = ~np.isnan(dlt)
            if ok.sum():
                out["paired"].setdefault(f"{a}_minus_{b}", {})[k] = {
                    "mean": float(dlt[ok].mean()), "sd": float(dlt[ok].std(ddof=1)) if ok.sum() > 1 else 0.0,
                    "n_up": int((dlt[ok] > 1e-12).sum()), "n_down": int((dlt[ok] < -1e-12).sum()), "n": int(ok.sum())}
    return out
