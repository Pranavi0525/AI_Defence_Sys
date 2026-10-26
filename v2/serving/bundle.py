"""
v2/serving/bundle.py -- TRAINING-side export of the served model (needs scikit-learn; the API never imports this file).

Same procedure as the nested evaluation in v2/fusion.py, applied once to the whole training world:
  1. inner out-of-fold base scores (customer-grouped 4-fold)  -> stacker + thresholds are learned on OUT-OF-FOLD scores only
  2. final base models fit on all data
  3. export: xgb.json, novelty.npz, policy.json, drift_reference.json, entity_store.json   (all portable, no pickles)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from v2.evalkit import NoveltyModels, folds_for, xgb
from v2.fusion import DEFAULTS, Ecdf, _logit, base_scores, choose_thresholds
from v2.serving import drift
from v2.serving.numpy_models import export_novelty
from v2.serving.store import replay_features

LIMITATIONS = [
    "Trained and evaluated on SYNTHETIC data only; scores are not real-world fraud probabilities.",
    "The attack families and the stealth level are assumptions made by the author, not estimated from real payments.",
    "Unseen fraud types are mostly missed by the supervised model; novelty detectors catch some (e.g. deepfake-KYC style) but not others (e.g. fake merchants).",
    "Legitimate 'new payee then pay' and person-to-person spree behaviour is flagged noticeably often (see evaluation_summary).",
    "Thresholds depend on assumed fraud prevalence and cost constants (cost_assumptions); change them and the thresholds change.",
]


def fit_bundle(dfw, records, feats, nov_feats, out_dir, seed=11, evaluation_summary=None, **policy):
    P = {**DEFAULTS, **policy}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = dfw.fraud.values

    # 1. inner OOF scores -> stacker + thresholds (never fit on in-sample scores)
    S = np.zeros((len(dfw), 3))
    for itr, iva in folds_for(dfw, seed + 1, n=4):
        S[iva] = base_scores(dfw.iloc[itr], dfw.iloc[iva], feats, nov_feats)
    ref_if, ref_ae = np.sort(S[y == 0, 1]), np.sort(S[y == 0, 2])
    ecdf_if, ecdf_ae = Ecdf(ref_if), Ecdf(ref_ae)
    meta = lambda M: np.c_[_logit(M[:, 0]), ecdf_if(M[:, 1]), ecdf_ae(M[:, 2])]
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(meta(S), y)
    oof_stacked = stack.predict_proba(meta(S))[:, 1]
    t_r, t_b = choose_thresholds(oof_stacked, y, **policy)

    # 2. final base models on all data
    model = xgb(42).fit(dfw[feats].values, y)
    nov = NoveltyModels(42).fit(dfw[dfw.fraud == 0][nov_feats].values)

    # 3. export portable artifacts
    model.get_booster().save_model(str(out / "xgb.json"))
    nov_arrays = export_novelty(nov)
    np.savez(out / "novelty.npz", **nov_arrays)
    _, store = replay_features(records)
    (out / "entity_store.json").write_text(store.to_json(), encoding="utf-8")
    (out / "drift_reference.json").write_text(json.dumps(drift.build_reference(dfw[feats].values, feats)), encoding="utf-8")
    digest = hashlib.sha256()
    for name in ("xgb.json", "entity_store.json"):
        digest.update((out / name).read_bytes())
    for k in sorted(nov_arrays):          # hash array CONTENT (zip bytes embed timestamps and would change every run)
        digest.update(np.ascontiguousarray(nov_arrays[k]).tobytes())
    policy_doc = {
        "model_version": f"v2-{digest.hexdigest()[:10]}", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": feats, "novelty_features": nov_feats,
        "ecdf_iforest": ref_if.tolist(), "ecdf_autoencoder": ref_ae.tolist(),
        "stack_coef": stack.coef_[0].tolist(), "stack_intercept": float(stack.intercept_[0]),
        "t_review": t_r, "t_block": t_b,
        "trained_on": {"traces": int(len(dfw)), "fraud": int(y.sum()), "families": sorted(set(dfw.family) - {"legitimate"}),
                       "note": "synthetic World B (v2 generator)"},
        "cost_assumptions": {k: P[k] for k in ("L", "q", "c_r", "c_fp", "prevalence", "review_cap")},
        "evaluation_summary": evaluation_summary or {}, "limitations": LIMITATIONS,
    }
    (out / "policy.json").write_text(json.dumps(policy_doc), encoding="utf-8")
    return {"xgb": model, "novelty": nov, "stack": stack, "ecdf_if": ecdf_if, "ecdf_ae": ecdf_ae, "t_review": t_r, "t_block": t_b,
            "policy": policy_doc}
