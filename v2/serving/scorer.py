"""
v2/serving/scorer.py -- online scoring of one trace with the FULL v2 cascade.

    trace -> core features + history-only graph features
          -> XGBoost probability          (supervised: known fraud patterns)
          -> Isolation Forest + autoencoder percentiles (novelty: unusual vs normal)
          -> stacked probability (logistic regression on [logit(xgb), iforest pct, ae pct])
          -> ALLOW / REVIEW / BLOCK by thresholds learned on out-of-fold scores (cost-based)
Dependencies at runtime: numpy, scipy, xgboost. No scikit-learn, pandas or pydantic here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from v2.serving.core_features import extract_features_core, normalize_record
from v2.serving.drift import DriftMonitor
from v2.serving.numpy_models import NumpyNovelty
from v2.serving.store import EntityStore

DECISIONS = ("ALLOW", "REVIEW", "BLOCK")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ecdf(ref_sorted: np.ndarray, x):
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


class V2Scorer:
    def __init__(self, directory: Path):
        import xgboost as xgb
        self.dir = Path(directory)
        self.policy = json.loads((self.dir / "policy.json").read_text(encoding="utf-8"))
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.dir / "xgb.json"))
        self.novelty = NumpyNovelty(dict(np.load(self.dir / "novelty.npz")))
        self.store = EntityStore.from_json((self.dir / "entity_store.json").read_text(encoding="utf-8"))
        self.feats, self.nov_feats = self.policy["features"], self.policy["novelty_features"]
        self.ecdf_if = np.asarray(self.policy["ecdf_iforest"])
        self.ecdf_ae = np.asarray(self.policy["ecdf_autoencoder"])
        self.coef, self.intercept = np.asarray(self.policy["stack_coef"]), float(self.policy["stack_intercept"])
        self.t_review, self.t_block = self.policy["t_review"], self.policy["t_block"]
        self.model_version = self.policy["model_version"]
        self.drift = DriftMonitor(json.loads((self.dir / "drift_reference.json").read_text(encoding="utf-8")), self.feats)

    # ---- pure scoring from an already-built feature dict (also used by tests / training-side equivalence checks)
    def score_features(self, f: dict) -> dict:
        import xgboost as xgb
        x = np.array([[f[k] for k in self.feats]], dtype=float)
        p_xgb = float(self.booster.predict(xgb.DMatrix(x, missing=np.nan))[0])
        nov = self.novelty.score(np.array([[f[k] for k in self.nov_feats]], dtype=float))
        pct_if, pct_ae = float(ecdf(self.ecdf_if, nov["iforest"])[0]), float(ecdf(self.ecdf_ae, nov["autoencoder"])[0])
        z = float(self.coef @ np.array([float(logit(p_xgb)), pct_if, pct_ae]) + self.intercept)
        stacked = float(1.0 / (1.0 + np.exp(-z)))
        d = 2 if stacked >= self.t_block else (1 if stacked >= self.t_review else 0)
        return {"xgb_probability": p_xgb, "iforest_percentile": pct_if, "autoencoder_percentile": pct_ae,
                "iforest_raw": float(nov["iforest"][0]), "autoencoder_raw": float(nov["autoencoder"][0]),
                "stacked_probability": stacked, "decision_index": d, "decision": DECISIONS[d], "_x": x}

    def explain(self, x: np.ndarray, top: int = 5) -> list[dict]:
        import xgboost as xgb
        contrib = self.booster.predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:top]
        return [{"feature": self.feats[i], "value": None if np.isnan(x[0, i]) else float(x[0, i]),
                 "contribution_log_odds": float(contrib[i])} for i in order if abs(contrib[i]) > 1e-9]

    def score(self, record: dict, commit: bool = False) -> dict:
        t0 = time.perf_counter()
        rec = normalize_record(record)
        f = extract_features_core(rec)
        graph = self.store.features(rec)               # history only
        f.update(graph)
        res = self.score_features(f)
        x = res.pop("_x")
        self.drift.observe(x[0])
        if commit:
            self.store.ingest(rec)
        novelty_note = None
        if res["decision"] != "ALLOW" and res["xgb_probability"] < 0.5 and res["iforest_percentile"] >= 0.98:
            novelty_note = "Flagged mainly because the behaviour is unusual compared with normal traffic, not because it matches known fraud."
        return {**res, "graph_features": graph, "top_features": self.explain(x), "novelty_note": novelty_note,
                "thresholds": {"t_review": self.t_review, "t_block": self.t_block}, "committed": bool(commit),
                "model_version": self.model_version, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def model_card(self) -> dict:
        return {k: self.policy[k] for k in ("model_version", "trained_on", "cost_assumptions", "evaluation_summary", "limitations",
                                            "features", "novelty_features", "t_review", "t_block") if k in self.policy}
