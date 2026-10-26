"""v2/serving/drift.py -- Population Stability Index (PSI) drift monitoring. numpy only.

PSI per feature compares the live distribution of recently scored traces with the training distribution:
  < 0.10 stable | 0.10-0.25 watch | > 0.25 drift.  New fraud usually shows up first as drift in a few features.
"""
from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-4
MIN_SAMPLES = 50


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0..len(edges)-2 for value bins, last index = NaN bin."""
    idx = np.digitize(x, edges[1:-1]) if len(edges) > 2 else np.zeros(len(x), int)
    return np.where(np.isnan(x), len(edges) - 1, idx)


def build_reference(matrix: np.ndarray, names: list[str], n_bins: int = 10) -> dict:
    ref = {}
    for j, name in enumerate(names):
        col = matrix[:, j].astype(float)
        good = col[~np.isnan(col)]
        edges = np.unique(np.quantile(good, np.linspace(0, 1, n_bins + 1))) if len(good) else np.array([0.0, 1.0])
        if len(edges) < 2:
            edges = np.array([edges[0], edges[0] + 1.0])
        idx = _bin_index(col, edges)
        props = np.bincount(idx, minlength=len(edges)) / len(col)
        ref[name] = {"edges": edges.tolist(), "proportions": props.tolist()}
    return ref


def psi(reference: dict, matrix: np.ndarray, names: list[str]) -> dict:
    out = {}
    for j, name in enumerate(names):
        edges = np.asarray(reference[name]["edges"])
        e = np.clip(np.asarray(reference[name]["proportions"]), EPS, None)
        a = np.bincount(_bin_index(matrix[:, j].astype(float), edges), minlength=len(e))[: len(e)] / len(matrix)
        a = np.clip(a, EPS, None)
        out[name] = float(np.sum((a - e) * np.log(a / e)))
    return out


def status(value: float) -> str:
    return "stable" if value < 0.10 else ("watch" if value < 0.25 else "drift")


class DriftMonitor:
    def __init__(self, reference: dict, names: list[str], window: int = 500):
        self.reference, self.names = reference, names
        self.buffer: deque = deque(maxlen=window)

    def observe(self, vector: np.ndarray) -> None:
        self.buffer.append(np.asarray(vector, dtype=float))

    def report(self) -> dict:
        n = len(self.buffer)
        if n < MIN_SAMPLES:
            return {"status": "insufficient_data", "n_observed": n, "min_required": MIN_SAMPLES}
        vals = psi(self.reference, np.vstack(self.buffer), self.names)
        top = sorted(vals.items(), key=lambda kv: -kv[1])[:8]
        worst = top[0][1]
        return {"status": status(worst), "n_observed": n, "max_psi": worst,
                "top_features": [{"feature": k, "psi": v, "status": status(v)} for k, v in top]}
