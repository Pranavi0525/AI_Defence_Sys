"""
v2/serving/numpy_models.py -- scikit-learn-free inference for the novelty detectors.

Why: the deployed API pins scikit-learn 1.6.1 while training may run on a different version; pickled sklearn models are
not portable across versions. So training EXPORTS plain arrays (export_novelty) and serving runs pure numpy/scipy
(NumpyNovelty). tests/test_v2_serving.py asserts the numpy scores equal scikit-learn's to ~1e-9.

Mirrors exactly: SimpleImputer(median) -> QuantileTransformer(normal) -> IsolationForest.score_samples and
MLPRegressor.predict (reconstruction MSE), as used by v2/evalkit.NoveltyModels.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

BOUNDS_THRESHOLD = 1e-7


def _avg_path_length(n):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    m = n > 2
    out[m] = 2.0 * (np.log(n[m] - 1.0) + np.euler_gamma) - 2.0 * (n[m] - 1.0) / n[m]
    return out


def export_novelty(nov) -> dict:
    """nov: a fitted v2.evalkit.NoveltyModels. Returns a dict of numpy arrays (save with np.savez)."""
    arr = {"imp_stats": np.asarray(nov.imp.statistics_, float),
           "qt_quantiles": np.asarray(nov.qt.quantiles_, float), "qt_references": np.asarray(nov.qt.references_, float)}
    lefts, rights, feats, thrs, leafv, offsets = [], [], [], [], [], [0]
    for tree, tf in zip(nov.iforest.estimators_, nov.iforest.estimators_features_):
        t = tree.tree_
        depth = t.compute_node_depths()
        leaf_value = depth + _avg_path_length(t.n_node_samples) - 1.0
        gfeat = np.where(t.feature >= 0, np.asarray(tf)[np.clip(t.feature, 0, None)], -1)
        lefts.append(t.children_left); rights.append(t.children_right); feats.append(gfeat)
        thrs.append(t.threshold); leafv.append(leaf_value); offsets.append(offsets[-1] + t.node_count)
    arr.update({"if_left": np.concatenate(lefts), "if_right": np.concatenate(rights), "if_feature": np.concatenate(feats),
                "if_threshold": np.concatenate(thrs), "if_leaf_value": np.concatenate(leafv),
                "if_offsets": np.asarray(offsets, dtype=np.int64),
                "if_denominator": np.asarray([len(nov.iforest.estimators_) * _avg_path_length([nov.iforest._max_samples])[0]])})
    arr["ae_n_layers"] = np.asarray([len(nov.ae.coefs_)])
    for i, (w, b) in enumerate(zip(nov.ae.coefs_, nov.ae.intercepts_)):
        arr[f"ae_w{i}"], arr[f"ae_b{i}"] = np.asarray(w, float), np.asarray(b, float)
    return arr


class NumpyNovelty:
    def __init__(self, arrays):
        self.a = {k: np.asarray(arrays[k]) for k in arrays}

    def _impute_transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        mask = np.isnan(X)
        X[mask] = np.broadcast_to(self.a["imp_stats"], X.shape)[mask]
        q, refs = self.a["qt_quantiles"], self.a["qt_references"]
        clip_min = ndtri(BOUNDS_THRESHOLD - np.spacing(1))
        clip_max = ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
        for j in range(X.shape[1]):
            col, qj = X[:, j].copy(), q[:, j]
            lo = col - BOUNDS_THRESHOLD < qj[0]
            hi = col + BOUNDS_THRESHOLD > qj[-1]
            col = 0.5 * (np.interp(col, qj, refs) - np.interp(-col, -qj[::-1], -refs[::-1]))
            col[hi], col[lo] = 1.0, 0.0
            X[:, j] = np.clip(ndtri(col), clip_min, clip_max)
        return X

    def _iforest(self, Z):
        a = self.a
        off = a["if_offsets"]
        Z32 = Z.astype(np.float32).astype(np.float64)       # sklearn trees compare float32 inputs
        depths = np.zeros(len(Z))
        for t in range(len(off) - 1):
            s, e = off[t], off[t + 1]
            left, right = a["if_left"][s:e], a["if_right"][s:e]
            feat, thr, lv = a["if_feature"][s:e], a["if_threshold"][s:e], a["if_leaf_value"][s:e]
            for i in range(len(Z)):
                node = 0
                while left[node] != -1:
                    node = left[node] if Z32[i, feat[node]] <= thr[node] else right[node]
                depths[i] += lv[node]
        return 2.0 ** (-depths / float(a["if_denominator"][0]))

    def _autoencoder(self, Z):
        n = int(self.a["ae_n_layers"][0])
        h = Z
        for i in range(n):
            h = h @ self.a[f"ae_w{i}"] + self.a[f"ae_b{i}"]
            if i != n - 1:
                h = np.maximum(h, 0.0)
        return ((h - Z) ** 2).mean(axis=1)

    def score(self, X) -> dict:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = self._impute_transform(X)
        return {"iforest": self._iforest(Z), "autoencoder": self._autoencoder(Z)}
