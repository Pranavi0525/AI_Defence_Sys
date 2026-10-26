"""
Tests for risk_fusion.py.

SCOPE NOTE (part of the Blue Team readiness review that preceded this
file): before this test module, NOTHING under the Blue Team path
(blue_team_pipeline.py, cascade_with_graph.py, cascade_with_autoencoder.py,
decision_policy.py, gcn.py, autoencoder.py) had any test coverage at
all. This file only covers risk_fusion.py's own new code, specifically
the pure `fit_fusion_oof` meta-fusion function (fast, no base-model
training required) plus the metric-formatting helpers. It does NOT
retroactively add coverage for the pre-existing Blue Team files -- that
gap should be closed separately and is called out, not hidden, in this
module's docstring and in STAGE_STATUS.md.

These tests intentionally avoid touching NormalWorld / XGBoost / the
real GCN or Autoencoder training loops, so they run in well under a
second and can't flake on model-training randomness.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

import risk_fusion as rf


def make_folds(y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_dummy = np.zeros((len(y), 1))
    return list(skf.split(X_dummy, y))


class TestFitFusionOOF:
    def test_output_shape_and_bounds(self):
        """Fused score must be one probability per row, in [0, 1]."""
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, 2, size=n)
        meta_X = rng.uniform(0, 1, size=(n, 3))
        folds = make_folds(y)

        fused, fold_coefs = rf.fit_fusion_oof(meta_X, y, folds)

        assert fused.shape == (n,)
        assert np.all(fused >= 0.0) and np.all(fused <= 1.0)
        assert len(fold_coefs) == len(folds)

    def test_every_row_gets_a_prediction(self):
        """Every row must appear in exactly one fold's test_idx, so no
        row should be left at its zero-initialized default by accident
        (which would silently look like 'confidently legitimate')."""
        rng = np.random.default_rng(1)
        n = 150
        y = rng.integers(0, 2, size=n)
        meta_X = rng.uniform(0, 1, size=(n, 3))
        folds = make_folds(y)

        # every index must appear in exactly one test_idx across folds
        all_test_idx = np.concatenate([test_idx for _, test_idx in folds])
        assert sorted(all_test_idx.tolist()) == list(range(n))

        fused, _ = rf.fit_fusion_oof(meta_X, y, folds)
        # with random uniform features, scores should not degenerately
        # collapse to a single value
        assert fused.std() > 0.0

    def test_strong_single_signal_dominates_fusion(self):
        """If one base score is a near-perfect separator and the other
        two are pure noise, fusion should learn to weight the informative
        one heavily and still classify well -- sanity check that the
        meta-model is actually learning something, not just averaging
        blindly."""
        rng = np.random.default_rng(2)
        n = 400
        y = rng.integers(0, 2, size=n)
        # stage_1_2: strongly correlated with y
        strong_signal = np.clip(y + rng.normal(0, 0.08, size=n), 0, 1)
        # gcn / ae: pure noise, uncorrelated with y
        noise_1 = rng.uniform(0, 1, size=n)
        noise_2 = rng.uniform(0, 1, size=n)
        meta_X = np.column_stack([strong_signal, noise_1, noise_2])
        folds = make_folds(y)

        fused, fold_coefs = rf.fit_fusion_oof(meta_X, y, folds)
        preds = (fused >= 0.5).astype(int)
        accuracy = (preds == y).mean()

        assert accuracy > 0.85, (
            f"Fusion should recover a near-perfect single-feature signal, "
            f"got accuracy={accuracy:.3f}"
        )
        avg_stage_1_2_weight = np.mean([abs(f["stage_1_2_score"]) for f in fold_coefs])
        avg_noise_weight = np.mean(
            [abs(f["gcn_score"]) + abs(f["ae_score"]) for f in fold_coefs]
        ) / 2
        assert avg_stage_1_2_weight > avg_noise_weight, (
            "The informative signal should be weighted more heavily than "
            "the noise signals, on average."
        )

    def test_always_zero_feature_gets_learned_away(self):
        """Mirrors the real-corpus finding: when GCN score is 0 for
        every row (no graph edges above the noise threshold), the
        meta-model should not blow up or let that dead feature corrupt
        the fused score -- it should simply carry ~no information."""
        rng = np.random.default_rng(3)
        n = 300
        y = rng.integers(0, 2, size=n)
        strong_signal = np.clip(y + rng.normal(0, 0.1, size=n), 0, 1)
        always_zero = np.zeros(n)
        noise = rng.uniform(0, 1, size=n)
        meta_X = np.column_stack([strong_signal, always_zero, noise])
        folds = make_folds(y)

        fused, fold_coefs = rf.fit_fusion_oof(meta_X, y, folds)
        preds = (fused >= 0.5).astype(int)
        accuracy = (preds == y).mean()

        assert accuracy > 0.85
        assert np.all(np.isfinite(fused))

    def test_deterministic_given_fixed_seed(self):
        """Same inputs + same random_state should reproduce identical
        fused scores -- important for reproducibility claims in any
        writeup that reports these numbers."""
        rng = np.random.default_rng(4)
        n = 100
        y = rng.integers(0, 2, size=n)
        meta_X = rng.uniform(0, 1, size=(n, 3))
        folds = make_folds(y)

        fused_1, _ = rf.fit_fusion_oof(meta_X, y, folds, random_state=42)
        fused_2, _ = rf.fit_fusion_oof(meta_X, y, folds, random_state=42)

        np.testing.assert_array_equal(fused_1, fused_2)


class TestBlockMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.9, 0.8])
        result, preds = rf.block_metrics(y, proba, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert list(preds) == [0, 0, 1, 1]

    def test_threshold_is_respected(self):
        y = np.array([0, 1])
        proba = np.array([0.4, 0.6])
        _, preds_low = rf.block_metrics(y, proba, threshold=0.3)
        _, preds_high = rf.block_metrics(y, proba, threshold=0.7)
        assert list(preds_low) == [1, 1]
        assert list(preds_high) == [0, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
