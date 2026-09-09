"""
Regression test for Phase 4A: fold-local preprocessing standardization in
risk_fusion.compute_base_scores().

BUG (see reports/stage_leakage_audit_risk_fusion_decision_policy.md):
mu/sigma used to standardize features were previously computed once over
the ENTIRE dataset before the fold loop ran. That means every fold's
held-out (test) rows influenced the mean/std used to standardize
themselves -- a preprocessing-leakage violation of the OOF principle
(not a label leak: mu/sigma never touch `y`).

FIX: mu/sigma are now computed per-fold from X_raw[train_idx] only, and
applied (frozen) to every row scored in that fold, including its
held-out rows.

THIS TEST DOES NOT REIMPLEMENT THE STANDARDIZATION MATH INDEPENDENTLY
AND COMPARE (that was the tautological version, caught and rejected
during Phase 4A review). Instead it calls compute_base_scores() itself,
with the expensive base-model internals (GCN, Autoencoder, Stage 1+2
cascade) monkeypatched to fast, transparent stand-ins that RECORD the
exact standardized arrays production code hands them. That lets the
test observe the real internal boundary -- what compute_base_scores
actually computes for mu/sigma/X_std per fold -- without ever
recomputing that boundary itself.

Four properties are checked, directly against compute_base_scores():
  1. Training-fold mean does not depend on held-out (test_idx) values.
  2. Training-fold std does not depend on held-out (test_idx) values.
  3. Changing a held-out value does not alter the training statistics.
  4. The held-out row is transformed using the FROZEN training
     statistics (not its own, or the full-dataset, statistics).

The same test is first run against the ORIGINAL (pre-fix) implementation
of compute_base_scores (reconstructed inline below as
`_buggy_compute_base_scores`, byte-for-byte the pre-fix logic) to prove
it fails there, then against the current risk_fusion.compute_base_scores
to prove it passes. Both results are asserted in this file, so a
regression in either direction fails the suite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import risk_fusion as rf
import blue_team_pipeline as btp


# ---------------------------------------------------------------------------
# Synthetic dataset: small, deterministic, with an EXTREME held-out value on
# one feature so that (if leakage exists) it visibly drags the mean/std used
# to standardize the training fold.
# ---------------------------------------------------------------------------
N_TRAIN = 20
N_TEST = 4
FEATURE_COLS = btp.FEATURE_COLS
EXTREME_VALUE = 10_000.0


def _make_df(extreme_value=EXTREME_VALUE):
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    data = {}
    for col in FEATURE_COLS:
        data[col] = rng.normal(loc=1.0, scale=0.5, size=n)
    df = pd.DataFrame(data)
    # All train rows legitimate (fraud=0); all test rows fraud=1 -- this
    # makes legit_train_idx == train_idx exactly, so the autoencoder's
    # "normal_train" call in compute_base_scores standardizes/exposes the
    # WHOLE train fold, not a subset, which is what this test inspects.
    df["fraud"] = [0] * N_TRAIN + [1] * N_TEST
    df["trace_id"] = [f"trace_{i}" for i in range(n)]
    df["attack_family"] = ["legitimate"] * N_TRAIN + ["ATO"] * N_TEST
    # Put an extreme value on ONE feature, ONE held-out (test) row only.
    df.loc[N_TRAIN, FEATURE_COLS[0]] = extreme_value
    return df.reset_index(drop=True)


def _fixed_folds(df):
    """Single fold: rows [0, N_TRAIN) train, rows [N_TRAIN, N_TRAIN+N_TEST) test."""
    train_idx = np.arange(N_TRAIN)
    test_idx = np.arange(N_TRAIN, N_TRAIN + N_TEST)
    return [(train_idx, test_idx)]


class _RecordingAE:
    """Stands in for autoencoder.Autoencoder. Records every array its
    reconstruction_error() is called with, verbatim, so the test can
    inspect exactly what compute_base_scores standardized -- without
    running any real training or reimplementing the standardization."""

    def __init__(self, in_dim, hidden_dim, seed):
        self.in_dim = in_dim

    def reconstruction_error(self, X):
        _CAPTURED["ae_reconstruction_calls"].append(np.array(X, copy=True))
        return np.zeros(len(X))


class _RecordingGCN:
    def __init__(self, in_dim, hidden_dim, seed):
        self.p = np.zeros(10_000)  # never indexed: connected_mask is all-False below


def _noop_train_ae(ae, X, epochs, lr):
    _CAPTURED["ae_train_calls"].append(np.array(X, copy=True))


def _noop_train_gcn(gcn, M, y, train_mask, epochs, lr):
    _CAPTURED["gcn_train_calls"].append(np.array(M, copy=True))


def _noop_error_to_score(test_err, normal_train_err, high_percentile):
    return np.zeros(len(test_err))


_CAPTURED = {}


def _run_compute_base_scores(monkeypatch, df, compute_fn):
    """Runs the given compute_base_scores implementation with the heavy
    base-model internals replaced by transparent recorders, and returns
    the captured arrays. `compute_fn` is either the CURRENT (fixed)
    risk_fusion.compute_base_scores, or the reconstructed pre-fix
    version below -- same monkeypatching, same recorders, either way."""
    global _CAPTURED
    _CAPTURED = {"ae_train_calls": [], "ae_reconstruction_calls": [], "gcn_train_calls": []}

    monkeypatch.setattr(rf, "Autoencoder", _RecordingAE)
    monkeypatch.setattr(rf, "OneLayerGCN", _RecordingGCN)
    monkeypatch.setattr(rf, "train_ae", _noop_train_ae)
    monkeypatch.setattr(rf, "train_gcn", _noop_train_gcn)
    monkeypatch.setattr(rf, "error_to_score", _noop_error_to_score)

    folds = _fixed_folds(df)
    monkeypatch.setattr(
        btp, "compute_stage_1_2_cascade",
        lambda df, feature_cols, cfg, n_splits: (np.zeros(len(df)), np.zeros(len(df), dtype=bool), folds),
    )

    n = len(df)
    A = np.zeros((n, n))  # no graph edges -> connected_mask all False, GCN never scores
    connected_mask = np.zeros(n, dtype=bool)

    compute_fn(df, A, connected_mask, n_splits=1)
    return dict(_CAPTURED)


def _buggy_compute_base_scores(df, A, connected_mask, n_splits=5):
    """Byte-for-byte reconstruction of the PRE-FIX compute_base_scores
    body (global mu/sigma fit before the fold loop). Kept here only so
    this test can prove it fails against the old behavior; this is not
    imported or used anywhere else in the codebase."""
    feature_cols = FEATURE_COLS
    X_raw = df[feature_cols].fillna(0).values.astype(float)
    y = df["fraud"].values.astype(int)

    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    X_std = (X_raw - mu) / sigma

    stage_1_2_proba, escalate, folds = btp.compute_stage_1_2_cascade(
        df, feature_cols, btp.CONFIG, n_splits=n_splits
    )

    gcn_score = np.zeros(len(df))
    ae_score = np.zeros(len(df))

    A_hat = rf.normalize_adjacency(A)
    M = A_hat @ X_std

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        gcn = rf.OneLayerGCN(in_dim=X_std.shape[1], hidden_dim=rf.GCN_HIDDEN_DIM,
                              seed=rf.RANDOM_STATE + fold)
        rf.train_gcn(gcn, M, y.astype(float), train_mask, epochs=rf.GCN_EPOCHS, lr=rf.GCN_LR)
        for i in test_idx:
            if connected_mask[i]:
                gcn_score[i] = gcn.p[i]

        legit_train_idx = train_idx[y[train_idx] == 0]
        ae = rf.Autoencoder(in_dim=X_std.shape[1], hidden_dim=rf.AE_HIDDEN_DIM,
                             seed=rf.RANDOM_STATE + fold)
        rf.train_ae(ae, X_std[legit_train_idx], epochs=rf.AE_EPOCHS, lr=rf.AE_LR)
        normal_train_err = ae.reconstruction_error(X_std[legit_train_idx])
        test_err = ae.reconstruction_error(X_std[test_idx])
        rf.error_to_score(test_err, normal_train_err, high_percentile=rf.HIGH_PERCENTILE)

    return stage_1_2_proba, gcn_score, ae_score, y, folds


class TestFoldLocalStandardization:
    """Checked against the CURRENT (fixed) risk_fusion.compute_base_scores."""

    def test_training_stats_independent_of_held_out_values(self, monkeypatch):
        df_normal = _make_df(extreme_value=1.0)          # no outlier
        df_extreme = _make_df(extreme_value=EXTREME_VALUE)  # extreme held-out outlier

        captured_normal = _run_compute_base_scores(monkeypatch, df_normal, rf.compute_base_scores)
        captured_extreme = _run_compute_base_scores(monkeypatch, df_extreme, rf.compute_base_scores)

        # ae_train_calls[0] is X_std_fold[legit_train_idx] == X_std_fold[train_idx]
        # (all train rows are legitimate in this synthetic df).
        train_std_normal = captured_normal["ae_train_calls"][0]
        train_std_extreme = captured_extreme["ae_train_calls"][0]

        # Property 1-3: an extreme value that ONLY exists in the held-out
        # rows must not change the standardized TRAINING rows at all --
        # i.e. train-fold mu/sigma did not move.
        np.testing.assert_allclose(train_std_normal, train_std_extreme, rtol=0, atol=1e-10)

    def test_held_out_row_uses_frozen_training_stats(self, monkeypatch):
        df = _make_df(extreme_value=EXTREME_VALUE)
        captured = _run_compute_base_scores(monkeypatch, df, rf.compute_base_scores)

        train_std = captured["ae_train_calls"][0]  # X_std_fold[train_idx]
        # Reconstruct train-only mu/sigma independently (from RAW data,
        # not from any standardization logic) purely to invert what
        # scale was actually applied to the frozen train stats.
        X_raw = df[FEATURE_COLS].values.astype(float)
        mu_train = X_raw[:N_TRAIN].mean(axis=0)
        sigma_train = X_raw[:N_TRAIN].std(axis=0) + 1e-8

        # The captured held-out (test) standardized row, from the second
        # reconstruction_error call, must equal (raw_test - mu_train) / sigma_train
        # -- i.e. transformed with TRAIN stats, not its own or full-data stats.
        test_std_captured = captured["ae_reconstruction_calls"][1]  # X_std_fold[test_idx]
        expected_test_std = (X_raw[N_TRAIN:N_TRAIN + N_TEST] - mu_train) / sigma_train
        np.testing.assert_allclose(test_std_captured, expected_test_std, rtol=0, atol=1e-10)

        # Sanity: this must be recognizably different from naive full-data
        # standardization, precisely because of the injected extreme value.
        mu_full = X_raw.mean(axis=0)
        sigma_full = X_raw.std(axis=0) + 1e-8
        naive_test_std = (X_raw[N_TRAIN:N_TRAIN + N_TEST] - mu_full) / sigma_full
        assert not np.allclose(test_std_captured, naive_test_std, atol=1e-6)


class TestOldBuggyBehaviorIsCaughtByThisTest:
    """Proves the tests above are not vacuous: run the identical
    property-1 check against the reconstructed PRE-FIX implementation
    and confirm it FAILS, i.e. this test genuinely distinguishes old vs.
    new behavior rather than always passing."""

    def test_buggy_version_fails_the_same_check(self, monkeypatch):
        df_normal = _make_df(extreme_value=1.0)
        df_extreme = _make_df(extreme_value=EXTREME_VALUE)

        captured_normal = _run_compute_base_scores(monkeypatch, df_normal, _buggy_compute_base_scores)
        captured_extreme = _run_compute_base_scores(monkeypatch, df_extreme, _buggy_compute_base_scores)

        train_std_normal = captured_normal["ae_train_calls"][0]
        train_std_extreme = captured_extreme["ae_train_calls"][0]

        # With the OLD buggy code, mu/sigma are fit on the FULL dataset,
        # so an extreme value that exists ONLY in the held-out rows DOES
        # change the standardized training rows. Confirm that's what we
        # observe -- i.e. the arrays are NOT close -- proving this test
        # would have failed loudly against the pre-fix implementation.
        assert not np.allclose(train_std_normal, train_std_extreme, atol=1e-6), (
            "Expected the pre-fix implementation to leak the held-out "
            "extreme value into training statistics; if this assertion "
            "fails, the reconstructed buggy function no longer reproduces "
            "the original bug and this regression test may be vacuous."
        )
