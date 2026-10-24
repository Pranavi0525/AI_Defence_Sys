"""Phase 8 tests (synthetic frames only: the real dataset is never needed or shipped)."""
import numpy as np
import pandas as pd
import pytest

from v2 import public_check as pc


def _frame(n=6000, fraud=0.03, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud).astype(int)
    X = rng.normal(size=(n, 28))
    X[y == 1, :5] += signal
    df = pd.DataFrame(X, columns=pc.V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172792, n))
    df["Amount"] = np.round(rng.lognormal(3, 1, n), 2)
    df["Class"] = y
    return df[["Time", *pc.V_COLS, "Amount", "Class"]]


def test_time_split_is_chronological_disjoint_and_complete():
    tr, va, te = pc.time_split(1000)
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(np.unique(np.concatenate([tr, va, te]))) == 1000


def test_random_split_is_disjoint_complete_and_stratified():
    y = (np.random.default_rng(1).random(5000) < 0.02).astype(int)
    parts = pc.random_split(y, seed=3)
    assert len(np.unique(np.concatenate(parts))) == 5000 and sum(len(p) for p in parts) == 5000
    rates = [y[p].mean() for p in parts]
    assert max(rates) - min(rates) < 0.005
    assert [len(p) for p in parts] == [len(p) for p in pc.time_split(5000)] or abs(len(parts[0]) - 3000) <= 2


def test_features_exclude_raw_time():
    df = _frame(200)
    X = pc.make_X(df)
    assert X.shape == (200, 29)
    assert not np.allclose(X[:, -1], df["Time"].values)


def test_ece_small_when_calibrated_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.random(20000) * 0.1
    y = (rng.random(20000) < p).astype(int)
    assert pc.ece(y, p) < 0.01
    assert pc.ece(y, np.full(20000, 0.5)) > 0.4


def test_at_budget_toy_example():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    b = pc.at_budget(y, p, 0.2)          # top 2 alerts: one fraud caught of two
    assert b["k"] == 2 and b["recall"] == pytest.approx(0.5) and b["precision"] == pytest.approx(0.5)


def test_calibrators_are_valid_probabilities_and_isotonic_is_monotone():
    rng = np.random.default_rng(2)
    p = rng.random(3000)
    y = (rng.random(3000) < p ** 2).astype(int)
    cal = pc.fit_calibrators(p, y)
    grid = np.linspace(0.01, 0.99, 50)
    for f in cal.values():
        out = f(grid)
        assert out.min() >= 0 and out.max() <= 1
    assert np.all(np.diff(cal["isotonic"](grid)) >= -1e-12)


def test_label_shuffle_control_scores_near_base_rate_while_real_labels_do_not():
    df = _frame()
    X, y = pc.make_X(df), df.Class.values
    split = pc.time_split(len(y))
    ctrl = pc.label_shuffle_control(X, y, split, seed=1, repeats=2)
    assert ctrl["mean_average_precision"] < 4 * ctrl["test_prevalence"]
    tr, _, te = split
    real = pc.metrics(y[te], pc.fit_xgb(X[tr], y[tr]).predict_proba(X[te])[:, 1])["average_precision"]
    assert real > 5 * ctrl["mean_average_precision"]


def test_evaluate_split_reports_both_models_and_all_calibrations():
    df = _frame(4000)
    X, y = pc.make_X(df), df.Class.values
    out = pc.evaluate_split(X, y, pc.time_split(len(y)), boot=20)
    assert set(out["models"]) == {"xgboost", "logreg"}
    assert set(out["models"]["xgboost"]) == {"raw", "sigmoid", "isotonic"}
    assert "average_precision_ci95" in out["models"]["xgboost"]["raw"]
    assert out["fraud"]["train"] + out["fraud"]["validation"] + out["fraud"]["test"] == int(y.sum())


def test_rolling_origin_blocks_never_train_on_the_future():
    df = _frame(4000)
    X, y = pc.make_X(df), df.Class.values
    folds = pc.rolling_origin(X, y, n_blocks=5)
    assert [f["fold"] for f in folds] == [2, 3, 4]
    assert all(f["train_rows"] < 4000 for f in folds) and folds[0]["train_rows"] < folds[-1]["train_rows"]


def test_loader_validates_schema_sorts_and_warns_on_unexpected_counts(tmp_path):
    df = _frame(500).sample(frac=1, random_state=0)
    path = tmp_path / "cc.csv"
    df.to_csv(path, index=False)
    loaded, notes = pc.load_creditcard(path)
    assert loaded.Time.is_monotonic_increasing and notes and "differ" in notes[0]
    df.drop(columns=["V3"]).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        pc.load_creditcard(path)
