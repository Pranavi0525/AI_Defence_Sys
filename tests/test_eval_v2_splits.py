"""Tests for eval_v2.splits (run: PYTHONPATH=src pytest tests/test_eval_v2_splits.py -v)."""
import numpy as np
import pandas as pd
import pytest

from eval_v2.splits import (
    assert_no_group_overlap, customer_grouped_folds, fold_composition, group_overlap_share,
)


def _toy_df(n_customers=120, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    fams = ["legitimate"] * 6 + ["ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"]
    for c in range(n_customers):
        fam = fams[c % len(fams)]
        for _ in range(rng.randint(1, 6)):
            rows.append({"customer_id": f"c{c}", "attack_family": fam, "fraud": int(fam != "legitimate")})
    return pd.DataFrame(rows)


def test_no_customer_in_both_train_and_test():
    df = _toy_df()
    folds = customer_grouped_folds(df, 5, 1)
    assert_no_group_overlap(df, folds)
    assert all(r["share_test_rows_customer_in_train"] == 0.0 for r in group_overlap_share(df, folds))


def test_folds_cover_every_row_exactly_once_in_test():
    df = _toy_df()
    folds = customer_grouped_folds(df, 5, 1)
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test.tolist()) == list(range(len(df)))


def test_every_fold_has_every_family():
    df = _toy_df()
    comp = fold_composition(df, customer_grouped_folds(df, 5, 1))
    assert (comp > 0).all().all()


def test_deterministic_for_same_seed_and_different_for_other_seed():
    df = _toy_df()
    a, b, c = customer_grouped_folds(df, 5, 7), customer_grouped_folds(df, 5, 7), customer_grouped_folds(df, 5, 8)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, c))


def test_overlap_detector_catches_a_bad_split():
    df = _toy_df()
    bad = [(np.arange(0, len(df) // 2), np.arange(len(df) // 2 - 3, len(df)))]  # deliberate overlap
    with pytest.raises(AssertionError):
        assert_no_group_overlap(df, bad)
