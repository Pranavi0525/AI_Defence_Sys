"""
eval_v2/splits.py
=================
Customer-grouped, family-stratified cross-validation folds.

WHY THIS EXISTS
---------------
The v1 folds (blue_team_pipeline.stable_kfold_split) are keyed on a hash of
trace_id. That is stable and reproducible, but it ignores WHO the trace
belongs to. A customer with several sessions can therefore land in both the
training and the test fold, so the model can partly "recognise the customer"
instead of recognising fraud. BLUE_TEAM_INTEGRATION_SPEC.md section 9 warns
about exactly this.

HOW IT WORKS
------------
StratifiedGroupKFold keeps every customer entirely inside ONE fold (the
"group"), while keeping the class mix per fold as even as it can. We stratify
on attack_family (legitimate / ATO / APP / MULE_NETWORK), not just on
fraud 0/1, so each fold contains every family.

v1 is NOT modified: results from v1 stay as frozen evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assert_no_group_overlap(df: pd.DataFrame, folds, group_col: str = "customer_id") -> None:
    """Raise AssertionError if any group (customer) appears in both train and test of any fold."""
    for k, (tr, te) in enumerate(folds):
        shared = set(df.iloc[tr][group_col]) & set(df.iloc[te][group_col])
        assert not shared, f"fold {k}: {len(shared)} {group_col} values appear in both train and test"


def customer_grouped_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    group_col: str = "customer_id",
    strata_col: str = "attack_family",
):
    """Return a list of (train_idx, test_idx) with positional indices, no customer shared across train/test."""
    df = df.reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(df, df[strata_col].values, groups=df[group_col].values))
    assert_no_group_overlap(df, folds, group_col)
    return folds


def group_overlap_share(df: pd.DataFrame, folds, group_col: str = "customer_id") -> list[dict]:
    """For each fold: what share of TEST rows belong to a customer that also appears in TRAIN?
    0.0 means a clean customer-level split; v1 folds show roughly 0.85-0.91."""
    df = df.reset_index(drop=True)
    out = []
    for k, (tr, te) in enumerate(folds):
        train_groups = set(df.loc[tr, group_col])
        seen = df.loc[te, group_col].isin(train_groups)
        out.append({
            "fold": k,
            "n_test": int(len(te)),
            "share_test_rows_customer_in_train": float(seen.mean()),
        })
    return out


def fold_composition(df: pd.DataFrame, folds, strata_col: str = "attack_family") -> pd.DataFrame:
    """Rows per family in each test fold, to confirm every fold sees every family."""
    df = df.reset_index(drop=True)
    rows = []
    for k, (_, te) in enumerate(folds):
        counts = df.loc[te, strata_col].value_counts().to_dict()
        counts["fold"] = k
        rows.append(counts)
    return pd.DataFrame(rows).set_index("fold").fillna(0).astype(int)
