#!/usr/bin/env python3
"""
scripts/v2_leakage_check.py
===========================
Phase 1.1 -- measures how much the v1 trace-hash cross-validation overstates
performance compared with a clean customer-grouped split.

Everything printed in reports/v2/leakage_check.md is GENERATED here (no hand-typed numbers).

Run from the repo root:
    PowerShell:  $env:PYTHONPATH="src"; python scripts/v2_leakage_check.py
    bash:        PYTHONPATH=src python3 scripts/v2_leakage_check.py

Design choices (and why)
  * Same XGBoost settings as v1 so the ONLY thing that changes is the split.
  * Fixed threshold 0.5, same as v1, so precision/recall are comparable.
  * 5 different fold seeds, mean +/- std: with only ~10 errors per run, a single
    run's difference of 0.003 F1 is noise. Averaging makes the comparison fair.
  * Three feature sets: all 28 / without HESITATION_DELTA / v1's own "ablation" set.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

import blue_team_pipeline as btp
from eval_v2.splits import (
    assert_no_group_overlap,
    customer_grouped_folds,
    fold_composition,
    group_overlap_share,
)

FAMILIES = ["ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"]
SEEDS = [11, 22, 33, 44, 55]

ALL_FEATURES = list(btp.FEATURE_COLS)
NO_HESITATION = [c for c in ALL_FEATURES if c != "HESITATION_DELTA"]
_EXCLUDE = set(btp.STRUCTURAL_SHORTCUT_FEATURES) | {
    "beneficiary_added_before_transaction", "time_from_beneficiary_add_to_transaction",
    "new_device_present", "time_device_registration_to_transaction",
    "count_beneficiary_addition", "count_device_registration", "HESITATION_DELTA",
}
ABLATION = [c for c in ALL_FEATURES if c not in _EXCLUDE]
FEATURE_SETS = {
    "all_28_features": ALL_FEATURES,
    "no_HESITATION_DELTA": NO_HESITATION,
    "v1_ablation_no_shortcut_benef_device": ABLATION,
}


def make_model() -> XGBClassifier:
    # identical to blue_team_pipeline.compute_stage_1_2_cascade
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="aucpr", random_state=42,
        n_jobs=1, tree_method="exact",
    )


def out_of_fold_scores(df: pd.DataFrame, folds, cols) -> np.ndarray:
    X = df[cols].values
    y = df["fraud"].values
    p = np.zeros(len(df))
    for tr, te in folds:
        p[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["fraud"].values
    pred = (p >= 0.5).astype(int)
    out = {
        "precision": precision_score(y, pred), "recall": recall_score(y, pred),
        "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, p),
        "false_positives": int(((pred == 1) & (y == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y == 1)).sum()),
    }
    for fam in FAMILIES:
        m = (df["attack_family"] == fam).values
        out[f"recall_{fam}"] = recall_score(y[m], pred[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "v2"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building v1 dataset (deterministic)...")
    df = btp.build_dataset(btp.CONFIG).reset_index(drop=True)

    # ---- structural facts about the data -------------------------------------
    fraud_c, legit_c = set(df[df.fraud == 1].customer_id), set(df[df.fraud == 0].customer_id)
    structure = {
        "n_rows": int(len(df)),
        "n_fraud": int(df.fraud.sum()),
        "rows_per_family": df.attack_family.value_counts().to_dict(),
        "n_unique_customers": int(df.customer_id.nunique()),
        "customers_in_both_fraud_and_legit": len(fraud_c & legit_c),
    }
    v1_folds = btp.stable_kfold_split(df, "fraud", 5, btp.CONFIG["RANDOM_STATE"])
    structure["v1_fold_overlap"] = group_overlap_share(df, v1_folds)
    g_folds = customer_grouped_folds(df, 5, 42)
    structure["grouped_fold_overlap"] = group_overlap_share(df, g_folds)
    structure["grouped_fold_composition"] = fold_composition(df, g_folds).to_dict(orient="index")

    # ---- the comparison grid -------------------------------------------------
    rows = []
    for split_name in ("v1_trace_hash", "v2_customer_grouped"):
        for fs_name, cols in FEATURE_SETS.items():
            per_seed = []
            for seed in SEEDS:
                if split_name == "v1_trace_hash":
                    folds = btp.stable_kfold_split(df, "fraud", 5, seed)
                else:
                    folds = customer_grouped_folds(df, 5, seed)
                    assert_no_group_overlap(df, folds)
                per_seed.append(metrics(df, out_of_fold_scores(df, folds, cols)))
            agg = {"split": split_name, "features": fs_name, "n_seeds": len(SEEDS)}
            for k in per_seed[0]:
                vals = np.array([r[k] for r in per_seed], dtype=float)
                agg[f"{k}_mean"], agg[f"{k}_std"] = float(vals.mean()), float(vals.std(ddof=1))
            rows.append(agg)
            print(f"  {split_name:20s} {fs_name:40s} F1={agg['f1_mean']:.3f}+/-{agg['f1_std']:.3f}")

    result = {
        "environment": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        },
        "seeds": SEEDS, "structure": structure, "grid": rows,
    }
    (out_dir / "leakage_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- markdown report (generated, never hand-edited) -------------------
    v1o = np.mean([r["share_test_rows_customer_in_train"] for r in structure["v1_fold_overlap"]])
    go = np.mean([r["share_test_rows_customer_in_train"] for r in structure["grouped_fold_overlap"]])
    L = [
        "# Phase 1.1 - customer-leakage check (generated by scripts/v2_leakage_check.py)", "",
        f"Environment: Python {result['environment']['python']}, scikit-learn {result['environment']['sklearn']}, "
        f"xgboost {result['environment']['xgboost']}.", "",
        "## Data structure", "",
        f"- Rows: {structure['n_rows']} ({structure['n_fraud']} fraud). Customers: {structure['n_unique_customers']}.",
        f"- Customers appearing as BOTH legit and fraud: {structure['customers_in_both_fraud_and_legit']}.",
        f"- Share of test rows whose customer is also in training: v1 folds {v1o:.2f}, grouped folds {go:.2f}.", "",
        f"## Results (mean +/- std over {len(SEEDS)} fold seeds, threshold 0.5)", "",
        "| split | features | precision | recall | F1 | PR-AUC | FP | FN |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['split']} | {r['features']} | {r['precision_mean']:.3f}+/-{r['precision_std']:.3f} | "
            f"{r['recall_mean']:.3f}+/-{r['recall_std']:.3f} | {r['f1_mean']:.3f}+/-{r['f1_std']:.3f} | "
            f"{r['pr_auc_mean']:.3f} | {r['false_positives_mean']:.1f} | {r['false_negatives_mean']:.1f} |"
        )
    L += ["", "## How to read this", "",
          "- If grouped F1 is close to v1 F1, customer overlap is NOT what drives the high scores.",
          "- Compare the gap with the std: a gap smaller than ~2 std is within noise.",
          "- High scores in the ablation row point to the simulator making attacks easy to separate (Phase 1.6 and Phase 2 test this)."]
    (out_dir / "leakage_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir / 'leakage_check.json'} and leakage_check.md")


if __name__ == "__main__":
    main()
