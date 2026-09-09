"""
Round-1 vs Round-2 Retraining Comparison
==========================================
Blue Team closed-loop, final step

Position in the loop:

    Blue Team detector -> misses.jsonl -> hard_example_generator.py
        -> hard_examples.jsonl (19 validated: 2 ATO stage2, 17 APP stage1)
        -> THIS SCRIPT: merge, retrain, compare
        -> Round-1 vs Round-2 report

WHAT THIS SCRIPT DOES NOT DO, ON PURPOSE:
  - Does NOT modify reports/ato_corpus_raw.json or reports/app_corpus_raw.json.
    hard_examples.jsonl is loaded and merged ENTIRELY IN MEMORY; the original
    corpus files on disk are untouched, exactly as instructed.
  - Does NOT change FEATURE_COLS, the XGBoost hyperparameters, the 5-fold
    StratifiedKFold procedure, the Stage-1 rule thresholds, or the 0.5
    decision threshold between Round 1 and Round 2. The ONLY difference
    between the two rounds is 19 additional training rows. This is
    deliberate: the point of this comparison is to isolate the effect of
    the hard examples, not to also quietly retune the model while doing so.
    If methodology changed too, a numbers improvement would tell us nothing
    about whether the hard-example loop itself works.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 retrain_round2.py

Reads:  blue_team_output/hard_examples.jsonl (from hard_example_generator.py)
        reports/ato_corpus_raw.json, reports/app_corpus_raw.json (READ ONLY)
Writes: blue_team_output/round1_vs_round2_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from blue_team_pipeline import (
    CONFIG, FEATURE_COLS,
    load_attack_corpus, build_legitimate_traces, extract_features,
    add_hesitation_delta,
    cross_validated_evaluate, EvaluationHarness, block,
)

REPO_ROOT = Path(__file__).parent
OUT_DIR = REPO_ROOT / "blue_team_output"
HARD_EXAMPLES_PATH = OUT_DIR / "hard_examples.jsonl"


# ---------------------------------------------------------------------------
# Step 1 -- Load hard_examples.jsonl and convert to the same record shape
# used everywhere else in the pipeline (trace_id, customer_id, events,
# observation_window, fraud, attack_family, attack_difficulty). The
# generation_metadata field (source miss, seed, validation scores) is
# deliberately dropped here -- it must never reach feature extraction or
# the model, it was only ever for our own audit trail.
# ---------------------------------------------------------------------------
def load_hard_examples(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{path} not found or empty. Run hard_example_generator.py first."
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h = json.loads(line)
            ot = h["observable_trace"]
            gt = h["ground_truth"]
            records.append({
                "trace_id": ot["trace_id"],
                "customer_id": ot["customer_id"],
                "events": ot["events"],
                "observation_window": ot["observation_window"],
                "fraud": 1,
                "attack_family": gt["attack_family"],
                "attack_difficulty": gt["attack_difficulty"],
            })
    return records


# ---------------------------------------------------------------------------
# Step 2 -- Build Round-1 (original corpus) and Round-2 (original + hard
# examples) datasets, using the EXACT SAME feature extraction as the main
# pipeline (imported directly, not reimplemented).
# ---------------------------------------------------------------------------
def build_round_datasets(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("Loading original Red Team corpora (read-only)...")
    ato_records = load_attack_corpus(cfg["REPO_ROOT"] / cfg["ATO_CORPUS_PATH"], "ATO")
    app_records = load_attack_corpus(cfg["REPO_ROOT"] / cfg["APP_CORPUS_PATH"], "APP")

    print("Building legitimate population (identical to Round 1, same seed)...")
    legit_records = build_legitimate_traces(cfg)

    round1_records = ato_records + app_records + legit_records
    print(f"Round 1: {len(round1_records)} traces "
          f"({len(ato_records)} ATO + {len(app_records)} APP + {len(legit_records)} legit)")

    print("\nLoading validated hard examples (in-memory merge only, "
          "original corpus files NOT touched)...")
    hard_records = load_hard_examples(HARD_EXAMPLES_PATH)
    hard_ato = [r for r in hard_records if r["attack_family"] == "ACCOUNT_TAKEOVER"]
    hard_app = [r for r in hard_records if r["attack_family"] == "AUTHORIZED_PUSH_PAYMENT"]
    print(f"  {len(hard_records)} hard examples loaded "
          f"({len(hard_ato)} ATO, {len(hard_app)} APP)")

    # NOTE: hard_example_generation_report.json validates ATO and APP hard
    # examples against two DIFFERENT bars: ATO examples are validated as true
    # Stage-2 (XGBoost) blind spots (xgboost_predicted_probability < 0.5),
    # while APP examples are validated only as Stage-1 rule blind spots
    # (stage1_escalation_rate == 0%). The APP examples are deliberately short,
    # sparse traces built to test Stage-1 rules -- they were never validated
    # to carry the richer behavioral signal Stage-2's features rely on, so
    # they must NOT be fed into Stage-2 (XGBoost) training/CV as ground-truth
    # fraud rows. Only the ATO-validated hard examples are merged here.
    round2_records = round1_records + hard_ato
    print(f"Round 2: {len(round2_records)} traces "
          f"({len(round1_records)} original + {len(hard_ato)} Stage-2-validated ATO hard examples; "
          f"{len(hard_app)} Stage-1-only APP hard examples excluded from Stage-2 retraining)")

    def to_feature_df(records: list[dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            feats = extract_features(rec)
            feats["fraud"] = rec["fraud"]
            feats["attack_family"] = rec["attack_family"]
            feats["attack_difficulty"] = rec["attack_difficulty"]
            feats["customer_id"] = rec["customer_id"]
            rows.append(feats)

        # HESITATION_DELTA is derived from the per-customer transaction
        # pacing baseline, so customer_id must be retained until this
        # calculation is complete. This mirrors the main pipeline.
        df = pd.DataFrame(rows)
        df = add_hesitation_delta(df)
        return df

    print("\nExtracting features for both rounds (same extract_features() as the main pipeline)...")
    df_round1 = to_feature_df(round1_records)
    df_round2 = to_feature_df(round2_records)

    meta = {
        "round1_n": len(round1_records),
        "round2_n": len(round2_records),
        "hard_examples_loaded_total": len(hard_records),
        "hard_examples_added_to_stage2": len(hard_ato),
        "hard_examples_ato": len(hard_ato),
        "hard_examples_app": len(hard_app),
        "hard_examples_app_excluded_reason": (
            "APP hard examples are validated only as Stage-1 rule blind spots "
            "(stage1_escalation_rate == 0%), not as Stage-2 XGBoost blind spots. "
            "They are excluded from Stage-2 training/CV to avoid evaluating "
            "Stage-2 on rows it was never meant to classify."
        ),
    }
    return df_round1, df_round2, meta


# ---------------------------------------------------------------------------
# Step 3 -- Run the UNCHANGED evaluation methodology on both datasets:
# same single-stage 5-fold CV, same cascade EvaluationHarness. Nothing
# about the procedure differs between rounds -- only df differs.
# ---------------------------------------------------------------------------
def evaluate_round(df: pd.DataFrame, cfg: dict, label: str) -> dict:
    print(f"\n--- Evaluating {label} (single-stage, 5-fold CV) ---")
    single_stage_results, _, _ = cross_validated_evaluate(df, cfg, FEATURE_COLS, n_splits=5)

    print(f"--- Evaluating {label} (cascade: Stage 1 rules -> Stage 2 XGBoost via EvaluationHarness) ---")
    harness = EvaluationHarness(FEATURE_COLS, cfg)
    cascade_results, _, _, _ = harness.run(df, n_splits=5)

    return {"single_stage": single_stage_results, "cascade": cascade_results}


# ---------------------------------------------------------------------------
# Step 4 -- Round-1 vs Round-2 comparison, explicit numbers, both overall
# and per-family, for both single-stage and cascade.
# ---------------------------------------------------------------------------
def compare(r1: dict, r2: dict, view: str) -> dict:
    """view = 'single_stage' or 'cascade'"""
    o1, o2 = r1[view]["overall"], r2[view]["overall"]
    cm1, cm2 = o1["confusion_matrix"], o2["confusion_matrix"]  # [[TN,FP],[FN,TP]]

    comparison = {
        "overall": {
            "precision": {"round1": o1["precision"], "round2": o2["precision"], "delta": round(o2["precision"] - o1["precision"], 4)},
            "recall": {"round1": o1["recall"], "round2": o2["recall"], "delta": round(o2["recall"] - o1["recall"], 4)},
            "f1": {"round1": o1["f1"], "round2": o2["f1"], "delta": round(o2["f1"] - o1["f1"], 4)},
            "roc_auc": {"round1": o1["roc_auc"], "round2": o2["roc_auc"], "delta": round(o2["roc_auc"] - o1["roc_auc"], 4)},
            "false_positives_count": {"round1": cm1[0][1], "round2": cm2[0][1], "delta": cm2[0][1] - cm1[0][1]},
            "fraud_misses_count_false_negatives": {"round1": cm1[1][0], "round2": cm2[1][0], "delta": cm2[1][0] - cm1[1][0]},
            "n_round1": o1["n"], "n_round2": o2["n"],
        },
        "by_family": {},
    }
    for fam in set(list(r1[view]["by_family"].keys()) + list(r2[view]["by_family"].keys())):
        f1_ = r1[view]["by_family"].get(fam, {})
        f2_ = r2[view]["by_family"].get(fam, {})
        comparison["by_family"][fam] = {
            "precision": {"round1": f1_.get("precision"), "round2": f2_.get("precision")},
            "recall": {"round1": f1_.get("recall"), "round2": f2_.get("recall")},
            "f1": {"round1": f1_.get("f1"), "round2": f2_.get("f1")},
            "n_round1": f1_.get("n"), "n_round2": f2_.get("n"),
        }
    return comparison


def main():
    cfg = CONFIG
    OUT_DIR.mkdir(exist_ok=True)

    df_r1, df_r2, meta = build_round_datasets(cfg)
    print(f"\n{json.dumps(meta, indent=2)}")

    r1_results = evaluate_round(df_r1, cfg, "ROUND 1 (original corpus only)")
    r2_results = evaluate_round(df_r2, cfg, f"ROUND 2 (original + {meta['hard_examples_added_to_stage2']} Stage-2-validated ATO hard examples)")

    single_stage_comparison = compare(r1_results, r2_results, "single_stage")
    cascade_comparison = compare(r1_results, r2_results, "cascade")

    print("\n" + "=" * 70)
    print("ROUND 1 vs ROUND 2 -- CASCADE (Stage 1 rules -> Stage 2 XGBoost)")
    print("=" * 70)
    print(json.dumps(cascade_comparison["overall"], indent=2))
    print("\n--- by family ---")
    print(json.dumps(cascade_comparison["by_family"], indent=2))

    print("\n" + "=" * 70)
    print("ROUND 1 vs ROUND 2 -- SINGLE-STAGE XGBOOST (no cascade)")
    print("=" * 70)
    print(json.dumps(single_stage_comparison["overall"], indent=2))
    print("\n--- by family ---")
    print(json.dumps(single_stage_comparison["by_family"], indent=2))

    report = {
        "meta": meta,
        "methodology_note": "Identical FEATURE_COLS, XGBoost hyperparameters, "
                             "5-fold StratifiedKFold procedure, Stage-1 rule "
                             "thresholds, and 0.5 decision threshold used for "
                             "both rounds. Only the training data differs "
                             f"({meta['hard_examples_added_to_stage2']} Stage-2-validated ATO hard "
                             f"examples added in Round 2; {meta['hard_examples_app']} Stage-1-only "
                             "APP hard examples were deliberately excluded from Stage-2 "
                             "retraining/CV -- see hard_examples_app_excluded_reason). "
                             "Original corpus files on disk were not modified.",
        "cascade_comparison": cascade_comparison,
        "single_stage_comparison": single_stage_comparison,
        "round1_full_results": r1_results,
        "round2_full_results": r2_results,
    }
    with open(OUT_DIR / "round1_vs_round2_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {OUT_DIR / 'round1_vs_round2_report.json'}")


if __name__ == "__main__":
    main()
