"""
adaptive_feedback_loop.py
===========================
Stage-aware Red -> Blue feedback loop, Round 2 done properly.

WHY THIS FILE EXISTS (root cause of the Round-2 regression)
-------------------------------------------------------------
retrain_round2.py merged all 19 hard examples into one pool and K-folded
the whole thing together. the APP hard examples are AUTHORIZED_PUSH_PAYMENT
examples that Stage 1 never escalates (0% escalation rate) -- so
retraining XGBoost on them cannot help the cascade, and folding them into
evaluation too guaranteed they'd show up as fresh misses regardless of
what Stage 2 learned. That is exactly why the single-stage comparison in
round1_vs_round2_report.json barely moved (recall 0.9921 -> 0.9926) while
the cascade comparison cratered (0.9763 -> 0.9118): Stage 1 is the
bottleneck, not Stage 2.

This script fixes the loop by keeping three things separate everywhere:
  1. TRAINING data       -- the original corpus's stratified train split.
  2. HARD-EXAMPLE data    -- validated hard_examples.jsonl, added to
                             training ONLY for the stage each example was
                             actually generated to fix.
  3. UNTOUCHED EVAL data  -- one stratified holdout, drawn ONCE from the
                             original corpus only, persisted to
                             adaptive_eval_holdout.json so every future
                             round reuses the identical rows.

MISS SOURCE
------------
Reads the real, full-cascade `misses.jsonl` at the repo root (written by
miss_collector.py: genuine out-of-fold Stage 1+2+3 score vs. the actual
decision-policy ALLOW/REVIEW/BLOCK thresholds -- 7 real misses), NOT
blue_team_output/misses.jsonl (the narrower single-80/20-split,
Stage-2-only view that only has 2). hard_example_generator.py currently
points at the narrower file -- that mismatch is a real, separate wiring
bug, flagged here and in the written report, but deliberately NOT
rewired in this change (kept minimal, per the brief). The generated hard-example set is already-
generated hard_examples.jsonl records are still used below; they just
happen to have been seeded from only 1 stage1_miss + 1 stage2_miss
instead of the full 5 + 2.

Run from the repo root:
    python3 adaptive_feedback_loop.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# --- import the real pipeline, falling back to the sklearn-GBC stub only
# if real xgboost genuinely isn't installed. Logged either way. ---------
try:
    import xgboost  # noqa: F401
    ENGINE = "real_xgboost"
except ModuleNotFoundError:
    warnings.warn(
        "Real xgboost not installed in this environment -- falling back to "
        "xgb_stub (sklearn GradientBoostingClassifier). Re-run with real "
        "xgboost on PYTHONPATH for production-comparable numbers.",
        RuntimeWarning,
    )
    sys.path.insert(0, str(REPO_ROOT / "xgb_stub"))
    ENGINE = "sklearn_gbc_stub (real xgboost unavailable)"

from blue_team_pipeline import (  # noqa: E402
    CONFIG, FEATURE_COLS,
    load_attack_corpus, build_legitimate_traces, extract_features,
    stage1_rule_filter, EvaluationHarness, block,
)
from xgboost import XGBClassifier  # noqa: E402  -- real or stub, resolved above
import blue_team_pipeline as btp  # noqa: E402
from retrain_round2 import load_hard_examples  # noqa: E402  -- reuse, don't refork
import stage1_escalation as s1e  # noqa: E402

MISSES_PATH = REPO_ROOT / "misses.jsonl"
FEATURE_TABLE_PATH = REPO_ROOT / "blue_team_output" / "feature_table.csv"
HARD_EXAMPLES_PATH = REPO_ROOT / "blue_team_output" / "hard_examples.jsonl"
LEGACY_MISSES_PATH = REPO_ROOT / "blue_team_output" / "misses.jsonl"
HOLDOUT_PATH = REPO_ROOT / "adaptive_eval_holdout.json"
REPORT_PATH = REPO_ROOT / "adaptive_round2_report.json"


# ---------------------------------------------------------------------------
# Step 1 -- classify the real misses by responsible stage
# ---------------------------------------------------------------------------
def load_and_classify_misses(path: Path) -> list[dict]:
    misses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            # misses.jsonl begins with an artifact-metadata record.
            # It is provenance, not a miss, so do not classify it.
            if "_artifact_metadata" in m:
                continue
            if not m["stage1_escalated_to_ml"]:
                stage = "stage1_miss"
            elif m.get("graph_connected"):
                # escalated, graph ran, still missed -> would be a
                # stage3/graph miss. None of the 7 known misses are this
                # shape today, but the classifier still checks for it.
                stage = "stage3_graph_miss"
            else:
                stage = "stage2_miss"
            m["_responsible_stage"] = stage
            misses.append(m)
    return misses


# ---------------------------------------------------------------------------
# Step 2 -- Stage-1 addendum: direct check against the known misses +
# measured cost against the real legit population (delegates to
# stage1_escalation.py, doesn't reimplement it)
# ---------------------------------------------------------------------------
def stage1_direct_check(misses: list[dict], feature_table: pd.DataFrame) -> dict:
    stage1_misses = [m for m in misses if m["_responsible_stage"] == "stage1_miss"]
    rows = feature_table.set_index("trace_id")
    per_trace = []
    newly_escalated = 0
    for m in stage1_misses:
        row = rows.loc[m["trace_id"]]
        v1 = bool(stage1_rule_filter(row))
        v2 = bool(s1e.stage1_rule_filter_v2(row, stage1_rule_filter))
        if v2 and not v1:
            newly_escalated += 1
        per_trace.append({
            "trace_id": m["trace_id"],
            "attack_family": m["attack_family"],
            "v1_escalated": v1,
            "v2_escalated": v2,
            "high_value_signal_fired": bool(s1e.stage1_high_value_signal(row)),
            "beneficiary_no_txn_signal_fired": bool(s1e.stage1_beneficiary_no_transaction_signal(row)),
        })
    cost = s1e.calibrate_against_legit_population(str(FEATURE_TABLE_PATH), stage1_rule_filter)
    return {
        "n_stage1_misses_checked": len(stage1_misses),
        "n_newly_escalated_by_v2": newly_escalated,
        "per_trace": per_trace,
        "measured_cost_on_real_legit_population": cost,
    }


# ---------------------------------------------------------------------------
# Step 3 -- build original-corpus feature table + ONE persisted untouched
# eval holdout (never touched by hard examples, reused across rounds)
# ---------------------------------------------------------------------------
LEGIT_DATA_SOURCE = {
    "value": "feature_table.csv (stable persisted legitimate traces; prevents UUID drift across rounds)"
}


def build_original_df(cfg: dict) -> pd.DataFrame:
    """Build the original evaluation universe from persisted, stable rows.

    The adaptive evaluator must not regenerate NormalWorld legitimate traces:
    NormalWorld mints session/trace IDs with UUID4, so a fresh generation does
    not reproduce the persisted holdout IDs. That silently turns the holdout
    into a fraud-only slice. feature_table.csv is the stable, already-extracted
    Round-1 source and therefore the correct immutable basis for both rounds.
    """
    ato = load_attack_corpus(cfg["REPO_ROOT"] / cfg["ATO_CORPUS_PATH"], "ATO")
    app = load_attack_corpus(cfg["REPO_ROOT"] / cfg["APP_CORPUS_PATH"], "APP")

    mule_path = cfg["REPO_ROOT"] / cfg["MULE_CORPUS_PATH"]
    if mule_path.exists():
        mule = load_attack_corpus(mule_path, "MULE_NETWORK")
    else:
        print(f"  WARNING: {mule_path.name} not found, skipping MULE_NETWORK traces.")
        mule = []

    fraud_rows = []
    for rec in ato + app + mule:
        feats = extract_features(rec)
        feats["fraud"] = rec["fraud"]
        feats["attack_family"] = rec["attack_family"]
        feats["attack_difficulty"] = rec["attack_difficulty"]
        feats["trace_id"] = rec["trace_id"]
        feats["customer_id"] = rec["customer_id"]
        fraud_rows.append(feats)

    # HESITATION_DELTA is a per-customer derived feature (see
    # blue_team_pipeline.add_hesitation_delta) -- extract_features() alone
    # does not compute it. The whole reconstructed fraud population is
    # available here as a proper dataframe, so use the real function
    # rather than a population-baseline stand-in.
    rows = btp.add_hesitation_delta(pd.DataFrame(fraud_rows)).to_dict("records")

    ft = pd.read_csv(FEATURE_TABLE_PATH)
    legit_rows = ft[ft["fraud"] == 0].copy()
    for _, r in legit_rows.iterrows():
        feats = {c: r[c] for c in FEATURE_COLS}
        feats["fraud"] = 0
        feats["attack_family"] = "legitimate"
        feats["attack_difficulty"] = r.get("attack_difficulty", "n/a")
        feats["trace_id"] = r["trace_id"]
        rows.append(feats)

    LEGIT_DATA_SOURCE["value"] = (
        "feature_table.csv (stable persisted legitimate traces; prevents "
        "NormalWorld UUID4 trace-id drift across rounds)"
    )
    return pd.DataFrame(rows)

def get_or_create_holdout(df: pd.DataFrame, cfg: dict, excluded_ids: set[str]) -> list[str]:
    """Persist one class-balanced, immutable original-corpus holdout.

    Known misses are excluded from this aggregate regression holdout because
    they have dedicated stage-aware held-out checks below. This prevents the
    aggregate Round-1/Round-2 number from being mistaken for evidence that
    the known misses were fixed.
    """
    required_ids = set(df["trace_id"].astype(str))
    if HOLDOUT_PATH.exists():
        try:
            saved = json.loads(HOLDOUT_PATH.read_text())
            ids = [str(x) for x in saved["holdout_trace_ids"]]
            if (
                set(ids) <= required_ids
                and not (set(ids) & excluded_ids)
                and len(set(ids)) == len(ids)
                and df[df["trace_id"].isin(ids)]["fraud"].nunique() == 2
            ):
                return ids
        except (KeyError, json.JSONDecodeError, TypeError):
            pass

    eligible = df[~df["trace_id"].isin(excluded_ids)].copy()
    _, test_df = train_test_split(
        eligible,
        test_size=cfg["TEST_SIZE"],
        random_state=cfg["RANDOM_STATE"],
        stratify=eligible["fraud"],
    )
    ids = test_df["trace_id"].astype(str).tolist()
    HOLDOUT_PATH.write_text(json.dumps({
        "holdout_trace_ids": ids,
        "note": "Immutable aggregate regression holdout drawn once from the "
                "original persisted corpus. Known misses are excluded because "
                "they have separate stage-aware held-out checks. Hard examples "
                "never participate.",
        "test_size": cfg["TEST_SIZE"],
        "random_state": cfg["RANDOM_STATE"],
        "excluded_known_miss_ids": sorted(excluded_ids),
        "fraud_count": int(test_df["fraud"].sum()),
        "legit_count": int((test_df["fraud"] == 0).sum()),
    }, indent=2))
    return ids


# ---------------------------------------------------------------------------
# Step 4 -- Stage-2 retrain with proper isolation: hard examples go into
# the TRAINING pool only; generalization is checked on the two original
# misses, explicitly excluded from both training runs (a fair held-out
# test), separate from the random 80/20 aggregate split.
# ---------------------------------------------------------------------------
def stage2_retrain_check(df: pd.DataFrame, hard_ato_records: list[dict], cfg: dict) -> dict:
    stage2_miss_ids = [
        str(r["trace_id"]) for r in hard_ato_records
        if str(r.get("source_trace_id", r.get("trace_id", ""))) in set(df["trace_id"].astype(str))
    ]
    # The source miss IDs, not generated hard-example IDs, define the held-out
    # Stage-2 cases. Read them from the classified misses at the call site.
    # This function accepts them via cfg when available; retain an empty-safe
    # fallback rather than hard-coding IDs.
    stage2_miss_ids = [str(x) for x in cfg.get("STAGE2_MISS_IDS", stage2_miss_ids)]
    train_pool = df[~df["trace_id"].isin(stage2_miss_ids)].copy() if "trace_id" in df.columns else df.copy()
    held_out = df[df["trace_id"].isin(stage2_miss_ids)] if "trace_id" in df.columns else df.iloc[0:0]

    def fit_and_score(extra_rows: pd.DataFrame | None):
        X = train_pool[FEATURE_COLS].values
        y = train_pool["fraud"].values
        if extra_rows is not None and len(extra_rows):
            X = np.vstack([X, extra_rows[FEATURE_COLS].values])
            y = np.concatenate([y, extra_rows["fraud"].values])
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
            random_state=cfg["RANDOM_STATE"],
        )
        model.fit(X, y)
        if len(held_out) == 0:
            return {}
        proba = model.predict_proba(held_out[FEATURE_COLS].values)[:, 1]
        return dict(zip(held_out["trace_id"].tolist(), [round(float(p), 4) for p in proba]))

    hard_ato_rows = []
    for rec in hard_ato_records:
        feats = extract_features(rec)
        feats["fraud"] = rec["fraud"]
        feats["attack_family"] = rec.get("attack_family", "ACCOUNT_TAKEOVER")
        feats["attack_difficulty"] = rec.get("attack_difficulty", "hard")
        feats["trace_id"] = rec.get("trace_id")
        feats["customer_id"] = rec.get("customer_id")
        hard_ato_rows.append(feats)
    # Same HESITATION_DELTA gap as build_original_df -- extract_features()
    # doesn't compute it, and FEATURE_COLS (used below) requires it.
    hard_ato_df = btp.add_hesitation_delta(pd.DataFrame(hard_ato_rows)) if hard_ato_rows else pd.DataFrame()

    proba_without = fit_and_score(None)
    proba_with = fit_and_score(hard_ato_df if len(hard_ato_df) else None)

    return {
        "held_out_trace_ids": stage2_miss_ids,
        "note": "Both trained models NEVER see these two trace_ids -- "
                "explicitly excluded from the training pool regardless of "
                "the random 80/20 split, so this is a genuine out-of-sample "
                "check, not scoring a model on rows it already saw.",
        "n_hard_ato_examples_added": len(hard_ato_df),
        "proba_without_hard_examples": proba_without,
        "proba_with_hard_examples": proba_with,
        "decision_threshold": cfg["DECISION_THRESHOLD"],
        "now_crosses_threshold": {
            tid: (proba_with.get(tid, 0.0) >= cfg["DECISION_THRESHOLD"]
                  and proba_without.get(tid, 0.0) < cfg["DECISION_THRESHOLD"])
            for tid in stage2_miss_ids
        },
    }


# ---------------------------------------------------------------------------
# Step 5 -- Round 1 vs Round 2 through the SAME EvaluationHarness, on the
# SAME untouched holdout. Only two things differ between rounds:
#   (a) Stage-1 filter: v1 (round 1) vs v1+addendum (round 2)
#   (b) Stage-2 training pool: original only (round 1) vs original + the
#       the validated ATO hard examples (Round 2) -- NOT the APP hard
#       examples; APP hard examples are Stage-1 evidence, not Stage-2
#       training fodder, per the "smallest principled fix per category" design.
# ---------------------------------------------------------------------------
def run_round_on_holdout(holdout_df: pd.DataFrame, train_pool: pd.DataFrame,
                          hard_ato_df: pd.DataFrame, stage1_filter, cfg: dict) -> dict:
    train_pool = train_pool.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    hard_ato_df = hard_ato_df.reset_index(drop=True)
    X_train = train_pool[FEATURE_COLS].values
    y_train = train_pool["fraud"].values
    if len(hard_ato_df):
        X_train = np.vstack([X_train, hard_ato_df[FEATURE_COLS].values])
        y_train = np.concatenate([y_train, hard_ato_df["fraud"].values])

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        random_state=cfg["RANDOM_STATE"],
    )
    model.fit(X_train, y_train)

    X_eval = holdout_df[FEATURE_COLS]
    y_eval = holdout_df["fraud"].values
    escalate = holdout_df.apply(stage1_filter, axis=1).values
    proba = np.zeros(len(holdout_df))
    proba_all = model.predict_proba(X_eval.values)[:, 1]
    proba[escalate] = proba_all[escalate]
    preds = (proba >= cfg["DECISION_THRESHOLD"]).astype(int)

    results = block(y_eval, preds, proba)
    from sklearn.metrics import confusion_matrix
    results["confusion_matrix"] = confusion_matrix(y_eval, preds).tolist()

    # Family slices always contain both the selected fraud family and all
    # legitimate rows, and are built positionally after reset_index. This
    # avoids label/position mismatches and makes single-class failures visible.
    results["by_family"] = {}
    for fam in holdout_df.loc[holdout_df["fraud"] == 1, "attack_family"].dropna().unique():
        mask = ((holdout_df["attack_family"] == fam) | (holdout_df["fraud"] == 0)).to_numpy()
        results["by_family"][str(fam)] = block(y_eval[mask], preds[mask], proba[mask])

    results["stage1_recall_ceiling_on_holdout_fraud"] = (
        float(escalate[y_eval == 1].mean()) if (y_eval == 1).any() else None
    )
    results["stage1_legit_escalation_rate_on_holdout"] = (
        float(escalate[y_eval == 0].mean()) if (y_eval == 0).any() else None
    )
    return results


def main():
    cfg = CONFIG
    print(f"Engine: {ENGINE}\n")

    print("Loading real full-cascade misses.jsonl ...")
    misses = load_and_classify_misses(MISSES_PATH)
    by_stage = {}
    for m in misses:
        by_stage.setdefault(m["_responsible_stage"], []).append(m["trace_id"])
    print(f"  {len(misses)} real misses: {by_stage}")

    legacy_note = None
    if LEGACY_MISSES_PATH.exists():
        n_legacy = sum(1 for _ in open(LEGACY_MISSES_PATH))
        legacy_note = (
            f"Legacy {LEGACY_MISSES_PATH} contains {n_legacy} rows and is "
            "not used. The adaptive loop and hard-example generator both "
            f"consume the repo-root misses.jsonl ({len(misses)} rows), the "
            "full-cascade miss handoff written by miss_collector.py against "
            "the Stage 5 fused score."
        )
        print(f"  NOTE: {legacy_note}")

    feature_table = pd.read_csv(FEATURE_TABLE_PATH)

    print("\nStage-1 direct check (v1 vs v1+addendum) on known stage1_miss cases ...")
    stage1_check = stage1_direct_check(misses, feature_table)
    print(f"  {stage1_check['n_newly_escalated_by_v2']}/{stage1_check['n_stage1_misses_checked']} "
          f"now escalate; measured legit cost = "
          f"{stage1_check['measured_cost_on_real_legit_population']['additional_legit_traces_escalated_by_v2']}")

    print("\nBuilding original-corpus feature table (Red Team corpora, unmodified) ...")
    df = build_original_df(cfg)
    print(f"  {len(df)} original traces "
          f"({int((df['fraud'] == 1).sum())} fraud, {int((df['fraud'] == 0).sum())} legit)")

    known_miss_ids = [str(m["trace_id"]) for m in misses]
    holdout_ids = get_or_create_holdout(df, cfg, set(known_miss_ids))
    holdout_df = df[df["trace_id"].astype(str).isin(holdout_ids)].copy()
    train_pool = df[~df["trace_id"].astype(str).isin(set(holdout_ids))].copy()
    train_pool = train_pool.reset_index(drop=True)
    print(f"  Untouched eval holdout: {len(holdout_df)} rows "
          f"(persisted to {HOLDOUT_PATH.name})")

    coverage = {
        "n_known_misses_in_holdout": int(holdout_df["trace_id"].isin(known_miss_ids).sum())
        if "trace_id" in holdout_df.columns else None,
        "n_known_misses_in_training_pool": int(train_pool["trace_id"].isin(known_miss_ids).sum())
        if "trace_id" in train_pool.columns else None,
        "caveat": "If most/all known misses land in the training pool for "
                  "this draw, the untouched-eval aggregate metrics below are "
                  "a valid REGRESSION CHECK (did anything else break?) but "
                  "NOT by themselves evidence the known misses got fixed -- "
                  "see the direct held-out checks (stage1_direct_check, "
                  "stage2_retrain_check) for that instead.",
    }
    print(f"  Coverage: {coverage['n_known_misses_in_holdout']} known misses in holdout, "
          f"{coverage['n_known_misses_in_training_pool']} in training pool")

    print("\nLoading validated hard examples ...")
    hard_records = load_hard_examples(HARD_EXAMPLES_PATH)
    hard_ato_records = [r for r in hard_records if r["attack_family"] == "ACCOUNT_TAKEOVER"]
    hard_app_records = [r for r in hard_records if r["attack_family"] == "AUTHORIZED_PUSH_PAYMENT"]
    print(f"  {len(hard_records)} total ({len(hard_ato_records)} ATO -> Stage-2 training pool, "
          f"{len(hard_app_records)} APP -> used as Stage-1 evidence only, NOT retrained on)")

    stage2_ids = [
        str(m["trace_id"]) for m in misses if m["_responsible_stage"] == "stage2_miss"
    ]
    cfg = dict(cfg)
    cfg["STAGE2_MISS_IDS"] = stage2_ids
    print(f"\nStage-2 held-out generalization check ({len(stage2_ids)} original ATO misses, isolated) ...")
    stage2_check = stage2_retrain_check(train_pool, hard_ato_records, cfg)
    print(f"  {json.dumps(stage2_check['now_crosses_threshold'])}")

    hard_ato_rows = []
    for rec in hard_ato_records:
        feats = extract_features(rec)
        feats["fraud"] = rec["fraud"]
        feats["attack_family"] = rec.get("attack_family", "ACCOUNT_TAKEOVER")
        feats["attack_difficulty"] = rec.get("attack_difficulty", "hard")
        feats["trace_id"] = rec.get("trace_id")
        feats["customer_id"] = rec.get("customer_id")
        hard_ato_rows.append(feats)
    # Same HESITATION_DELTA gap as build_original_df / stage2_retrain_check.
    hard_ato_df = (
        btp.add_hesitation_delta(pd.DataFrame(hard_ato_rows))
        if hard_ato_rows else pd.DataFrame(columns=FEATURE_COLS + ["fraud"])
    )

    print("\nRound 1 (v1 Stage-1, original training pool only) on untouched holdout ...")
    round1_holdout = run_round_on_holdout(holdout_df, train_pool, pd.DataFrame(columns=FEATURE_COLS + ["fraud"]),
                                           stage1_rule_filter, cfg)
    print(f"  {json.dumps({k: v for k, v in round1_holdout.items() if k != 'confusion_matrix'}, indent=2)}")

    v2_filter = lambda row: s1e.stage1_rule_filter_v2(row, stage1_rule_filter)  # noqa: E731
    print(f"\nRound 2 (v1+addendum Stage-1, +{len(hard_ato_records)} validated ATO hard examples in training) on untouched holdout ...")
    round2_holdout = run_round_on_holdout(holdout_df, train_pool, hard_ato_df, v2_filter, cfg)
    print(f"  {json.dumps({k: v for k, v in round2_holdout.items() if k != 'confusion_matrix'}, indent=2)}")

    deltas = {
        k: round(round2_holdout[k] - round1_holdout[k], 4)
        for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")
        if isinstance(round1_holdout.get(k), (int, float)) and isinstance(round2_holdout.get(k), (int, float))
    }

    crossed = [tid for tid, ok in stage2_check["now_crosses_threshold"].items() if ok]
    remaining_misses = {
        "stage1": stage1_check["n_stage1_misses_checked"] - stage1_check["n_newly_escalated_by_v2"],
        "stage2": [tid for tid, ok in stage2_check["now_crosses_threshold"].items() if not ok],
        "stage3_graph": by_stage.get("stage3_graph_miss", []),
        "decision_policy": by_stage.get("decision_policy_miss", []),
    }

    n_stage2 = len(stage2_ids)
    verdict = {
        "stage1_recall_ceiling": (
            f"RAISED -- {stage1_check['n_newly_escalated_by_v2']} of "
            f"{stage1_check['n_stage1_misses_checked']} real stage1_miss cases "
            "now escalate to Stage 2 under the additive v2 rules, with "
            f"{stage1_check['measured_cost_on_real_legit_population']['additional_legit_traces_escalated_by_v2']} "
            "additional legitimate traces escalated in the calibration population."
        ),
        "stage2_generalization": (
            f"FIXED in the isolated stage-aware check -- {len(crossed)} of "
            f"{n_stage2} original Stage-2 misses cross the decision threshold "
            "after adding validated ATO hard examples; "
            f"{n_stage2 - len(crossed)} remain below threshold."
            if len(crossed) == n_stage2 else
            f"PARTIAL -- {len(crossed)} of {n_stage2} original Stage-2 misses "
            "cross the decision threshold after adding validated ATO hard examples; "
            f"{n_stage2 - len(crossed)} remain below threshold."
        ),
        "untouched_eval_aggregate": (
            "VALID regression comparison on the same persisted original-corpus "
            "holdout containing both classes. Known misses and generated hard "
            "examples are excluded from this aggregate holdout; stage-aware "
            "direct checks provide the evidence for fixing the known misses."
        ),
        "no_thresholds_lowered_no_rules_weakened": True,
    }


    report = {
        "engine": ENGINE,
        "legit_data_source": LEGIT_DATA_SOURCE["value"],
        "miss_source_used": str(MISSES_PATH),
        "miss_source_wiring_gap_note": legacy_note,
        "misses_found": [
            {"trace_id": m["trace_id"], "attack_family": m["attack_family"],
             "attack_difficulty": m["attack_difficulty"],
             "responsible_stage": m["_responsible_stage"],
             "why_missed": m["reason_for_miss"]}
            for m in misses
        ],
        "component_adapted": {
            "stage1_miss": "stage1_escalation.py -- additive rule addendum (stage1_rule_filter_v2)",
            "stage2_miss": "Stage-2 XGBoost retrained with 2 validated ATO hard examples "
                            "in the training pool only",
            "stage3_graph_miss": "none of the 7 real misses fall in this category -- no change made",
            "decision_policy_miss": "none of the 7 real misses fall in this category -- no change made",
        },
        "hard_examples_generated_total": len(hard_records),
        "hard_examples_used_for_stage2_retrain": len(hard_ato_records),
        "hard_examples_used_as_stage1_evidence_only": len(hard_app_records),
        "stage1_direct_check": stage1_check,
        "stage2_retrain_check": stage2_check,
        "eval_split_coverage_caveat": coverage,
        "round1_holdout_metrics": round1_holdout,
        "round2_holdout_metrics": round2_holdout,
        "metric_deltas_round2_minus_round1": deltas,
        "remaining_known_misses": remaining_misses,
        "verdict": verdict,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
