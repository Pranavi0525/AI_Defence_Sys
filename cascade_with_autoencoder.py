"""
Stage 4 -- Autoencoder Anomaly Escalation on top of the Verified
Stage 1+2 Cascade
======================================================================

WHAT THIS FILE DOES
--------------------
Extends blue_team_pipeline.py's EvaluationHarness (Stage 1 rules ->
Stage 2 XGBoost, unchanged, unmodified, verified) with a fourth stage:
a per-fold Autoencoder (autoencoder.py -- the same hand-rolled numpy
model whose reconstruction math was verified on a toy Gaussian-blob
problem before ever touching this data) that can ESCALATE a trace
Stage 1+2 scored low, if that trace's feature vector reconstructs
poorly against a model trained ONLY on normal behavior. Exactly the
same non-destructive escalation contract as Stage 3's graph layer:
final score = max(stage_1_2_score, autoencoder_score); it can never
downgrade a Stage 1+2 catch.

WHY THIS STAGE, AND WHY IT'S SCORED SEPARATELY FROM STAGE 3
----------------------------------------------------------------
This is layered directly on Stage 1+2 (not on top of the 3-stage
graph cascade) so its OWN marginal contribution is measurable in
isolation, same reasoning cascade_with_graph.py used to isolate
Stage 3's contribution rather than reporting one blended number nobody
could attribute credit to. Risk Fusion (the next planned component) is
where Stage 3 and Stage 4 get combined together on top of Stage 1+2.

CRITICAL DIFFERENCE FROM STAGE 2/3: THIS MODEL IS UNSUPERVISED
----------------------------------------------------------------
The autoencoder is trained, per fold, ONLY on that fold's TRAINING-SPLIT
LEGITIMATE rows -- it never sees a fraud label, and never sees any
fraud row at all during fitting. It has no notion of "fraud" as a
class; it only ever learns "what does normal behavior look like."
Its anomaly score is therefore not a fraud-probability in the same
sense Stage 2's XGBoost output is -- it's calibrated (via
autoencoder.error_to_score) against the 95th percentile of
reconstruction error on that fold's OWN normal training rows, so
"score >= 0.5" means "reconstructs worse than ~95% of the normal
behavior this fold's model was shown."

WHAT'S UNCHANGED FROM THE VERIFIED PIPELINE
----------------------------------------------
  - blue_team_pipeline.compute_stage_1_2_cascade -- the single source
    of truth for the Stage 1+2 baseline and the exact StratifiedKFold
    fold partition, same function cascade_with_graph.py reuses.
  - blue_team_pipeline.FEATURE_COLS -- same feature set Stage 2/3 use.
  - autoencoder.Autoencoder / train / error_to_score -- unmodified from
    the version whose math was verified on the toy problem.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 cascade_with_autoencoder.py

Outputs land in ./blue_team_output/stage4_autoencoder_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

import blue_team_pipeline as btp
from autoencoder import Autoencoder, train as train_ae, error_to_score

RANDOM_STATE = btp.CONFIG["RANDOM_STATE"]
N_SPLITS = 5
AE_HIDDEN_DIM = 8
# Feature count is 27 (btp.FEATURE_COLS); a hidden_dim of 8 gives a real
# bottleneck (< in_dim) without squeezing so hard the model can't learn
# ANY structure in normal behavior -- swept 4/8/16 on a held-out fold,
# 8 gave the cleanest separation between held-out-normal and fraud
# reconstruction error without needing more epochs to converge.
AE_EPOCHS = 400
AE_LR = 0.03
HIGH_PERCENTILE = 95.0
DECISION_THRESHOLD = btp.CONFIG["DECISION_THRESHOLD"]


# ---------------------------------------------------------------------------
# Step 1 -- feature table (unmodified path -- exact Stage 1+2 feature set)
# ---------------------------------------------------------------------------
def build_feature_table(cfg: dict):
    print("Building Stage 1+2's exact feature table...")
    df = btp.build_dataset(cfg)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 -- Stage 1+2 baseline + per-fold autoencoder escalation
# ---------------------------------------------------------------------------
def run_autoencoder_cascade(df, n_splits: int = N_SPLITS):
    feature_cols = btp.FEATURE_COLS
    X_raw = df[feature_cols].fillna(0).values.astype(float)
    y = df["fraud"].values.astype(int)

    # Standardize once, globally, the same way cascade_with_graph.py
    # standardizes for the GCN -- a neural net with gradient-descent
    # training is scale-sensitive in a way XGBoost's tree splits aren't.
    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    X_std = (X_raw - mu) / sigma

    # --- Stage 1+2, via the SAME function EvaluationHarness.run() uses.
    # Also hands back `folds`, so the autoencoder trains/tests on
    # IDENTICAL rows per fold as Stage 1+2 was scored on. ---
    stage_1_2_proba, escalate, folds = btp.compute_stage_1_2_cascade(
        df, feature_cols, btp.CONFIG, n_splits=n_splits
    )

    stage_1_2_4_proba = stage_1_2_proba.copy()
    ae_raw_score = np.zeros(len(df))  # for diagnostics/reporting

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        # Train ONLY on this fold's training-split LEGITIMATE rows --
        # an unsupervised anomaly detector must never see fraud during
        # fitting, or it would just learn to reconstruct fraud well too.
        legit_train_idx = train_idx[y[train_idx] == 0]

        model = Autoencoder(in_dim=X_std.shape[1], hidden_dim=AE_HIDDEN_DIM,
                             seed=RANDOM_STATE + fold)
        train_ae(model, X_std[legit_train_idx], epochs=AE_EPOCHS, lr=AE_LR)

        normal_train_err = model.reconstruction_error(X_std[legit_train_idx])
        test_err = model.reconstruction_error(X_std[test_idx])
        ae_scores = error_to_score(test_err, normal_train_err, high_percentile=HIGH_PERCENTILE)

        for local_i, global_i in enumerate(test_idx):
            ae_raw_score[global_i] = float(ae_scores[local_i])
            # Escalation rule: Stage 4 can only ADD score, never remove
            # it, exactly matching Stage 3's non-destructive contract.
            stage_1_2_4_proba[global_i] = max(stage_1_2_proba[global_i], ae_scores[local_i])

        print(f"  fold {fold}/{n_splits} done "
              f"(trained on {len(legit_train_idx)} legit rows, "
              f"normal train error 95th pct threshold used for scoring)")

    return stage_1_2_proba, stage_1_2_4_proba, ae_raw_score, y


def block_metrics(y_true, proba, threshold=DECISION_THRESHOLD):
    """Same metric function EvaluationHarness.run() and
    cascade_with_graph.py use (btp.block), so all three reports are
    computed by the one maintained implementation, not independently
    reimplemented and liable to silently diverge."""
    preds = (proba >= threshold).astype(int)
    result = btp.block(y_true, preds, proba)
    result["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()
    return result, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = btp.CONFIG
    out_dir = cfg["REPO_ROOT"] / cfg["OUTPUT_DIR"]
    out_dir.mkdir(exist_ok=True)

    df = build_feature_table(cfg)

    print(f"\nRunning Stage 1+2 -> +Autoencoder cascade, {N_SPLITS}-fold CV "
          f"(retrains a fresh autoencoder per fold, legit-only)...")
    stage_1_2_proba, stage_1_2_4_proba, ae_raw_score, y = run_autoencoder_cascade(df)

    stage_1_2_overall, stage_1_2_preds = block_metrics(y, stage_1_2_proba)
    stage_1_2_4_overall, stage_1_2_4_preds = block_metrics(y, stage_1_2_4_proba)

    rescued = int(((stage_1_2_preds == 0) & (stage_1_2_4_preds == 1) & (y == 1)).sum())
    downgraded = int(((stage_1_2_preds == 1) & (stage_1_2_4_preds == 0)).sum())
    new_false_positives = int(((stage_1_2_preds == 0) & (stage_1_2_4_preds == 1) & (y == 0)).sum())

    # by-family breakdown, same shape as EvaluationHarness.run()'s
    by_family_1_2 = {}
    by_family_1_2_4 = {}
    for fam in df["attack_family"].unique():
        if fam == "legitimate":
            continue
        mask = ((df["attack_family"] == fam) | (df["fraud"] == 0)).values
        idx = np.where(mask)[0]
        by_family_1_2[fam] = btp.block(y[idx], stage_1_2_preds[idx], stage_1_2_proba[idx])
        by_family_1_2_4[fam] = btp.block(y[idx], stage_1_2_4_preds[idx], stage_1_2_4_proba[idx])

    result = {
        "stage_1_2_overall": stage_1_2_overall,
        "stage_1_2_4_overall": stage_1_2_4_overall,
        "stage_1_2_by_family": by_family_1_2,
        "stage_1_2_4_by_family": by_family_1_2_4,
        "fraud_cases_rescued_by_stage4": rescued,
        "legit_cases_newly_flagged_by_stage4": new_false_positives,
        "fraud_cases_downgraded_by_stage4_should_be_zero": downgraded,
        "ae_hidden_dim": AE_HIDDEN_DIM,
        "ae_score_calibration_percentile": HIGH_PERCENTILE,
        "note": "downgraded is guaranteed 0 by construction (final score = "
                "max(stage_1_2, ae_score)) -- reported anyway as an "
                "explicit, checkable guardrail. legit_cases_newly_flagged "
                "is the real cost of this stage: unlike Stage 3 (which "
                "only touches graph-connected nodes), the autoencoder "
                "scores every row, so it can also escalate legitimate "
                "traces whose behavior is merely unusual, not fraudulent.",
    }

    print("\n" + "=" * 72)
    print("STAGE 1+2 (existing, verified) vs STAGE 1+2+4 (with autoencoder escalation)")
    print("=" * 72)
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("stage_1_2_by_family", "stage_1_2_4_by_family")}, indent=2))

    out_path = out_dir / "stage4_autoencoder_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
