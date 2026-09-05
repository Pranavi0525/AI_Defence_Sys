"""
Stage 5 -- Risk Fusion
=======================================================================
WHAT THIS FILE DOES
--------------------
Combines the four Blue Team signals -- Stage 1 rules, Stage 2 XGBoost,
Stage 3 GCN (graph), Stage 4 Autoencoder (novelty) -- into ONE final
risk score, replacing the `max(stage_1_2, stage_3, stage_4)` escalation
contract used to validate Stage 3 and Stage 4 in isolation.

WHY NOT JUST KEEP USING max()
----------------------------------------------------------------
max() says "if ANY detector fires, trust it completely." That's fine
for an ablation experiment (it isolates one stage's marginal
contribution, which is exactly why Stage 3 and Stage 4 used it) but it
is a bad PRODUCTION decision rule: stage4_autoencoder_results.json
shows precision falling from 98.4% (Stage 1+2 alone) to 79.3% once the
autoencoder is folded in with max() -- 66 legitimate traces newly
flagged, because "reconstructs unusually" and "is fraud" are not the
same thing. Risk Fusion's job is to let each detector contribute
*proportional to how much its evidence has actually been worth,
historically*, not with equal, unconditional veto power.

HOW FUSION WORKS: STACKED LOGISTIC REGRESSION ON OOF BASE SCORES
----------------------------------------------------------------
1. Stage 1+2, Stage 3 (GCN), and Stage 4 (Autoencoder) each already
   produce an OUT-OF-FOLD score for every row (see
   blue_team_pipeline.compute_stage_1_2_cascade,
   cascade_with_graph.run_three_stage_cascade,
   cascade_with_autoencoder.run_autoencoder_cascade) -- every row is
   scored by a model that never saw its label during training. That
   makes these three numbers safe to use as meta-features.
2. A small Logistic Regression is then trained on
   [stage_1_2_score, gcn_score, ae_score] -> fraud, itself evaluated
   with its OWN StratifiedKFold split (same folds Stage 1+2 used, so
   everything stays aligned) so the FINAL fused score is also
   out-of-fold and honestly reportable.
3. Logistic regression is deliberately chosen over a bigger model
   here: its coefficients are directly interpretable ("Risk Fusion
   currently weighs XGBoost's score 3.2x more heavily than the
   autoencoder's") which is the whole point of this stage -- the doc
   that motivated this file explicitly asked for that interpretability,
   not just a better F1 number.

HONEST FINDING CARRIED FORWARD FROM THIS FILE'S OWN VALIDATION
----------------------------------------------------------------
cascade_with_graph.py's cross-customer graph uses MIN_FANOUT_FOR_EDGE=6
to separate the 4 synthetic ring collectors (6 distinct customers each)
from natural ID-pool collisions (which top out at 3 distinct customers
in this simulator). That threshold was tuned to a graph WITH the
synthetic ring overlay present. On the real, unmodified corpus (no
overlay), no entity clears fanout >= 6, meaning `connected_mask` is
all-False and the GCN's score is 0 for every single row. This is not a
Risk Fusion bug -- it is the same honest finding Stage 3 already
documented (the ring signal only exists in this dataset via the
deliberately-injected overlay), surfaced again here because Risk
Fusion is the layer where it actually matters: a meta-model trained on
a feature that is always 0 will correctly learn to ignore it, and the
comparison table below reports BOTH the real-corpus run (GCN inert, by
construction) and a ring-overlay diagnostic run (proving the fusion
layer, not just max(), can still recover the graph signal when it's
actually present -- see run_fusion_with_ring_diagnostic()).

CARRIED-FORWARD LIMITATION (not fixed here, flagged same as upstream)
----------------------------------------------------------------
BLUE_TEAM_INTEGRATION_SPEC.md Section 9 calls for splitting by
customer/entity when a customer can appear in multiple traces.
Legitimate traces are session-windowed per customer (one customer can
produce several session rows), but every split used by Stage 1+2
(and therefore by every stage built on top of it, including this one)
is row-level StratifiedKFold, not customer-level. This was found
during the Blue Team readiness review that preceded this file and is
recorded here rather than silently inherited.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 risk_fusion.py

Outputs land in ./blue_team_output/risk_fusion_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import blue_team_pipeline as btp
from gcn import OneLayerGCN, normalize_adjacency, train as train_gcn
from autoencoder import Autoencoder, train as train_ae, error_to_score
from cascade_with_graph import build_cross_customer_graph, GCN_EPOCHS, GCN_LR, GCN_HIDDEN_DIM
from quiet_ring_overlay import apply_quiet_ring_overlay, N_RING_TRACES

RANDOM_STATE = btp.CONFIG["RANDOM_STATE"]
N_SPLITS = 5
AE_HIDDEN_DIM = 8
AE_EPOCHS = 400
AE_LR = 0.03
HIGH_PERCENTILE = 95.0
DECISION_THRESHOLD = btp.CONFIG["DECISION_THRESHOLD"]


# ---------------------------------------------------------------------------
# Step 1 -- base scores (Stage 1+2, Stage 3, Stage 4), all OOF, all on the
# IDENTICAL fold partition
# ---------------------------------------------------------------------------
def compute_base_scores(df: pd.DataFrame, A: np.ndarray, connected_mask: np.ndarray,
                         n_splits: int = N_SPLITS):
    """Returns (stage_1_2_proba, gcn_score, ae_score, y, folds), every
    score array out-of-fold and length == len(df).

    This intentionally reuses btp.compute_stage_1_2_cascade as the
    SINGLE source of truth for the fold partition -- same reasoning
    cascade_with_graph.py and cascade_with_autoencoder.py already
    apply, extended here so Stage 3 and Stage 4 are ALSO computed on
    that exact partition rather than each independently re-deriving
    (and risking silently drifting from) it.
    """
    feature_cols = btp.FEATURE_COLS
    X_raw = df[feature_cols].fillna(0).values.astype(float)
    y = df["fraud"].values.astype(int)

    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    X_std = (X_raw - mu) / sigma

    stage_1_2_proba, escalate, folds = btp.compute_stage_1_2_cascade(
        df, feature_cols, btp.CONFIG, n_splits=n_splits
    )

    gcn_score = np.zeros(len(df))
    ae_score = np.zeros(len(df))

    A_hat = normalize_adjacency(A)
    M = A_hat @ X_std  # message-passed features, fixed for the whole run

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        # --- Stage 3: GCN, transductive, trained only on this fold's
        # train labels, scored over connected test rows only ---
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        gcn = OneLayerGCN(in_dim=X_std.shape[1], hidden_dim=GCN_HIDDEN_DIM,
                           seed=RANDOM_STATE + fold)
        train_gcn(gcn, M, y.astype(float), train_mask, epochs=GCN_EPOCHS, lr=GCN_LR)
        for i in test_idx:
            if connected_mask[i]:
                gcn_score[i] = gcn.p[i]

        # --- Stage 4: Autoencoder, trained only on this fold's
        # train-split LEGITIMATE rows ---
        legit_train_idx = train_idx[y[train_idx] == 0]
        ae = Autoencoder(in_dim=X_std.shape[1], hidden_dim=AE_HIDDEN_DIM,
                          seed=RANDOM_STATE + fold)
        train_ae(ae, X_std[legit_train_idx], epochs=AE_EPOCHS, lr=AE_LR)
        normal_train_err = ae.reconstruction_error(X_std[legit_train_idx])
        test_err = ae.reconstruction_error(X_std[test_idx])
        ae_scores = error_to_score(test_err, normal_train_err, high_percentile=HIGH_PERCENTILE)
        for local_i, global_i in enumerate(test_idx):
            ae_score[global_i] = float(ae_scores[local_i])

        print(f"  fold {fold}/{n_splits} base scores done")

    return stage_1_2_proba, gcn_score, ae_score, y, folds


# ---------------------------------------------------------------------------
# Step 2 -- the fusion meta-model itself: pure function of arrays, no
# base-model training, so it is fast and independently unit-testable
# (see tests/test_risk_fusion.py).
# ---------------------------------------------------------------------------
def fit_fusion_oof(meta_X: np.ndarray, y: np.ndarray, folds: list[tuple],
                    random_state: int = RANDOM_STATE):
    """Trains a Logistic Regression meta-model per fold (on the OTHER
    folds' base scores + labels) and predicts on that fold's test
    rows, producing a final out-of-fold fused score for every row.

    Reusing the SAME fold partition base scores were computed on is
    deliberate: base scores are already OOF w.r.t. the base models, so
    this is standard two-level stacking -- the meta-model never sees a
    row's label used to produce that row's own meta-features twice.

    Returns:
        fused_proba: (n,) OOF fused fraud probability
        fold_coefs: list of dicts, one per fold, {feature: weight} --
            kept per-fold (not just averaged) so you can see whether
            the learned weighting is stable across folds or noisy,
            which matters for the "how much should we trust this
            weighting" interview question.
    """
    n = len(y)
    fused_proba = np.zeros(n)
    fold_coefs = []

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        meta_model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=random_state,
        )
        meta_model.fit(meta_X[train_idx], y[train_idx])
        fused_proba[test_idx] = meta_model.predict_proba(meta_X[test_idx])[:, 1]
        fold_coefs.append({
            "stage_1_2_score": float(meta_model.coef_[0][0]),
            "gcn_score": float(meta_model.coef_[0][1]),
            "ae_score": float(meta_model.coef_[0][2]),
            "intercept": float(meta_model.intercept_[0]),
        })

    return fused_proba, fold_coefs


# ---------------------------------------------------------------------------
# Step 3 -- comparison table + metrics, same shapes as every prior stage
# ---------------------------------------------------------------------------
def block_metrics(y_true, proba, threshold=DECISION_THRESHOLD):
    preds = (proba >= threshold).astype(int)
    result = btp.block(y_true, preds, proba)
    result["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()
    return result, preds


def by_family(df: pd.DataFrame, y: np.ndarray, proba: np.ndarray, preds: np.ndarray):
    out = {}
    for fam in df["attack_family"].unique():
        if fam == "legitimate":
            continue
        mask = ((df["attack_family"] == fam) | (df["fraud"] == 0)).values
        idx = np.where(mask)[0]
        out[fam] = btp.block(y[idx], preds[idx], proba[idx])
    return out


def run_risk_fusion(df: pd.DataFrame, A: np.ndarray, connected_mask: np.ndarray,
                     n_splits: int = N_SPLITS):
    """Full comparison: Stage1+2 alone / +GCN alone / +AE alone /
    naive max(all three) / Risk Fusion (stacked LR). All on the exact
    same rows and fold partition, per the doc's request for an
    apples-to-apples experimental story.
    """
    stage_1_2, gcn_score, ae_score, y, folds = compute_base_scores(df, A, connected_mask, n_splits)

    stage_1_2_3 = np.maximum(stage_1_2, gcn_score)
    stage_1_2_4 = np.maximum(stage_1_2, ae_score)
    naive_max_all = np.maximum(stage_1_2, np.maximum(gcn_score, ae_score))

    meta_X = np.column_stack([stage_1_2, gcn_score, ae_score])
    fused_proba, fold_coefs = fit_fusion_oof(meta_X, y, folds)

    comparison = {}
    for name, proba in [
        ("stage_1_2_only", stage_1_2),
        ("stage_1_2_plus_gcn_max", stage_1_2_3),
        ("stage_1_2_plus_autoencoder_max", stage_1_2_4),
        ("naive_max_all_three", naive_max_all),
        ("risk_fusion_stacked_lr", fused_proba),
    ]:
        overall, preds = block_metrics(y, proba)
        comparison[name] = {
            "overall": overall,
            "by_family": by_family(df, y, proba, preds),
        }

    avg_coefs = {
        k: float(np.mean([f[k] for f in fold_coefs]))
        for k in fold_coefs[0]
    }

    result = {
        "comparison": comparison,
        "fusion_fold_coefficients": fold_coefs,
        "fusion_avg_coefficients": avg_coefs,
        "n_graph_connected_nodes": int(connected_mask.sum()),
        "note": (
            "n_graph_connected_nodes reflects the REAL corpus (no ring "
            "overlay) -- if this is 0, the GCN score is 0 for every row "
            "by construction (no entity clears the fan-out threshold "
            "without the synthetic ring), and fusion_avg_coefficients' "
            "gcn_score weight should be interpreted as 'the meta-model "
            "correctly learned to ignore an always-zero feature', not "
            "'the GCN is a weak detector.' See run_fusion_with_ring_diagnostic() "
            "for the run that proves fusion recovers a graph signal when "
            "one actually exists."
        ),
    }
    return result, fused_proba, y


# ---------------------------------------------------------------------------
# Step 4 -- ring-overlay diagnostic: proves the fusion layer (not just
# max()) correctly integrates the graph signal WHEN one is present.
# Mirrors cascade_with_graph.py's own validation methodology.
# ---------------------------------------------------------------------------
def run_fusion_with_ring_diagnostic(cfg: dict, n_splits: int = N_SPLITS):
    from cascade_with_graph import load_all_records, build_feature_table_and_graph

    print("Building ring-overlay diagnostic dataset (synthetic, clearly flagged)...")

    # load_all_records() returns ONE list of records.
    all_records = load_all_records(cfg)

    # Split the real corpus into attack and legitimate records.
    # The synthetic overlay is applied ONLY to legitimate records.
    legit_records = [
        r for r in all_records
        if r["attack_family"] == "legitimate"
    ]

    attack_records = [
        r for r in all_records
        if r["attack_family"] != "legitimate"
    ]

    # Inject the deliberately synthetic quiet-ring diagnostic.
    # This does NOT modify the genuine Red Team MULE_NETWORK corpus.
    legit_records, ring_ids = apply_quiet_ring_overlay(
        legit_records,
        n_ring=N_RING_TRACES,
        seed=RANDOM_STATE,
    )

    # Recombine untouched real attacks + synthetic diagnostic traces.
    diagnostic_records = attack_records + legit_records

    # Evaluation-only ring membership.
    ring_membership = {
        trace_id: "synthetic_quiet_ring"
        for trace_id in ring_ids
    }

    df, A, connected_mask = build_feature_table_and_graph(
        diagnostic_records,
        ring_membership,
    )

    result, fused_proba, y = run_risk_fusion(
        df,
        A,
        connected_mask,
        n_splits,
    )

    # Evaluate Risk Fusion specifically on the synthetic ring traces.
    ring_mask = df["is_ring"].values.astype(bool)

    ring_overall, ring_preds = block_metrics(
        y[ring_mask],
        fused_proba[ring_mask],
    )

    result["ring_only_risk_fusion_metrics"] = ring_overall
    result["n_ring_traces"] = int(ring_mask.sum())
    result["is_synthetic_diagnostic"] = True
    result["diagnostic_purpose"] = (
        "This run uses the deliberately-injected quiet ring overlay "
        "(NOT real Red Team output -- see quiet_ring_overlay.py) purely "
        "to verify the fusion meta-model can recover a graph signal "
        "when one exists. The genuine MULE_NETWORK corpus remains "
        "untouched."
    )

    return result

def main():
    cfg = btp.CONFIG
    out_dir = cfg["REPO_ROOT"] / cfg["OUTPUT_DIR"]
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("PART 1 / 2 -- Risk Fusion on the REAL corpus (no ring overlay)")
    print("=" * 72)
    df = btp.build_dataset(cfg).reset_index(drop=True)
    all_records_for_graph = (
        btp.load_attack_corpus(
            cfg["REPO_ROOT"] / cfg["ATO_CORPUS_PATH"],
            "ATO",
        )
        + btp.load_attack_corpus(
            cfg["REPO_ROOT"] / cfg["APP_CORPUS_PATH"],
            "APP",
        )
        + btp.load_attack_corpus(
            cfg["REPO_ROOT"] / cfg["MULE_CORPUS_PATH"],
            "MULE_NETWORK",
        )
    )
    # Rebuild legit records the same way build_dataset did internally, so we
    # have the raw `events` field (needed for graph-building) alongside the
    # already-extracted feature df above. Cheap to regenerate (deterministic
    # seed) rather than threading raw records through build_dataset's return.
    legit_records = btp.build_legitimate_traces(cfg)
    all_records = all_records_for_graph + legit_records
    from cascade_with_graph import build_cross_customer_graph as _bccg
    edges = _bccg(all_records)
    trace_id_to_idx = {tid: i for i, tid in enumerate(df["trace_id"])}
    A = np.zeros((len(df), len(df)))
    for a, b in edges:
        if a in trace_id_to_idx and b in trace_id_to_idx:
            i, j = trace_id_to_idx[a], trace_id_to_idx[b]
            A[i, j] = 1
            A[j, i] = 1
    connected_mask = A.sum(axis=1) > 0

    real_result, _, _ = run_risk_fusion(df, A, connected_mask)

    print("\n" + "=" * 72)
    print("PART 2 / 2 -- Ring-overlay diagnostic (synthetic, proves fusion")
    print("integrates graph signal correctly when one is present)")
    print("=" * 72)
    diagnostic_result = run_fusion_with_ring_diagnostic(cfg)

    output = {
        "real_corpus_risk_fusion": real_result,
        "ring_overlay_diagnostic": diagnostic_result,
    }
    from artifact_metadata import stamp_artifact
    output = stamp_artifact(
        output,
        cfg["REPO_ROOT"],
        seeds={"RANDOM_STATE": cfg.get("RANDOM_STATE")},
        dataset_files=[
            cfg["REPO_ROOT"] / cfg["ATO_CORPUS_PATH"],
            cfg["REPO_ROOT"] / cfg["APP_CORPUS_PATH"],
        ],
        feature_cols=btp.FEATURE_COLS,
    )
    out_path = out_dir / "risk_fusion_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("RISK FUSION -- COMPARISON TABLE (real corpus)")
    print("=" * 72)
    for name, block_ in real_result["comparison"].items():
        o = block_["overall"]
        print(f"  {name:35s} precision={o.get('precision', 0):.3f}  "
              f"recall={o.get('recall', 0):.3f}  f1={o.get('f1', 0):.3f}")
    print("\nAvg fusion coefficients (which detector the meta-model trusts):")
    print(json.dumps(real_result["fusion_avg_coefficients"], indent=2))
    print(f"\nAll outputs written to {out_path}")


if __name__ == "__main__":
    main()
