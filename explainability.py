"""
explainability.py
===================
Stage 7 -- Explainability layer on top of the verified cascade:

    Stage 1 (deterministic rules)  -- blue_team_pipeline.stage1_rule_filter
    Stage 2 (XGBoost)              -- blue_team_pipeline FEATURE_COLS + model
    Stage 3 (graph / GCN)          -- cascade_with_graph.py (base score, one fusion input)
    Stage 4 (autoencoder)          -- cascade_with_autoencoder.py (base score, one fusion input)
    Stage 5 (Risk Fusion)          -- risk_fusion.py (stacked LR over Stage 2/3/4 scores)
    Stage 6 (cost-optimal policy)  -- decision_policy.py (t_review / t_block, now tuned
                                       against the Stage 5 fused score)

IMPORTANT: the decision-grade "final_score" used throughout this file is
now the Stage 5 FUSED score (GCN + autoencoder + logistic reweighting),
not a raw Stage 1+2+3 max()-style score. Any lift over the Stage 1+2
score can come from the graph, the autoencoder, or the fusion model's
reweighting -- NOT necessarily the graph alone. This file is careful not
to attribute a fusion-driven lift to "graph escalation" specifically
unless the graph signal is the one actually elevated (see
escalated_vs_stage_1_2 / ring_flagged_and_final_score_cleared_block
below); where it can't tell the difference, it says so.

Everything upstream of this file answers "what did the cascade decide?".
This file answers "WHY did it decide that?", at two levels:

  1. GLOBAL explainability -- across the whole validation population:
     which features drive Stage 2's score overall and per attack family
     (SHAP TreeExplainer on the fitted XGBoost model), plus Stage 1's
     rule fire-rates and Stage 3's graph-escalation footprint. This is
     the "what has the model learned" view for a judge/reviewer.

  2. PER-CASE explainability -- for one specific trace: which Stage 1
     rule(s) fired (or didn't), which features pushed Stage 2's score up
     or down and by how much (local SHAP values), whether Stage 3's
     graph escalation applied and to which neighbors, and what Stage 4's
     thresholds finally decided (ALLOW / REVIEW / BLOCK) and why. This
     is the "why did THIS transaction get flagged" view -- the artifact
     a fraud analyst or a judge asking "show me one" actually wants.

IMPORTANT SCOPE NOTE -- what's "real" here vs. "for explanation only"
----------------------------------------------------------------------
Two different model fits are used here, deliberately, for two different
jobs -- conflating them would either be dishonest (reporting an
optimistic score) or unnecessarily wasteful (retraining per-fold just to
ask "why"):

  - DECISION numbers (the score used to pick ALLOW/REVIEW/BLOCK for a
    case, and to pick which cases are "interesting" to show) come from
    cascade_with_graph.run_three_stage_cascade's genuine 5-fold
    OUT-OF-FOLD scores -- every row is scored by a fold that never
    trained on it, byte-identical protocol to the one that produced the
    numbers in three_stage_cascade_results.json and
    decision_policy_results.json. These are the numbers you can defend.

  - EXPLANATION mechanics (which features drove the score, SHAP local
    attributions, the GCN's learned neighbor-propagation behavior) come
    from a Stage 2 model and Stage 3 GCN each fit ONCE on the FULL
    dataset. A single stable fit is what SHAP's TreeExplainer needs to
    interrogate; retraining 5 different fold-models would give 5
    different, less legible explanations for the same case. Do NOT
    quote this second fit's raw probability as a performance number --
    it will look optimistic on rows it was also fit on. The displayed
    per-case "score" is always the honest out-of-fold one; only the
    feature-attribution breakdown underneath it comes from the full fit.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 explainability.py

Outputs land in ./blue_team_output/explainability/:
    global_feature_importance.json   -- overall + per-family mean |SHAP|
    global_shap_summary.png          -- bar chart, top features overall
    case_reports.json                -- structured per-case explanations
    case_reports.md                  -- the same, as readable narratives
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import blue_team_pipeline as btp
import cascade_with_graph as cwg
from gcn import OneLayerGCN, normalize_adjacency, train as train_gcn

matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_ok = False

RANDOM_STATE = cwg.RANDOM_STATE
FEATURE_COLS = btp.FEATURE_COLS
OUT_DIR = Path(__file__).parent / "blue_team_output" / "explainability"

# Human-readable phrasing for each feature -- used to turn a SHAP value
# ("min_time_between_transactions = -0.83") into a sentence a non-ML
# reader can act on. Falls back to the raw feature name if not listed.
FEATURE_LABELS = {
    "count_transaction": "number of transactions in the trace",
    "count_session_login": "number of logins in the trace",
    "count_device_registration": "number of new-device registrations",
    "count_beneficiary_addition": "number of new payees added",
    "total_events": "total number of events observed",
    "window_seconds": "length of the observation window",
    "transactions_per_hour": "transaction velocity (txns/hour)",
    "mean_time_between_transactions": "average gap between transactions",
    "min_time_between_transactions": "shortest gap between two transactions",
    "transactions_per_session": "transactions per login session",
    "time_login_to_first_transaction": "time from login to first transaction",
    "login_attempt_count_max": "number of login attempts",
    "auth_failure_present": "a failed login attempt occurred",
    "new_device_present": "a new device was registered",
    "time_device_registration_to_transaction": "time from new-device registration to transaction",
    "beneficiary_added_before_transaction": "a new payee was added before paying them",
    "time_from_beneficiary_add_to_transaction": "time from adding the payee to paying them",
    "failed_transaction_count": "number of failed transaction attempts",
    "failed_then_completed": "a failed transaction was immediately retried and succeeded",
    "amount_mean": "average transaction amount",
    "amount_max": "largest transaction amount",
    "amount_min": "smallest transaction amount",
    "amount_std": "spread in transaction amounts",
    "amount_cv": "relative spread in transaction amounts",
    "amount_change_after_failure": "the amount was lowered after a failed attempt",
    "amount_trend": "transaction amounts trending up/down within the trace",
    "distinct_channels": "number of distinct channels used",
}

STAGE1_RULES = [
    ("beneficiary_added_before_transaction", "==", 1,
     "a payee was added and then paid within the same session"),
    ("new_device_present", "==", 1,
     "a new device was registered on this trace"),
    ("min_time_between_transactions", "<", 3600,
     "two transactions happened under an hour apart"),
    ("transactions_per_hour", ">", 2.5,
     "transaction velocity exceeded 2.5/hour"),
]


# ---------------------------------------------------------------------------
# Step 1 -- build (or load) the reference artifacts every explanation needs
# ---------------------------------------------------------------------------
@dataclass
class ReferenceArtifacts:
    df: pd.DataFrame
    all_records: list
    A: np.ndarray
    connected_mask: np.ndarray
    ring_membership: set          # graph-connected trace ids (see get_graph_connected_trace_ids)
    stage2_model: XGBClassifier
    explainer: "shap.TreeExplainer"
    shap_values: np.ndarray          # (n, n_features)
    gcn_probs: np.ndarray            # full-graph demo GCN, see module docstring
    stage1_escalate: np.ndarray
    stage2_score_hypothetical: np.ndarray  # full-fit model score, regardless of stage1 (explanation only)
    oof_stage_1_2: np.ndarray        # GENUINE out-of-fold stage1+2 score (decision-grade, no graph/AE/fusion)
    oof_final: np.ndarray            # GENUINE out-of-fold Stage 5 FUSED score (decision-grade) --
                                      # GCN + autoencoder + logistic reweighting, via
                                      # decision_policy.get_validation_data_fused(); NOT a raw
                                      # stage1+2+3 max() score. See module docstring.
    t_review: float
    t_block: float


def _decision_thresholds() -> tuple[float, float]:
    """Pull the prevalence-corrected Stage 6 (decision policy) thresholds,
    tuned against the Stage 5 fused score, if available; fall back to the
    raw decision boundary otherwise."""
    path = Path(__file__).parent / "decision_policy_results.json"
    if path.exists():
        with open(path) as f:
            r = json.load(f)
        c = r["corrected"]
        return float(c["t_review"]), float(c["t_block"])
    return btp.CONFIG["DECISION_THRESHOLD"], 1.01  # no REVIEW band fallback


def build_reference_artifacts() -> ReferenceArtifacts:
    cfg = btp.CONFIG
    all_records = cwg.load_all_records(cfg)
    ring_membership = cwg.get_graph_connected_trace_ids(all_records)
    df, A, connected_mask = cwg.build_feature_table_and_graph(all_records, ring_membership)

    X_raw = df[FEATURE_COLS].fillna(0).values.astype(float)
    y = df["fraud"].values.astype(int)

    print("\nFitting Stage 2 reference model on the FULL dataset "
          "(for interpretability only -- see module docstring)...")
    stage2_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    stage2_model.fit(X_raw, y)

    print("Computing SHAP values (TreeExplainer, exact for tree models)...")
    explainer = shap.TreeExplainer(stage2_model)
    shap_values = explainer.shap_values(X_raw)
    if isinstance(shap_values, list):  # older shap API: list per class
        shap_values = shap_values[1]

    stage1_escalate = df.apply(btp.stage1_rule_filter, axis=1).values
    stage2_score_hypothetical = stage2_model.predict_proba(X_raw)[:, 1]

    print("Fitting Stage 3 reference GCN on the FULL graph "
          "(for interpretability only -- see module docstring)...")
    X_std = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    A_hat = normalize_adjacency(A)
    M = A_hat @ X_std
    gcn = OneLayerGCN(in_dim=X_std.shape[1], hidden_dim=cwg.GCN_HIDDEN_DIM, seed=RANDOM_STATE)
    train_mask = np.ones(len(df), dtype=bool)
    train_gcn(gcn, M, y.astype(float), train_mask, epochs=cwg.GCN_EPOCHS, lr=cwg.GCN_LR)
    gcn_probs = gcn.p

    # Genuine, decision-grade scores: the cache is written by
    # decision_policy.get_validation_data_fused(), so oof_final here is the
    # Stage 5 FUSED score -- byte-identical to what decision_policy.py's
    # frozen thresholds and miss_collector.py's misses.jsonl are based on.
    cache_path = Path(__file__).parent / "decision_policy_validation_cache.npz"
    if cache_path.exists():
        print("Loading cached out-of-fold Stage 5 fused scores "
              "(decision_policy_validation_cache.npz)...")
        cached = np.load(cache_path)
        assert len(cached["y"]) == len(df) and np.array_equal(cached["y"], y), (
            "Cached validation arrays don't align with this df -- delete "
            "decision_policy_validation_cache.npz and rerun to regenerate."
        )
        oof_final = cached["proba"]
        # The cache only stores the final (post-graph) score. Pre-graph
        # OOF is only needed to detect "rescued by graph"; recompute
        # ONLY that via a fresh OOF run since it's not cached separately.
        print("Running Stage 1+2+3 OOF cascade to recover the pre-graph "
              "OOF score too (needed to detect graph rescues)...")
        oof_stage_1_2, _, _ = cwg.run_three_stage_cascade(df, A, connected_mask)
    else:
        print("No cached OOF scores found -- running the full Risk Fusion "
              "pipeline now via decision_policy.get_validation_data_fused() "
              "(a couple of minutes, fresh GCN + autoencoder + fusion LR per "
              "fold) so oof_final is the same Stage 5 fused score decision_policy.py "
              "and miss_collector.py use, not a raw stage1+2+3 max() score...")
        import decision_policy as dp
        _, _, oof_final, _, _ = dp.get_validation_data_fused()
        oof_stage_1_2, _, _ = cwg.run_three_stage_cascade(df, A, connected_mask)

    t_review, t_block = _decision_thresholds()

    return ReferenceArtifacts(
        df=df, all_records=all_records, A=A, connected_mask=connected_mask,
        ring_membership=ring_membership, stage2_model=stage2_model, explainer=explainer,
        shap_values=shap_values, gcn_probs=gcn_probs,
        stage1_escalate=stage1_escalate,
        stage2_score_hypothetical=stage2_score_hypothetical,
        oof_stage_1_2=oof_stage_1_2, oof_final=oof_final,
        t_review=t_review, t_block=t_block,
    )


# ---------------------------------------------------------------------------
# Step 2 -- global explainability
# ---------------------------------------------------------------------------
def global_feature_importance(ref: ReferenceArtifacts, top_k: int = 12) -> dict:
    mean_abs = np.abs(ref.shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    overall = [
        {"feature": FEATURE_COLS[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order[:top_k]
    ]

    by_family = {}
    for fam in ref.df["attack_family"].unique():
        if fam == "legitimate":
            continue
        mask = (ref.df["attack_family"] == fam).values
        if mask.sum() == 0:
            continue
        fam_mean_abs = np.abs(ref.shap_values[mask]).mean(axis=0)
        fam_order = np.argsort(-fam_mean_abs)
        by_family[fam] = [
            {"feature": FEATURE_COLS[i], "mean_abs_shap": round(float(fam_mean_abs[i]), 4)}
            for i in fam_order[:top_k]
        ]

    # Explicit Stage 1 rule diagnostics (each condition individually,
    # not just the OR of all four)
    fraud_mask = ref.df["fraud"].values == 1
    legit_mask = ~fraud_mask
    stage1_rule_diag = {}
    for feat, op, thresh, _desc in STAGE1_RULES:
        vals = ref.df[feat].values
        if op == "==":
            fired = vals == thresh
        elif op == "<":
            fired = vals < thresh
        else:
            fired = vals > thresh
        stage1_rule_diag[feat] = {
            "condition": f"{feat} {op} {thresh}",
            "fire_rate_on_fraud": round(float(fired[fraud_mask].mean()), 4),
            "fire_rate_on_legit": round(float(fired[legit_mask].mean()), 4),
        }

    return {
        "overall_top_features": overall,
        "top_features_by_attack_family": by_family,
        "stage1_rule_diagnostics": stage1_rule_diag,
        "stage_5_fusion_lift": {
            "n_graph_connected_nodes": int(ref.connected_mask.sum()),
            "n_total_nodes": int(len(ref.df)),
            "n_fraud_rescued_by_fusion_over_stage_1_2": int(np.sum(
                (ref.oof_stage_1_2 < ref.t_block) & (ref.oof_final >= ref.t_block) & (ref.df["fraud"].values == 1)
            )),
            "note": "final_score (oof_final) is the Stage 5 fused score "
                    "(GCN + autoencoder + logistic reweighting), not a "
                    "graph-only score. This count is fraud rows the fused "
                    "score newly crosses t_block on, relative to Stage 1+2 "
                    "alone -- it should NOT be read as 'rescued by the "
                    "graph specifically' without also checking whether that "
                    "row is graph-connected (n_graph_connected_nodes) and, "
                    "even then, the autoencoder or fusion reweighting could "
                    "still be the actual driver.",
        },
        "stage_6_thresholds_used": {"t_review": ref.t_review, "t_block": ref.t_block},
    }


def plot_global_shap_bar(ref: ReferenceArtifacts, importance: dict, out_path: Path):
    if not matplotlib_ok:
        print("  matplotlib unavailable -- skipping PNG chart, JSON still written.")
        return
    top = importance["overall_top_features"]
    names = [FEATURE_LABELS.get(t["feature"], t["feature"]) for t in top][::-1]
    vals = [t["mean_abs_shap"] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="#4C72B0")
    ax.set_xlabel("mean |SHAP value|  (avg. impact on Stage 2 fraud score)")
    ax.set_title("Stage 2 (XGBoost) -- global feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Step 3 -- per-case explainability
# ---------------------------------------------------------------------------
def _decision(score: float, t_review: float, t_block: float) -> str:
    if score >= t_block:
        return "BLOCK"
    if score >= t_review:
        return "REVIEW"
    return "ALLOW"


# ---------------------------------------------------------------------------
# Case dossier -- investigator-facing summary for Review/Block decisions
# ---------------------------------------------------------------------------
def case_dossier(ref: "ReferenceArtifacts", idx: int, explanation: dict | None = None) -> dict | None:
    """
    Short investigator-facing summary for a Review/Block decision: which
    signals fired, which side is liable under the current role-aware
    policy (decision_policy.liable_side/acting_side), and similar past
    cases (graph-connected fraud neighbors already computed by
    explain_case -- this deliberately reuses Stage 3's existing signal
    rather than adding a second similarity model).

    Scope, explicitly: this is an INVESTIGATION AID. It never makes or
    overrides the Allow/Review/Block decision -- that decision is made
    upstream by decision_policy.py before this function is even called.
    It exists only to help a human reviewer act faster on a decision
    that's already been made.

    Returns None for ALLOW cases -- there's nothing for an investigator
    to review or act on there, so a dossier isn't useful.
    """
    if explanation is None:
        explanation = explain_case(ref, idx)

    decision = explanation["stage6_decision_policy"]["decision"]
    if decision == "ALLOW":
        return None

    import decision_policy as dp  # local import: keeps this file's own

    is_fraud = explanation["true_label"] == "fraud"
    fam = explanation["attack_family"] if is_fraud else None
    liable = dp.liable_side(fam) if is_fraud else "N/A"
    acting = dp.acting_side(fam) if is_fraud else "N/A"

    fired_signals = (
        [f["label"] for f in explanation["stage2"]["top_features"][:3]]
        if explanation["stage2"]["ran"]
        else [r["rule"] for r in explanation["stage1"]["rules_checked"] if r["fired"]]
    )

    similar = [n for n in explanation["stage3"]["neighbors"] if n["fraud_label"] == 1][:3]

    lines = [
        f"DECISION: {decision} -- trace {explanation['trace_id']} "
        f"(customer {explanation['customer_id']}), "
        f"${explanation['dollars_in_trace']:,.2f} moved.",
    ]
    if is_fraud:
        lines.append(
            f"Attack family: {fam} ({explanation['attack_difficulty']}). "
            f"Liable side under current policy: {liable} (acting side: {acting})."
        )
    else:
        lines.append(
            "True label: legitimate -- this is a false positive under the "
            "current thresholds. Liability fields don't apply."
        )
    lines.append(
        "Signals that drove this decision: "
        + (", ".join(fired_signals) if fired_signals else "none recorded") + "."
    )
    if similar:
        lines.append(
            "Similar past cases (graph-connected fraud traces): "
            + ", ".join(f"{s['trace_id']} (customer {s['customer_id']})" for s in similar) + "."
        )
    else:
        lines.append("No graph-connected similar past cases found for this trace.")
    lines.append(
        "NOTE: this dossier is an investigation aid only. It summarizes why "
        "the automated policy made its decision; it does not itself make or "
        "override that decision."
    )

    return {
        "trace_id": explanation["trace_id"],
        "decision": decision,
        "attack_family": fam,
        "liable_side": liable,
        "acting_side": acting,
        "signals_fired": fired_signals,
        "similar_past_cases": similar,
        "dossier_text": " ".join(lines),
    }


def _trace_dollars(record: dict) -> float:
    return sum(float(e["amount"]) for e in record["events"] if e["event_type"] == "TRANSACTION")


def explain_case(ref: ReferenceArtifacts, idx: int, top_k_features: int = 5) -> dict:
    row = ref.df.iloc[idx]
    record = ref.all_records[idx]
    x = ref.df[FEATURE_COLS].fillna(0).values[idx]
    sv = ref.shap_values[idx]

    # --- Stage 1 ---
    rules_checked = []
    any_fired = False
    for feat, op, thresh, desc in STAGE1_RULES:
        val = row[feat]
        fired = bool((val == thresh) if op == "==" else (val < thresh) if op == "<" else (val > thresh))
        any_fired = any_fired or fired
        rules_checked.append({
            "rule": desc, "feature": feat, "value": round(float(val), 2) if isinstance(val, (int, float, np.floating, np.integer)) else val,
            "threshold": thresh, "fired": fired,
        })
    stage1_escalated = bool(ref.stage1_escalate[idx])
    assert stage1_escalated == any_fired

    # --- Stage 2 ---
    order = np.argsort(-np.abs(sv))[:top_k_features]
    top_features = []
    for i in order:
        feat = FEATURE_COLS[i]
        top_features.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(float(x[i]), 3),
            "shap_value": round(float(sv[i]), 4),
            "pushed_score": "up" if sv[i] > 0 else "down",
        })
    stage2_score = float(ref.oof_stage_1_2[idx])           # decision-grade, out-of-fold
    stage2_score_hypothetical = float(ref.stage2_score_hypothetical[idx])  # explanation-only, full-fit

    # --- Stage 3 ---
    connected = bool(ref.connected_mask[idx])
    neighbors = []
    if connected:
        neighbor_idxs = np.where(ref.A[idx] > 0)[0]
        for j in neighbor_idxs:
            neighbors.append({
                "trace_id": ref.df.iloc[j]["trace_id"],
                "customer_id": ref.df.iloc[j]["customer_id"],
                "fraud_label": int(ref.df.iloc[j]["fraud"]),
                "is_ring": bool(ref.df.iloc[j]["is_ring"]),
            })
    gcn_score = float(ref.gcn_probs[idx]) if connected else None  # explanation-only, full-fit GCN
    final_score = float(ref.oof_final[idx])                        # decision-grade, out-of-fold, Stage 5 FUSED score
    # final_score is Stage 5's fused output (GCN + autoencoder + logistic
    # reweighting), not a graph-only score. A lift over stage2_score can
    # come from any of those three -- only claim "graph" specifically when
    # the row is graph-connected AND the illustrative (full-fit) graph
    # score is itself elevated; otherwise the honest label is "fusion".
    lifted_over_stage_1_2 = bool(final_score > stage2_score + 1e-9)
    graph_signal_elevated = bool(connected and gcn_score is not None and gcn_score > 0.5)
    escalated_by_graph = bool(lifted_over_stage_1_2 and graph_signal_elevated)
    escalated_by_fusion_other_than_graph = bool(lifted_over_stage_1_2 and not graph_signal_elevated)

    # --- Stage 6 ---
    decision = _decision(final_score, ref.t_review, ref.t_block)
    dollars = _trace_dollars(record)

    # --- Narrative ---
    lines = []
    lines.append(
        f"Trace {row['trace_id']} (customer {row['customer_id']}), "
        f"true label: {'FRAUD' if row['fraud'] == 1 else 'legitimate'}"
        + (f" [{row['attack_family']}, {row['attack_difficulty']}]" if row["fraud"] == 1 else "")
        + f", ${dollars:,.2f} moved in this trace."
    )
    if stage1_escalated:
        fired_descs = [r["rule"] for r in rules_checked if r["fired"]]
        lines.append(
            "Stage 1 (rules) escalated this trace to the ML model because: "
            + "; ".join(fired_descs) + "."
        )
    else:
        lines.append(
            "Stage 1 (rules) did NOT escalate this trace -- none of the four "
            "trigger conditions fired, so it was auto-cleared without the ML "
            "model ever running. (For reference only, the model's score had "
            f"it been escalated would have been {stage2_score_hypothetical:.3f}.)"
        )
    if stage1_escalated:
        feat_phrases = [
            f"{f['label']} ({f['value']}) pushed the score {f['pushed_score']}"
            for f in top_features[:3]
        ]
        lines.append(
            f"Stage 2 (XGBoost) scored it {stage2_score:.3f}. "
            "Top contributing signals: " + "; ".join(feat_phrases) + "."
        )
    if connected:
        ring_neighbors = [n for n in neighbors if n["is_ring"]]
        lines.append(
            f"Stage 3 (graph): this trace shares an entity (device or payee) "
            f"with {len(neighbors)} other customer(s)' traces"
            + (f", {len(ring_neighbors)} of which are part of a flagged mule ring" if ring_neighbors else "")
            + f". Illustrative (full-fit) graph-propagated score: {gcn_score:.3f}."
        )
    else:
        lines.append("Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op.")
    lines.append(
        f"Stage 5 (Risk Fusion): decision-grade out-of-fold fused score "
        f"(GCN + autoencoder + logistic reweighting) is {final_score:.3f}, "
        f"vs. {stage2_score:.3f} from Stage 1+2 alone."
        + (f" This is a graph-attributable lift (graph-connected, and the "
           f"illustrative graph score is elevated at {gcn_score:.3f})."
           if escalated_by_graph else
           " This is a lift over Stage 1+2 alone, but the illustrative "
           "graph score doesn't explain it, so the more likely driver is "
           "the autoencoder and/or the fusion model's reweighting, not "
           "the graph specifically."
           if escalated_by_fusion_other_than_graph else
           " Fusion did not raise the score above Stage 1+2 alone.")
    )
    lines.append(
        f"Stage 6 (decision policy): final score {final_score:.3f} vs. thresholds "
        f"REVIEW>={ref.t_review:.3f} / BLOCK>={ref.t_block:.3f} -> decision: {decision}."
    )
    narrative = " ".join(lines)

    return {
        "trace_id": row["trace_id"],
        "customer_id": row["customer_id"],
        "true_label": "fraud" if row["fraud"] == 1 else "legitimate",
        "attack_family": row["attack_family"],
        "attack_difficulty": row["attack_difficulty"],
        "dollars_in_trace": round(dollars, 2),
        "stage1": {"escalated": stage1_escalated, "rules_checked": rules_checked},
        "stage2": {
            "ran": stage1_escalated,
            "score": round(stage2_score, 4),
            "score_if_it_had_run_anyway": round(stage2_score_hypothetical, 4),
            "top_features": top_features,
        },
        "stage3": {
            "graph_connected": connected,
            "neighbors": neighbors,
            "graph_score": round(gcn_score, 4) if gcn_score is not None else None,
            "escalated_by_graph": escalated_by_graph,
        },
        "stage5_fusion": {
            "final_score": round(final_score, 4),
            "stage_1_2_score": round(stage2_score, 4),
            "lifted_over_stage_1_2": lifted_over_stage_1_2,
            "escalated_by_graph": escalated_by_graph,
            "escalated_by_fusion_other_than_graph": escalated_by_fusion_other_than_graph,
        },
        "stage6_decision_policy": {
            "final_score": round(final_score, 4),
            "t_review": ref.t_review, "t_block": ref.t_block,
            "decision": decision,
        },
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Step 4 -- pick a representative, illustrative set of cases
# ---------------------------------------------------------------------------
def pick_representative_cases(ref: ReferenceArtifacts) -> dict[str, int]:
    df = ref.df
    picks = {}

    def first_where(mask, label):
        idxs = np.where(mask)[0]
        if len(idxs):
            picks[label] = int(idxs[0])

    fraud = df["fraud"].values == 1
    is_ring = df["is_ring"].values.astype(bool)
    ato_easy = fraud & (df["attack_family"] == "ACCOUNT_TAKEOVER") & (df["attack_difficulty"] == "easy")
    app_hard = fraud & (df["attack_family"] == "AUTHORIZED_PUSH_PAYMENT") & (df["attack_difficulty"].isin(["hard", "advanced"]))
    blocked = ref.oof_final >= ref.t_block
    reviewed = (ref.oof_final >= ref.t_review) & (ref.oof_final < ref.t_block)
    allowed_fraud = fraud & (ref.oof_final < ref.t_review)
    legit_reviewed = (~fraud) & reviewed
    # NOTE: "final" here is oof_final, the Stage 5 FUSED score -- crossing
    # t_block relative to the Stage 1+2-only score can be driven by the
    # graph, the autoencoder, or fusion reweighting. Don't read this label
    # as "the graph rescued it" without checking explain_case()'s
    # escalated_by_graph flag for the picked trace.
    ring_rescued_by_fusion = is_ring & (ref.oof_stage_1_2 < ref.t_block) & (ref.oof_final >= ref.t_block)

    first_where(ato_easy & blocked, "easy_ato_correctly_blocked")
    first_where(app_hard & blocked, "hard_app_correctly_blocked")
    first_where(ring_rescued_by_fusion, "mule_ring_member_rescued_by_fusion")
    first_where(reviewed & fraud, "fraud_case_routed_to_review")
    first_where(allowed_fraud, "fraud_case_that_slipped_through_as_allow")
    first_where(legit_reviewed, "legitimate_case_routed_to_review")
    first_where((~fraud) & (ref.stage1_escalate == 0), "ordinary_legitimate_autocleared_by_stage1")

    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = build_reference_artifacts()

    print("\nComputing global feature importance...")
    importance = global_feature_importance(ref)
    with open(OUT_DIR / "global_feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    print(f"  saved {OUT_DIR / 'global_feature_importance.json'}")
    plot_global_shap_bar(ref, importance, OUT_DIR / "global_shap_summary.png")

    print("\nTop overall features driving Stage 2:")
    for t in importance["overall_top_features"][:8]:
        print(f"  {t['feature']:45s} mean|SHAP|={t['mean_abs_shap']}")

    print("\nBuilding representative per-case explanations...")
    picks = pick_representative_cases(ref)
    case_reports = {}
    for label, idx in picks.items():
        case_reports[label] = explain_case(ref, idx)
        print(f"  [{label}] -> trace {case_reports[label]['trace_id']}: "
        f"{case_reports[label]['stage6_decision_policy']['decision']}")

    with open(OUT_DIR / "case_reports.json", "w") as f:
        json.dump(case_reports, f, indent=2)
    print(f"  saved {OUT_DIR / 'case_reports.json'}")

    with open(OUT_DIR / "case_reports.md", "w") as f:
        f.write("# Per-case explainability reports\n\n")
        for label, rep in case_reports.items():
            f.write(f"## {label.replace('_', ' ')}\n\n")
            f.write(rep["narrative"] + "\n\n")
    print(f"  saved {OUT_DIR / 'case_reports.md'}")

    print("\nBuilding investigator case dossiers (Review/Block cases only)...")
    dossiers = {}
    for label, idx in picks.items():
        d = case_dossier(ref, idx, explanation=case_reports[label])
        if d is not None:
            dossiers[label] = d
    with open(OUT_DIR / "case_dossier_examples.md", "w") as f:
        f.write("# Case dossiers -- investigation aid only\n\n")
        f.write(
            "These summaries help a human reviewer act faster on a decision "
            "already made by decision_policy.py. They never make or override "
            "that decision.\n\n"
        )
        if not dossiers:
            f.write("No Review/Block cases among this run's representative picks.\n")
        for label, d in dossiers.items():
            f.write(f"## {label.replace('_', ' ')}\n\n")
            f.write(d["dossier_text"] + "\n\n")
    print(f"  saved {OUT_DIR / 'case_dossier_examples.md'} "
          f"({len(dossiers)} dossier(s))")

    if not picks:
        print("\nNOTE: no representative cases matched some categories -- "
              "this can happen depending on random_state/thresholds; check "
              "case_reports.json for what WAS found.")

    print(f"\nAll explainability outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()