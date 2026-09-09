import json
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# AI Defense Lab — competition-facing Streamlit prototype
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Defense Lab | Payment Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "blue_team_output_FROZEN"
REPORTS = ROOT / "frozen_reports"
EXPLAIN = FROZEN / "explainability"


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_jsonl(path: Path):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return rows


def first_value(obj, names):
    """Find the first non-empty value for one of the supplied keys."""
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] not in (None, ""):
                return obj[name]
        for value in obj.values():
            found = first_value(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = first_value(value, names)
            if found not in (None, ""):
                return found
    return None


def number(obj, names):
    value = first_value(obj, names)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value):
    if value is None:
        return "—"
    if value <= 1:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}%"


def integer(value):
    if value is None:
        return "—"
    return f"{int(value):,}"


def present(path: Path):
    return path.exists()


def metric_bundle(report):
    return {
        "Precision": number(report, ["precision", "precision_score", "test_precision"]),
        "Recall": number(report, ["recall", "recall_score", "test_recall", "sensitivity"]),
        "F1": number(report, ["f1", "f1_score", "f1_macro", "test_f1"]),
        "ROC-AUC": number(report, ["roc_auc", "roc_auc_score", "auc", "test_auc"]),
        "PR-AUC": number(report, ["pr_auc", "average_precision"]),
    }


def metric_cards(report, labels=("Precision", "Recall", "F1", "ROC-AUC")):
    values = metric_bundle(report)
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            st.metric(label, pct(values.get(label)))


def compact_table(rows):
    return pd.DataFrame(rows)



def stage_status(label, available, detail="Frozen evidence loaded"):
    return {
        "Stage": label,
        "Status": "READY" if available else "MISSING",
        "Evidence": detail if available else "Artifact not found",
    }


# -----------------------------------------------------------------------------
# Frozen evidence
# -----------------------------------------------------------------------------

results = load_json(FROZEN / "results.json")
cascade = load_json(FROZEN / "three_stage_cascade_results.json")
gnn = load_json(FROZEN / "gnn_results.json")
autoencoder = load_json(FROZEN / "stage4_autoencoder_results.json")
fusion = load_json(FROZEN / "risk_fusion_results.json")
decision = load_json(REPORTS / "decision_policy_results.json")
sensitivity = load_json(REPORTS / "decision_policy_sensitivity_results.json")
rounds = load_json(FROZEN / "round1_vs_round2_report.json")
adaptive = load_json(REPORTS / "adaptive_round2_report.json")
holdout = load_json(REPORTS / "adaptive_eval_holdout.json")
hard_generation = load_json(FROZEN / "hard_example_generation_report.json")

misses = load_jsonl(REPORTS / "misses.jsonl") or load_jsonl(FROZEN / "misses.jsonl")
hard_examples = load_jsonl(FROZEN / "hard_examples.jsonl")
cases = load_json(EXPLAIN / "case_reports.json")
feature_importance = load_json(EXPLAIN / "global_feature_importance.json")

# Prefer the fully evaluated Stage 1→2→3 cascade for the headline score because
# it is explicit and independently stored in the frozen evidence.
headline_report = cascade.get("stage_1_2_3_overall", {}) if isinstance(cascade, dict) else {}
if not headline_report:
    headline_report = results




# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 0 0.6rem 0;
    }
    .hero h1 {
        font-size: 3rem;
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.15rem;
        color: #9aa4b2;
        margin-top: 0.6rem;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.30);
        margin-right: .35rem;
        font-size: .85rem;
    }
    .flow {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        background: rgba(128,128,128,.05);
    }
    .flow strong { font-size: 1.05rem; }
    .small-muted { color: #9aa4b2; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🛡️ AI Defense Lab")
    st.caption("Mastercard Innovation Challenge @ GFF 2026")

    page = st.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Red Team",
            "Blue Team",
            "Adaptive Defense",
            "Explainability",
            "Evidence",
        ],
    )

    st.divider()
    st.markdown("### Prototype status")

    key_artifacts = [
        FROZEN / "results.json",
        FROZEN / "xgb_model.joblib",
        FROZEN / "hard_examples.jsonl",
        REPORTS / "decision_policy_results.json",
        EXPLAIN / "global_shap_summary.png",
    ]
    ready = sum(path.exists() for path in key_artifacts)
    st.progress(ready / len(key_artifacts))
    st.caption(f"{ready}/{len(key_artifacts)} key artifacts detected")

    if st.button("🔄 Refresh evidence", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption("Read-only prototype")
    st.caption("Frozen evaluation artifacts are not modified by the UI.")


# -----------------------------------------------------------------------------
# Global header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI Defense Lab</h1>
        <p>Closed-loop Red Team → Generate → Blue Team → Adaptive Defense</p>
        <span class="pill">Synthetic payment-security research</span>
        <span class="pill">Frozen verified evidence</span>
        <span class="pill">Competition prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This is a controlled synthetic environment. It demonstrates attack simulation, "
    "multi-stage detection, mitigation decisions, and adaptive evaluation without "
    "connecting to real payment systems."
)


# -----------------------------------------------------------------------------
# Compact pipeline visual
# -----------------------------------------------------------------------------

st.subheader("Closed-Loop Architecture")
flow_cols = st.columns(7)
flow_items = [
    ("1", "Identify", "Attack intelligence"),
    ("2", "Generate", "Synthetic attacks"),
    ("3", "Detect", "ML + graph + anomaly"),
    ("4", "Fuse", "Unified risk"),
    ("5", "Decide", "Allow / review / block"),
    ("6", "Explain", "Why was it flagged?"),
    ("7", "Adapt", "Misses → hard examples"),
]
for col, (n, title, desc) in zip(flow_cols, flow_items):
    with col:
        st.markdown(
            f'<div class="flow"><small>Stage {n}</small><br><strong>{title}</strong>'
            f'<br><span class="small-muted">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Executive Dashboard
# -----------------------------------------------------------------------------

if page == "Executive Dashboard":
    st.divider()
    st.header("Executive Dashboard")
    st.caption("The headline view for judges: what the system does and what the frozen evaluation demonstrates.")

    # Headline performance from the explicit 1→2→3 cascade report.
    metric_cards(headline_report)

    st.caption(
        "Headline metrics are from the frozen Stage 1 → Stage 2 → Stage 3 overall evaluation."
    )

    st.divider()
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Why the cascade matters")
        if cascade:
            before = cascade.get("stage_1_2_overall", {})
            after = cascade.get("stage_1_2_3_overall", {})
            rows = []
            for label, obj in [("Stage 1 + 2", before), ("Stage 1 + 2 + 3", after)]:
                rows.append({
                    "Evaluation": label,
                    "Precision": pct(obj.get("precision")),
                    "Recall": pct(obj.get("recall")),
                    "F1": pct(obj.get("f1")),
                    "ROC-AUC": pct(obj.get("roc_auc")),
                })
            st.dataframe(compact_table(rows), hide_index=True, use_container_width=True)

            rescued = cascade.get("fraud_cases_rescued_by_stage3")
            downgraded = cascade.get("fraud_cases_downgraded_by_stage3_should_be_zero")
            a, b = st.columns(2)
            a.metric("Fraud cases rescued by graph stage", integer(rescued))
            b.metric("Fraud cases downgraded", integer(downgraded))
        else:
            st.warning("Cascade report not found in frozen evidence.")

    with right:
        st.subheader("Decision policy")
        corrected = decision.get("corrected", {}) if isinstance(decision, dict) else {}
        if corrected:
            a, b, c = st.columns(3)
            a.metric("Block rate", pct(corrected.get("block_rate")))
            b.metric("Review rate", pct(corrected.get("review_rate")))
            c.metric("Fraud blocked", integer(corrected.get("fraud_blocked")))

            d, e = st.columns(2)
            d.metric("Legitimate blocked", integer(corrected.get("legit_blocked")))
            e.metric("Fraud allowed through", integer(corrected.get("fraud_allowed")))
        else:
            st.info("Decision-policy results are not available.")

    st.divider()
    st.subheader("Closed-loop proof")
    proof = [
        stage_status("Attack generation", bool(hard_examples or hard_generation)),
        stage_status("Detection misses", bool(misses)),
        stage_status("Adaptive evaluation", bool(adaptive or holdout)),
        stage_status("Decision policy", bool(decision)),
        stage_status("SHAP explainability", (EXPLAIN / "global_shap_summary.png").exists()),
    ]
    st.dataframe(compact_table(proof), hide_index=True, use_container_width=True)

    st.subheader("Round 1 → Round 2")
    if rounds:
        meta = rounds.get("meta", {}) if isinstance(rounds, dict) else {}
        a, b, c = st.columns(3)
        a.metric("Round 1 examples", integer(meta.get("round1_n")))
        b.metric("Round 2 examples", integer(meta.get("round2_n")))
        c.metric("Hard examples loaded", integer(meta.get("hard_examples_loaded_total")))
    else:
        st.info("Round-comparison report not found.")




# -----------------------------------------------------------------------------
# Red Team
# -----------------------------------------------------------------------------

elif page == "Red Team":
    st.header("🔴 Red Team — Identify & Generate")
    st.write(
        "The Red Team creates controlled synthetic attack scenarios that can be used "
        "to stress-test the defense. The UI shows the generated evidence without exposing "
        "the underlying raw JSON by default."
    )

    a, b, c, d = st.columns(4)
    a.metric("Hard examples", integer(len(hard_examples)))
    b.metric("Observed misses", integer(len(misses)))
    c.metric("Generation evidence", "READY" if hard_generation else "—")
    d.metric("Attack trace data", "READY" if hard_examples else "—")

    st.divider()
    st.subheader("Attack-generation evidence")

    if hard_examples:
        df = pd.json_normalize(hard_examples)
        preferred = [
            c for c in [
                "attack_family", "attack_type", "scenario_id", "label",
                "difficulty", "generation_method", "source", "valid",
            ] if c in df.columns
        ]
        if preferred:
            display = df[preferred].copy()
        else:
            display = df.iloc[:, : min(8, len(df.columns))].copy()
        st.dataframe(display.head(50), use_container_width=True, hide_index=True)
        st.caption(f"Showing up to 50 of {len(df):,} frozen hard-example records.")
    else:
        st.warning("No hard-example records were found.")

    st.divider()
    st.subheader("What the Red Team contributes")
    cols = st.columns(3)
    cols[0].markdown("**Identify**\n\nSurface plausible emerging fraud behaviors and attack families.")
    cols[1].markdown("**Generate**\n\nCompose controlled synthetic scenarios and observable event traces.")
    cols[2].markdown("**Validate**\n\nUse realism, novelty, and hard-example checks before defense evaluation.")


# -----------------------------------------------------------------------------
# Blue Team
# -----------------------------------------------------------------------------

elif page == "Blue Team":
    st.header("🔵 Blue Team — Multi-Stage Detection")
    st.write(
        "The defense combines complementary signals instead of depending on a single "
        "classifier: deterministic rules, behavioral ML, graph relationships, anomaly "
        "detection, risk fusion, and a decision policy."
    )

    tabs = st.tabs(["Stage 1–2", "Stage 3 — GCN", "Stage 4 — Autoencoder", "Stage 5 — Risk Fusion", "Stage 6 — Decision"])

    with tabs[0]:
        st.subheader("Stage 1 → Stage 2 — Rule Filter + XGBoost")
        if cascade:
            before = cascade.get("stage_1_2_overall", {})
            metric_cards(before)
            st.caption("Combined Stage 1 + Stage 2 evaluation")
            cm = before.get("confusion_matrix")
            if isinstance(cm, list) and len(cm) == 2:
                st.write("Confusion matrix")
                st.dataframe(
                    pd.DataFrame(cm, index=["Legitimate", "Fraud"], columns=["Predicted legitimate", "Predicted fraud"]),
                    use_container_width=True,
                )
        elif results:
            metric_cards(results)
        else:
            st.warning("XGBoost/cascade evidence not found.")

    with tabs[1]:
        st.subheader("Stage 3 — GCN / Graph Signals")
        if cascade:
            after = cascade.get("stage_1_2_3_overall", {})
            metric_cards(after)
            a, b, c = st.columns(3)
            a.metric("Graph-connected nodes", integer(cascade.get("n_graph_connected_nodes")))
            b.metric("Ring traces", integer(cascade.get("n_ring_traces")))
            c.metric("Fraud cases rescued", integer(cascade.get("fraud_cases_rescued_by_stage3")))
            st.caption("The graph stage adds relationship-level evidence to the behavioral detector.")
        elif gnn:
            metric_cards(gnn)
        else:
            st.warning("GCN evidence not found.")

    with tabs[2]:
        st.subheader("Stage 4 — Autoencoder")
        st.caption("Anomaly / novelty signal for behavior that does not look like the learned normal world.")
        if autoencoder:
            metric_cards(autoencoder)
            available_keys = [k for k in autoencoder.keys()] if isinstance(autoencoder, dict) else []
            st.write("Available evaluation fields:", ", ".join(available_keys[:12]) or "—")
        else:
            st.info("Autoencoder artifact/report is not present in the current frozen checkout.")

    with tabs[3]:
        st.subheader("Stage 5 — Risk Fusion")
        st.caption("Detector signals are combined into a single risk representation before policy decisions.")
        if fusion:
            metric_cards(fusion)
            if isinstance(fusion, dict):
                rows = []
                for key, value in fusion.items():
                    if isinstance(value, (int, float)):
                        rows.append({"Signal": key, "Value": round(float(value), 4)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Risk-fusion report not found.")

    with tabs[4]:
        st.subheader("Stage 6 — Decision Policy")
        st.caption("Maps risk into an operational response: allow, review, or block.")
        corrected = decision.get("corrected", {}) if isinstance(decision, dict) else {}
        if corrected:
            a, b, c = st.columns(3)
            a.metric("Allow", pct(corrected.get("allow_rate")))
            b.metric("Review", pct(corrected.get("review_rate")))
            c.metric("Block", pct(corrected.get("block_rate")))

            decision_df = pd.DataFrame({
                "Action": ["Allow", "Review", "Block"],
                "Rate": [
                    corrected.get("allow_rate", 0),
                    corrected.get("review_rate", 0),
                    corrected.get("block_rate", 0),
                ],
            }).set_index("Action")
            st.bar_chart(decision_df)

            d, e = st.columns(2)
            d.metric("Fraud blocked", integer(corrected.get("fraud_blocked")))
            e.metric("Legitimate blocked", integer(corrected.get("legit_blocked")))
        else:
            st.warning("Decision-policy results not found.")



# -----------------------------------------------------------------------------
# Adaptive Defense
# -----------------------------------------------------------------------------

elif page == "Adaptive Defense":
    st.header("🔁 Adaptive Defense — Close the Loop")
    st.write(
        "The key idea is not simply to detect fraud once. Detection misses are converted "
        "into hard examples, which are then used to evaluate whether the defense becomes stronger."
    )

    a, b, c = st.columns(3)
    a.metric("Detection misses", integer(len(misses)))
    b.metric("Hard examples", integer(len(hard_examples)))
    c.metric("Round 2 evidence", "READY" if adaptive or holdout else "—")

    st.divider()
    st.subheader("Feedback loop")
    loop = st.columns(4)
    loop[0].markdown("### 1. Miss\nA defense blind spot is observed.")
    loop[1].markdown("### 2. Capture\nThe miss becomes a hard example.")
    loop[2].markdown("### 3. Re-evaluate\nThe hard example is tested against the defense.")
    loop[3].markdown("### 4. Harden\nThe next evaluation measures whether the blind spot shrinks.")

    if rounds:
        st.divider()
        st.subheader("Round 1 → Round 2")
        meta = rounds.get("meta", {}) if isinstance(rounds, dict) else {}
        a, b, c = st.columns(3)
        a.metric("Round 1", integer(meta.get("round1_n")))
        b.metric("Round 2", integer(meta.get("round2_n")))
        c.metric("Hard examples loaded", integer(meta.get("hard_examples_loaded_total")))

    if misses:
        st.divider()
        st.subheader("Miss summary")
        miss_df = pd.json_normalize(misses)
        preferred = [c for c in ["attack_family", "attack_type", "scenario_id", "final_score", "label", "reason"] if c in miss_df.columns]
        st.dataframe(
            miss_df[preferred].head(50) if preferred else miss_df.iloc[:, : min(8, len(miss_df.columns))].head(50),
            use_container_width=True,
            hide_index=True,
        )



# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------

elif page == "Explainability":
    st.header("🔎 Explainability")
    st.write(
        "A production-facing fraud decision should be inspectable. This section exposes "
        "global feature influence and case-level evidence without forcing judges to read raw JSON."
    )

    shap_path = EXPLAIN / "global_shap_summary.png"
    if shap_path.exists():
        st.subheader("Global SHAP Summary")
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("Global SHAP summary image not found in the current checkout.")

    if isinstance(feature_importance, dict):
        rows = []
        for key, value in feature_importance.items():
            if isinstance(value, (int, float)):
                rows.append({"Feature": key, "Importance": float(value)})
        if rows:
            st.subheader("Global feature importance")
            df = pd.DataFrame(rows).sort_values("Importance", ascending=False).head(20)
            st.bar_chart(df.set_index("Feature"))
            st.dataframe(df, hide_index=True, use_container_width=True)

    if cases:
        st.divider()
        st.subheader("Case-level explanations")
        if isinstance(cases, list):
            case_df = pd.json_normalize(cases)
            preferred = [c for c in ["case_id", "scenario_id", "final_score", "decision", "attack_family", "reason"] if c in case_df.columns]
            st.dataframe(
                case_df[preferred].head(50) if preferred else case_df.iloc[:, : min(8, len(case_df.columns))].head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Case report is available but is not a list-shaped table.")


# -----------------------------------------------------------------------------
# Evidence
# -----------------------------------------------------------------------------

elif page == "Evidence":
    st.header("📁 Frozen Evidence")
    st.write(
        "This is the technical appendix of the prototype. It shows which artifacts are "
        "present and where the dashboard gets its evidence."
    )

    st.info("The main dashboard is intentionally judge-facing. Raw JSON is kept out of the primary views so the prototype reads like a product rather than a debugging console.")

    files = [
        FROZEN / "results.json",
        FROZEN / "gnn_results.json",
        FROZEN / "stage4_autoencoder_results.json",
        FROZEN / "risk_fusion_results.json",
        FROZEN / "round1_vs_round2_report.json",
        FROZEN / "hard_example_generation_report.json",
        FROZEN / "hard_examples.jsonl",
        FROZEN / "xgb_model.joblib",
        FROZEN / "calibrator.joblib",
        REPORTS / "adaptive_eval_holdout.json",
        REPORTS / "adaptive_round2_report.json",
        REPORTS / "decision_policy_results.json",
        REPORTS / "decision_policy_sensitivity_results.json",
        REPORTS / "misses.jsonl",
        EXPLAIN / "global_feature_importance.json",
        EXPLAIN / "global_shap_summary.png",
        EXPLAIN / "case_reports.json",
        EXPLAIN / "case_reports.md",
    ]

    rows = []
    for path in files:
        exists = path.exists()
        rows.append({
            "Artifact": str(path.relative_to(ROOT)),
            "Status": "✓ Present" if exists else "— Missing",
            "Size": f"{path.stat().st_size:,} bytes" if exists else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Repository structure")
    st.code(
        """README.md
src/
tests/
blue_team_output_FROZEN/
frozen_reports/
web_prototype/
streamlit_app/
    app.py
""",
        language="text",
    )

    st.success(
        "The Streamlit UI is read-only with respect to the frozen evaluation artifacts."
    )
