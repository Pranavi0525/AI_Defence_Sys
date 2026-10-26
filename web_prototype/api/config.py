"""
api/config.py
==============
Single source of truth for:
  - where the repo root is (so this works regardless of cwd uvicorn was
    launched from)
  - which file on disk is the CANONICAL, current output for each pipeline
    stage (several stages have an older root-level copy left over from
    earlier runs -- those are NOT what the scripts write to today, see
    the audit notes below, so the API always reads the canonical path)
  - the ordered list of stages the "run full pipeline" endpoint executes

Canonical output locations (verified against each script's own
out_dir / OUT_DIR / open() calls, not assumed):

    blue_team_pipeline.py        -> blue_team_output/results.json,
                                     blue_team_output/misses.jsonl (legacy,
                                     single-stage CV misses),
                                     blue_team_output/xgb_model.joblib,
                                     blue_team_output/calibrator.joblib
    cascade_with_graph.py        -> blue_team_output/three_stage_cascade_results.json
    cascade_with_autoencoder.py  -> blue_team_output/stage4_autoencoder_results.json
    risk_fusion.py                -> blue_team_output/risk_fusion_results.json
    decision_policy.py            -> decision_policy_results.json (repo root)
                                     decision_policy_validation_cache.npz (repo root)
    decision_policy_sensitivity.py-> decision_policy_sensitivity_results.json (repo root)
    explainability.py             -> blue_team_output/explainability/case_reports.json
                                     blue_team_output/explainability/case_reports.md
                                     blue_team_output/explainability/global_feature_importance.json
                                     blue_team_output/explainability/global_shap_summary.png
    miss_collector.py             -> misses.jsonl (repo root) -- the REAL
                                     full-cascade misses (5), distinct from
                                     the legacy blue_team_output/misses.jsonl
    adaptive_feedback_loop.py     -> adaptive_round2_report.json (repo root)
                                     blue_team_output/hard_examples.jsonl
    retrain_round2.py             -> blue_team_output/round1_vs_round2_report.json
                                     adaptive_eval_holdout.json (repo root)

Root-level copies of case_reports.json / global_feature_importance.json /
global_shap_summary.png / three_stage_cascade_results.json /
stage4_autoencoder_results.json that also exist in this repo are stale
manual copies from an earlier run -- the API deliberately ignores them.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
OUT_DIR = REPO_ROOT / "blue_team_output"
EXPLAIN_DIR = OUT_DIR / "explainability"

# ---------------------------------------------------------------------------
# Canonical report file paths (dashboard reads these live, every request)
# ---------------------------------------------------------------------------
REPORTS = {
    "stage1_2_results": OUT_DIR / "results.json",
    "stage3_graph_results": OUT_DIR / "three_stage_cascade_results.json",
    "stage4_autoencoder_results": OUT_DIR / "stage4_autoencoder_results.json",
    "risk_fusion_results": OUT_DIR / "risk_fusion_results.json",
    "decision_policy_results": REPO_ROOT / "decision_policy_results.json",
    "decision_policy_sensitivity_results": REPO_ROOT / "decision_policy_sensitivity_results.json",
    "case_reports": EXPLAIN_DIR / "case_reports.json",
    "global_feature_importance": EXPLAIN_DIR / "global_feature_importance.json",
    "global_shap_summary_png": EXPLAIN_DIR / "global_shap_summary.png",
    "misses": REPO_ROOT / "misses.jsonl",
    "adaptive_round2_report": REPO_ROOT / "adaptive_round2_report.json",
    "adaptive_eval_holdout": REPO_ROOT / "adaptive_eval_holdout.json",
    "round1_vs_round2_report": OUT_DIR / "round1_vs_round2_report.json",
    "hard_examples": OUT_DIR / "hard_examples.jsonl",
    "hard_example_generation_report": REPO_ROOT / "hard_example_generation_report.json",
}

# ---------------------------------------------------------------------------
# Ordered pipeline stages -- what "run full pipeline" executes, in order.
# Each script is run as its own subprocess (matches how it's documented to
# be run: `PYTHONPATH=src python3 <script>` from the repo root) so we don't
# have to fight each file's own argparse / __main__ / sys.exit assumptions.
# ---------------------------------------------------------------------------
STAGES = [
    {
        "id": "stage1_2",
        "label": "Stage 1+2: Rules + XGBoost",
        "script": "blue_team_pipeline.py",
        "outputs": ["stage1_2_results"],
        "est_seconds": 60,
    },
    {
        "id": "stage3_graph",
        "label": "Stage 3: GCN Graph Escalation",
        "script": "cascade_with_graph.py",
        "outputs": ["stage3_graph_results"],
        "est_seconds": 150,
    },
    {
        "id": "stage4_autoencoder",
        "label": "Stage 4: Autoencoder Novelty",
        "script": "cascade_with_autoencoder.py",
        "outputs": ["stage4_autoencoder_results"],
        "est_seconds": 150,
    },
    {
        "id": "stage5_fusion",
        "label": "Stage 5: Risk Fusion",
        "script": "risk_fusion.py",
        "outputs": ["risk_fusion_results"],
        "est_seconds": 90,
    },
    {
        "id": "decision_policy",
        "label": "Decision Policy (ALLOW/REVIEW/BLOCK)",
        "script": "decision_policy.py",
        "outputs": ["decision_policy_results"],
        "est_seconds": 30,
    },
    {
        "id": "decision_policy_sensitivity",
        "label": "Decision Policy Sensitivity Analysis",
        "script": "decision_policy_sensitivity.py",
        "outputs": ["decision_policy_sensitivity_results"],
        "est_seconds": 20,
    },
    {
        "id": "explainability",
        "label": "Explainability (SHAP)",
        "script": "explainability.py",
        "outputs": ["case_reports", "global_feature_importance", "global_shap_summary_png"],
        "est_seconds": 45,
    },
    {
        "id": "miss_collector",
        "label": "Collect Full-Cascade Misses",
        "script": "miss_collector.py",
        "outputs": ["misses"],
        "est_seconds": 30,
    },
    {
        "id": "adaptive_feedback_loop",
        "label": "Adaptive Feedback Loop",
        "script": "adaptive_feedback_loop.py",
        "outputs": ["adaptive_round2_report", "hard_examples"],
        "est_seconds": 60,
    },
    {
        "id": "retrain_round2",
        "label": "Retrain Round 2 (Adaptive)",
        "script": "retrain_round2.py",
        "outputs": ["round1_vs_round2_report", "adaptive_eval_holdout"],
        "est_seconds": 90,
    },
]

STAGE_IDS = [s["id"] for s in STAGES]
