#!/usr/bin/env python3
"""
scripts/build_v2_bundle.py -- trains the served v2 model on World B and writes portable artifacts to models/v2/.

    PowerShell: $env:PYTHONPATH="src"; python scripts/build_v2_bundle.py
Run AFTER scripts/run_v2_all.py so the model card can embed the cross-validated evaluation summary
(reports/v2/v2_results.json). Artifacts: xgb.json, novelty.npz, policy.json, drift_reference.json, entity_store.json.
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from v2.features import WORLD_B_HARD, build_world_b
from v2.phases import FEATS_B, NOV_B
from v2.serving.bundle import fit_bundle


def evaluation_summary(results_path: Path) -> dict:
    if not results_path.exists():
        return {"note": "run scripts/run_v2_all.py first to embed the evaluation summary"}
    R = json.loads(results_path.read_text(encoding="utf-8"))
    b = R["phase1"]["world_B"]
    nested = R["phase5"]["nested_world_B"]
    return {
        "source": "reports/v2/v2_results.json: customer-grouped cross-validation, SYNTHETIC World B, generator settings " + json.dumps(R.get("world_B_generator_settings")),
        "f1_mean": b["metrics"]["f1"]["mean"], "recall_mean": b["metrics"]["recall"]["mean"],
        "precision_on_synthetic_mix": b["metrics"]["precision"]["mean"],
        "precision_at_0.6pct_prevalence": b["bootstrap_ci"]["precision_at_0.6pct_prevalence"],
        "per_family_recall": b["per_family_recall"],
        "legit_flag_rate_by_kind": {k: v["fpr"] for k, v in b["fpr_by_legit_kind"].items()},
        "expected_cost_per_10k_txn": {"allow_everything": nested["allow_everything_cost_per_10k"],
                                      "xgboost_only": nested["xgb_only"]["expected_cost_per_10k_txn"],
                                      "stacked": nested["stacked"]["expected_cost_per_10k_txn"]},
        "unseen_family_recall_fixed_threshold": {f: {"supervised": v["xgb_rec"], "isolation_forest": v["iforest_rec"], "autoencoder": v["ae_rec"]}
                                                  for f, v in R["phase2"]["lofo_world_B"].items()},
    }


def main():
    dfB, recs = build_world_b(return_records=True, **WORLD_B_HARD)
    out = ROOT / "models" / "v2"
    b = fit_bundle(dfB, recs, FEATS_B, NOV_B, out, evaluation_summary=evaluation_summary(ROOT / "reports" / "v2" / "v2_results.json"))
    print("wrote", out, "version", b["policy"]["model_version"], "t_review %.3f t_block %.3f" % (b["t_review"], b["t_block"]))


if __name__ == "__main__":
    main()
