#!/usr/bin/env python3
"""
scripts/run_v2_all.py -- runs the whole v2 evaluation (Phases 1,2,3,4,5) and writes reports/v2/v2_results.json + v2_report.md.
Every number in the report is generated here.

    PowerShell:  $env:PYTHONPATH="src"; python scripts/run_v2_all.py
    bash:        PYTHONPATH=src python3 scripts/run_v2_all.py        (about 6-10 minutes; or one stage at a time:
                 --stage data | p1 | p2p4 | p5a | p5b | p5c | report)
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn, xgboost

from v2 import audit, fusion, phases as P
from v2.evalkit import BEHAVIOUR, FAMILIES_V1, FAMILIES_V2, GRAPH, NEW_SCHEMA, basic_metrics, folds_for, oof_xgb
from v2.features import WORLD_B_HARD, build_pooled, build_world_b, load_v1_records

OUT = ROOT / "reports" / "v2"
OUT.mkdir(parents=True, exist_ok=True)


def timed(name, fn, *a, **k):
    t = time.time(); r = fn(*a, **k); print(f"  [{name}] {time.time() - t:.0f}s", flush=True); return r


CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _save(name, obj):
    (CACHE / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _load(name):
    return json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))


def _worlds():
    df = pd.read_pickle(CACHE / "pooled.pkl")
    return (df,) + P.worlds(df)


def stage_data():
    print("Building pooled dataset...", flush=True)
    df = build_pooled(ROOT, **WORLD_B_HARD)
    df.to_pickle(CACHE / "pooled.pkl"); df.to_csv(OUT / "pooled_features.csv", index=False)
    A, B = P.worlds(df)
    recs = load_v1_records(ROOT)
    _save("data_audit", {
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "world_B_generator_settings": WORLD_B_HARD,
        "data": {"world_A_rows": len(A), "world_A_fraud": int(A.fraud.sum()), "world_B_rows": len(B),
                 "world_B_fraud": int(B.fraud.sum()), "families_A": FAMILIES_V1, "families_B": FAMILIES_V2},
        "audit": {"v1_raw_fields": audit.raw_field_audit(recs),
                  "world_A_top_feature_auc": audit.feature_auc_audit(A, P.FEATS_A),
                  "world_B_top_feature_auc": audit.feature_auc_audit(B, P.FEATS_B),
                  "legit_simulator_distinguishability_auc": audit.simulator_distinguishability(df, BEHAVIOUR)}})


def stage_p1():
    df, A, B = _worlds()
    r = {"world_A": timed("P1 A", P.phase1_world, A, P.FEATS_A, FAMILIES_V1),
         "world_B": timed("P1 B", P.phase1_world, B, P.FEATS_B, FAMILIES_V2)}
    sweep = {}
    for hs in (0.0, 0.45, 0.8):
        Bh = build_world_b(hard_share=hs, **WORLD_B_HARD); y = Bh.fraud.values; row = {}
        for name, feats in [("behaviour_only", BEHAVIOUR), ("behaviour+schema+graph", P.FEATS_B)]:
            ms = [basic_metrics(y, oof_xgb(Bh, feats, folds_for(Bh, s))) for s in P.SEEDS]
            row[name] = {k: float(np.mean([m[k] for m in ms])) for k in ("f1", "recall", "fp", "fn")}
        sweep[str(hs)] = row
    r["hard_negative_sweep_world_B"] = sweep
    _save("phase1", r)


def stage_p2p4():
    df, A, B = _worlds()
    _save("phase2", {"lofo_world_B": timed("LOFO B", P.lofo_world, B, P.FEATS_B, P.NOV_B, FAMILIES_V2),
                     "lofo_world_A": timed("LOFO A", P.lofo_world, A, P.FEATS_A, P.NOV_A, FAMILIES_V1),
                     "transfer_A_to_B": P.transfer_A_to_B(A, B)})
    _save("phase4", {"world_A": P.graph_ablation(A, BEHAVIOUR, P.FEATS_A, FAMILIES_V1),
                     "world_B": P.graph_ablation(B, BEHAVIOUR + NEW_SCHEMA, P.FEATS_B, FAMILIES_V2)})


def stage_p5a():
    df, A, B = _worlds()
    _save("phase5_nested", {"nested_world_B": timed("nested B", fusion.run_nested, B, P.FEATS_B, P.NOV_B),
                            "nested_world_A": timed("nested A", fusion.run_nested, A, P.FEATS_A, P.NOV_A)})


def stage_p5b():
    df, A, B = _worlds()
    rows = []
    for s in (11, 22):
        for r in timed(f"unseen B seed{s}", fusion.run_unseen_family_policy, B, P.FEATS_B, P.NOV_B, FAMILIES_V2, seed=s):
            rows.append({**r, "world": "B", "seed": s})
    _save("phase5_unseen_B", rows)


def stage_p5c():
    df, A, B = _worlds()
    rows = [{**r, "world": "A", "seed": 11} for r in
            timed("unseen A", fusion.run_unseen_family_policy, A, P.FEATS_A, P.NOV_A, FAMILIES_V1, seed=11)]
    _save("phase5_unseen_A", rows)


def stage_report():
    R = _load("data_audit")
    R["phase1"] = _load("phase1"); R["phase2"] = _load("phase2"); R["phase4"] = _load("phase4")
    R["phase5"] = _load("phase5_nested")
    R["phase5"]["unseen_family_policy"] = _load("phase5_unseen_B") + _load("phase5_unseen_A")
    (OUT / "v2_results.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
    write_report(R)
    print("Wrote", OUT / "v2_results.json", "and v2_report.md")


STAGES = {"data": stage_data, "p1": stage_p1, "p2p4": stage_p2p4, "p5a": stage_p5a, "p5b": stage_p5b,
          "p5c": stage_p5c, "report": stage_report}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + list(STAGES))
    a = ap.parse_args()
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"== stage {name}", flush=True)
        STAGES[name]()


def pct(x): return f"{100 * x:.1f}%"


def write_report(R):
    L = ["# AI Defence System v2 - generated evaluation report", "",
         "All numbers are generated by `scripts/run_v2_all.py`. SYNTHETIC data: nothing here is a real-world performance claim.",
         f"Environment: {R['environment']}", "",
         "## Data",
         f"- World A (v1 Red Team attacks vs Normal World legit): {R['data']['world_A_rows']} traces, {R['data']['world_A_fraud']} fraud.",
         f"- World B (v2 generator: hard-negative legit vs 4 GenAI-era families): {R['data']['world_B_rows']} traces, {R['data']['world_B_fraud']} fraud. Generator settings: {R.get('world_B_generator_settings')}",
         "- World B's generators are my own lightweight, assumption-driven simulations, not validated by the Red Team validators.", "",
         "## Artifact audit",
         "Raw v1 fields where legit and fraud value sets are DISJOINT (perfect label proxies for any model that reads them):"]
    for r in R["audit"]["v1_raw_fields"]:
        L.append(f"- `{r['field']}`: disjoint={r['disjoint_value_sets']} | legit {r['legit_values']} | fraud {r['fraud_values']}")
    L += ["", f"Legit-only simulator distinguishability (v1 vs v2 legit, AUC): {R['audit']['legit_simulator_distinguishability_auc']:.3f}"
          " (near 1.0 = large domain gap; cross-world transfer numbers are confounded by it).", ""]
    for w in ("A", "B"):
        p = R["phase1"][f"world_{w}"]; m = p["metrics"]
        L += [f"## Phase 1 - World {w}: grouped-CV baseline (mean over 3 fold seeds)",
              f"- F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f}, recall {m['recall']['mean']:.3f}, precision {m['precision']['mean']:.3f}, "
              f"PR-AUC {m['pr_auc']['mean']:.3f}, FP {m['fp']['mean']:.1f}, FN {m['fn']['mean']:.1f}",
              "- Per-family recall: " + ", ".join(f"{k} {v:.3f}" for k, v in p["per_family_recall"].items()),
              "- FPR by legit kind: " + ", ".join(f"{k} {v['fpr']:.3f} (n={v['n']})" for k, v in p["fpr_by_legit_kind"].items())]
        s1 = p["stage1_in_fold"]
        L += ["- Stage 1 learned in-fold (depth-3 tree; target = share of training fraud it must keep): " + "; ".join(
            f"keep {k}: auto-clears {pct(v['legit_auto_cleared_share'])} of legit, loses {v['fraud_wrongly_cleared']}/{v['fraud_total']} fraud, cascade F1 {v['cascade_metrics']['f1']:.3f}"
            for k, v in s1["by_fraud_keep_target"].items()) + f"; XGBoost alone F1 {s1['xgb_alone_metrics']['f1']:.3f}"]
        L += ["- Calibration (outer folds): " + "; ".join(f"{k}: Brier {v['brier']:.4f}, ECE {v['ece']:.4f}" for k, v in p["calibration"].items())]
        b = p["bootstrap_ci"]
        L += ["- Customer-cluster bootstrap 95% CI: " + "; ".join(f"{k} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in b.items()), ""]
    L += ["## Phase 1 - hard-negative sweep (World B; F1 / FN by share of hard-negative legit)", "| hard share | behaviour-only F1 | all-features F1 |", "|---|---|---|"]
    for hs, v in R["phase1"]["hard_negative_sweep_world_B"].items():
        L.append(f"| {hs} | {v['behaviour_only']['f1']:.3f} | {v['behaviour+schema+graph']['f1']:.3f} |")
    L += ["", "## Phase 2 - unseen-family recall at fixed operating points (leave-one-family-out)",
          "xgb = supervised at 0.5; iforest/ae = novelty detectors trained on legit only, flagging ~2% of training legit; lane = xgb OR iforest.", ""]
    for w, key in (("B", "lofo_world_B"), ("A", "lofo_world_A")):
        L += [f"World {w}:", "| held-out family | xgb | iforest | ae | lane | lane FPR |", "|---|---|---|---|---|---|"]
        for f, v in R["phase2"][key].items():
            L.append(f"| {f} | {v['xgb_rec']:.3f} | {v['iforest_rec']:.3f} | {v['ae_rec']:.3f} | {v['lane_rec']:.3f} | {v['lane_fpr']:.3f} |")
        L.append("")
    L += ["Transfer (train on World A, test on World B; common features only):", "| detector | FPR on B legit | " + " | ".join(FAMILIES_V2) + " |", "|---|---|" + "---|" * 4]
    for k, v in R["phase2"]["transfer_A_to_B"].items():
        L.append(f"| {k} | {v['fpr_on_B_legit']:.3f} | " + " | ".join(f"{v[f]:.3f}" for f in FAMILIES_V2) + " |")
    L += ["", "## Phase 4 - graph features ablation (grouped CV)"]
    for w in ("A", "B"):
        g = R["phase4"][f"world_{w}"]
        L.append(f"- World {w}: no graph F1 {g['no_graph']['f1']:.3f} (FP {g['no_graph']['fp']:.1f}, FN {g['no_graph']['fn']:.1f}) -> with graph F1 {g['with_graph']['f1']:.3f} (FP {g['with_graph']['fp']:.1f}, FN {g['with_graph']['fn']:.1f})")
    L += ["", "## Phase 5 - nested fusion and cost-aware policy",
          f"Assumptions: {R['phase5']['nested_world_B']['assumptions']}", ""]
    for w in ("B", "A"):
        n = R["phase5"][f"nested_world_{w}"]
        L += [f"World {w} (in-distribution, outer-test only; shares at assumed production prevalence):",
              "| strategy | allow | review | block | fraud recall (review+block) | block precision | cost per 10k txn |", "|---|---|---|---|---|---|---|"]
        for s in ("xgb_only", "stacked", "naive_max"):
            x = n[s]; t = x["traffic_share_at_assumed_prevalence"]
            L.append(f"| {s} | {pct(t['allow'])} | {pct(t['review'])} | {pct(t['block'])} | {pct(x['fraud_recall_review_plus_block'])} | {pct(x['block_precision_at_assumed_prevalence'])} | {x['cost_per_10k'] if False else x['expected_cost_per_10k_txn']:.0f} |")
        L += [f"(allow-everything cost per 10k: {n['allow_everything_cost_per_10k']:.0f})", ""]
    L += ["Policy threshold sensitivity (World B, stacked; mean/std over outer folds): see v2_results.json `policy_sensitivity_stacked`.", "",
          "### Unseen-family policy test (cost per 10k txn if the next fraud wave is the held-out family)",
          "| world | held-out family | policy | fraud flagged | legit flagged | cost/10k |", "|---|---|---|---|---|---|"]
    df_u = pd.DataFrame(R["phase5"]["unseen_family_policy"]).groupby(["world", "held_out_family", "policy"], as_index=False).mean(numeric_only=True)
    for _, r in df_u.iterrows():
        L.append(f"| {r['world']} | {r['held_out_family']} | {r['policy']} | {pct(r['recall_flag'])} | {pct(r['legit_flag'])} | {r['cost']:.0f} |")
    sw = OUT / "stealth_sweep.md"
    if sw.exists():
        L += ["", "## Stealth sweep (from scripts/v2_stealth_sweep.py)", "", sw.read_text(encoding="utf-8")]
    (OUT / "v2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
