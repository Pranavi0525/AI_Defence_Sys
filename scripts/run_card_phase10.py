"""Phase 10: card-transaction track. Usage: python scripts/run_card_phase10.py [--seeds 3]"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2.card.sim import simulate, PATTERNS, NOVEL
from v2.card.features import build_features
from v2.card.evaluate import evaluate

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--novel", action="store_true"); a = ap.parse_args()
runs = []
for s in range(a.seeds):
    f = build_features(simulate(seed=s, novel=a.novel)); r = evaluate(f, s); runs.append(r); print("seed", s, "done", flush=True)
os.makedirs("reports/v2", exist_ok=True)
json.dump(runs, open("reports/v2/phase10_card%s.json" % ("_novel" if a.novel else ""), "w"), indent=1)

def agg(get):
    v = np.array([get(r) for r in runs], float); return f"{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}"

models = ["rules_naive", "rules_cp", "xgb", "mlp", "percard_iforest", "lane_xgb_or_percard"]
L = ["# Phase 10 -- card-transaction track (SYNTHETIC; seeds=%d, mean ± sd over simulator seeds)\n" % a.seeds,
     "Fixed operating point: model thresholds set at 1% false-positive rate on a validation block of legitimate traffic (lane: 0.5% each). Rules have fixed thresholds.\n",
     "## Time split (train first 30 days, validate 30-42, test 42-60)\n", "| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |", "|---|---|---|---|---|"]
for m in models:
    L.append(f"| {m} | {agg(lambda r: r['time_split'][m]['event_recall'])} | {agg(lambda r: r['time_split'][m]['txn_recall'])} | "
             f"{agg(lambda r: r['time_split'][m]['fpr'])} | {agg(lambda r: r['time_split'][m]['precision_at_0p6pct'])} |")
L += ["\n## Unseen pattern (leave-one-pattern-out): EVENT recall on the held-out pattern\n", "| held-out pattern | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
for p in PATTERNS + (NOVEL if a.novel else []):
    L.append(f"| {p} | " + " | ".join(agg(lambda r, m=m: r['lopo'][p][m]['event_recall']) for m in models) + " |")
L.append("\n(legit FPR for each model in the LOPO runs is the same as in the time split.)\n")
open("reports/v2/phase10_card%s.md" % ("_novel" if a.novel else ""), "w").write("\n".join(L)); print("\n".join(L))
