
"""Phase 10: card-transaction track (synthetic evaluation)."""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.card.sim import simulate, PATTERNS, NOVEL, EXTENDED
from v2.card.features import build_features
from v2.card.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--novel", action="store_true")
a = ap.parse_args()

runs = []

# Train and evaluate independently for every simulator seed.
for s in range(a.seeds):
    print(f"Running seed {s}...", flush=True)

    f = build_features(
        simulate(seed=s, novel=a.novel, extended=True)
    )

    r = evaluate(f, s)
    runs.append(r)

    print(f"Seed {s} completed.", flush=True)


def agg(get):
    values = np.array([get(r) for r in runs], dtype=float)
    return f"{np.nanmean(values):.3f} ± {np.nanstd(values):.3f}"


models = [
    "rules_naive",
    "rules_cp",
    "xgb",
    "mlp",
    "percard_iforest",
    "lane_xgb_or_percard",
]

patterns = PATTERNS + EXTENDED + (NOVEL if a.novel else [])

L = [
    f"# Phase 10 -- card-transaction track (SYNTHETIC; seeds={a.seeds})\n",
    "Fixed operating point: model thresholds calibrated at 1% FPR "
    "on validation legitimate traffic; lane uses 0.5% per component. "
    "Rules use fixed thresholds.\n",
    "## Time split (train first 30 days, validate 30-42, test 42-60)\n",
    "| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |",
    "|---|---|---|---|---|",
]

for m in models:
    L.append(
        f"| {m} "
        f"| {agg(lambda r, m=m: r['time_split'][m]['event_recall'])} "
        f"| {agg(lambda r, m=m: r['time_split'][m]['txn_recall'])} "
        f"| {agg(lambda r, m=m: r['time_split'][m]['fpr'])} "
        f"| {agg(lambda r, m=m: r['time_split'][m]['precision_at_0p6pct'])} |"
    )

L += [
    "\n## Unseen pattern (leave-one-pattern-out): EVENT recall\n",
    "| held-out pattern | " + " | ".join(models) + " |",
    "|---|" + "---|" * len(models),
]

for p in patterns:
    L.append(
        f"| {p} | "
        + " | ".join(
            agg(lambda r, p=p, m=m: r["lopo"][p][m]["event_recall"])
            for m in models
        )
        + " |"
    )

L.append(
    "\n(LOPO test sets contain future legitimate transactions "
    "and future fraud transactions from the held-out pattern.)\n"
)

os.makedirs("reports/v2", exist_ok=True)

suffix = "_novel" if a.novel else ""

with open(f"reports/v2/phase10_card{suffix}.json", "w") as fp:
    json.dump(runs, fp, indent=1)

with open(f"reports/v2/phase10_card{suffix}.md", "w") as fp:
    fp.write("\n".join(L))

print("\n".join(L))
print(f"\nSaved reports/v2/phase10_card{suffix}.json")
print(f"Saved reports/v2/phase10_card{suffix}.md")
