"""Train and export the served card-fraud model (SYNTHETIC simulator data, fixed seed).

Usage (from the repo root):
    python scripts/train_card_model.py [--seed 0] [--out models/card] [--fpr 0.01] [--force]

Writes models/card/{xgb.json,policy.json}. Refuses to overwrite an existing bundle unless --force is given.
Does not touch any payment-fraud artifact (models/, models/v2/, blue_team_output/).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2.card.train import train_bundle, DEFAULT_FPR, DEFAULT_SEED

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "card"))
ap.add_argument("--fpr", type=float, default=DEFAULT_FPR)
ap.add_argument("--force", action="store_true", help="overwrite an existing card bundle")
a = ap.parse_args()

out = os.path.abspath(a.out)
if os.path.exists(os.path.join(out, "xgb.json")) and not a.force:
    sys.exit(f"[train_card_model] {out} already holds a bundle; pass --force to overwrite it (existing artifacts are never replaced silently).")
p = train_bundle(out, seed=a.seed, fpr=a.fpr)
ev = p["evaluation"]["at_threshold"]
print(json.dumps({"model_version": p["model_version"], "threshold": p["threshold"], "test_event_recall": p["evaluation"]["event_recall"],
                  "test_at_threshold": ev, "note": "SYNTHETIC simulator data -- not real-world evidence"}, indent=1))
