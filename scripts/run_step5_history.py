"""Step 5: does a per-customer baseline (history features) change unseen-family recall in World B?
Usage: python scripts/run_step5_history.py [--seeds 2026 2027 2028] [--strengths 0 0.5 1]
Compares, on the SAME customers: (a) original World B, (b) history world WITHOUT history features, (c) history world WITH them."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2.gen import build_v2_records
from v2.features import extract_features_v2, graph_features, WORLD_B_HARD
from v2.history import add_history
from v2.hist_features import hist_features, HIST
from v2.evalkit import FAMILIES_V2, basic_metrics, folds_for, oof_xgb
import v2.phases as P

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 1.0]); a = ap.parse_args()

def frame(recs, with_hist):
    feats = pd.DataFrame([extract_features_v2(r) for r in recs])
    meta = pd.DataFrame([{"trace_id": r["trace_id"], "customer_id": r["customer_id"], "fraud": r["fraud"], "attack_family": r["attack_family"],
                          "source": r["source"]} for r in recs])
    meta["family"] = np.where(meta.fraud == 0, "legitimate", meta.attack_family); meta["legit_kind"] = np.where(meta.fraud == 0, meta.attack_family, "")
    df = meta.merge(feats, on="trace_id").merge(graph_features(recs), on="trace_id")
    if with_hist:
        h = pd.DataFrame([hist_features(r) for r in recs]); h.insert(0, "trace_id", [r["trace_id"] for r in recs]); df = df.merge(h, on="trace_id")
    return df.reset_index(drop=True)

def run(df, feats, nov):
    L = P.lofo_world(df, feats, nov, FAMILIES_V2)
    y = df.fraud.values; ind = [basic_metrics(y, oof_xgb(df, feats, folds_for(df, s))) for s in (11, 22)]
    return {"lofo": L, "indist_f1": float(np.mean([m["f1"] for m in ind])), "indist_recall": float(np.mean([m["recall"] for m in ind]))}

out = {}; t0 = time.time()
for seed in a.seeds:
    base = build_v2_records(seed, 900, 0.45, 90, **WORLD_B_HARD)
    dfo = frame(base, False)
    out[f"{seed}|original"] = run(dfo, P.FEATS_B, P.NOV_B); print(seed, "original", round(time.time() - t0), flush=True)
    for s in a.strengths:
        recs = add_history(base, seed=7, strength=s); df = frame(recs, True)
        out[f"{seed}|s{s}|no_hist"] = run(df, P.FEATS_B, P.NOV_B)
        out[f"{seed}|s{s}|with_hist"] = run(df, P.FEATS_B + HIST, P.NOV_B + HIST); print(seed, s, round(time.time() - t0), flush=True)
    json.dump(out, open("reports/v2/step5_history.json", "w"), indent=1)

def m(keyfmt, fam, k):
    v = [out[keyfmt.format(seed=s)]["lofo"][fam][k] for s in a.seeds]; return f"{np.mean(v):.2f}±{np.std(v):.2f}"
def mi(keyfmt, k):
    v = [out[keyfmt.format(seed=s)][k] for s in a.seeds]; return f"{np.mean(v):.3f}"
L = ["# Step 5 -- per-customer history in World B (SYNTHETIC; seeds %s)\n" % a.seeds,
     "Unseen-family recall (held-out family never in training) at the fixed operating points; `lane` = XGBoost OR IsolationForest. Format mean±sd over world seeds.\n"]
settings = [("original world (no history)", "{seed}|original")] + [(f"strength {s}, no history features", "{seed}|s%s|no_hist" % s) for s in a.strengths] + \
           [(f"strength {s}, WITH history features", "{seed}|s%s|with_hist" % s) for s in a.strengths]
for fam in FAMILIES_V2:
    L += [f"\n## Held-out: {fam}\n", "| setting | xgb recall | iforest recall | lane recall | lane false alarms | in-dist F1 |", "|---|---|---|---|---|---|"]
    for nm, kf in settings:
        L.append(f"| {nm} | {m(kf, fam, 'xgb_rec')} | {m(kf, fam, 'iforest_rec')} | {m(kf, fam, 'lane_rec')} | {m(kf, fam, 'lane_fpr')} | {mi(kf, 'indist_f1')} |")
open("reports/v2/step5_history.md", "w").write("\n".join(L)); print("\n".join(L))
