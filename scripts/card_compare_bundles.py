"""Score two card bundles (e.g. original vs retrained) on the SAME held-out synthetic test blocks. SYNTHETIC data only.
Usage: python scripts/card_compare_bundles.py OLD_DIR NEW_DIR   -> prints JSON"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from v2.card.serving import CardScorer
from v2.card.sim import simulate
from v2.card.features import build_features, FEATURES
from v2.card.evaluate import split

def metrics(sc, te):
    s = sc.booster.predict(xgb.DMatrix(te[FEATURES].to_numpy(float), feature_names=FEATURES, missing=np.nan)); y = te.is_fraud.values; fl = s >= sc.threshold
    tp = int((fl & (y == 1)).sum()); fp = int((fl & (y == 0)).sum()); fn = int((~fl & (y == 1)).sum()); tn = int((~fl & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    big = te.amount.values >= 20000
    return dict(threshold=float(sc.threshold), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12), fpr=fp / max(fp + tn, 1), pr_auc=float(average_precision_score(y, s)),
                roc_auc=float(roc_auc_score(y, s)), tp=tp, fp=fp, fn=fn, tn=tn,
                legit_ge20k_flagged=f"{int(fl[big & (y == 0)].sum())}/{int((big & (y == 0)).sum())}", fraud_ge20k_caught=f"{int(fl[big & (y == 1)].sum())}/{int((big & (y == 1)).sum())}")

old, new = CardScorer(sys.argv[1]), CardScorer(sys.argv[2]); out = {}
for name, kw in {"original_simulator_seed0_test_block": dict(), "extended_simulator_seed0_test_block": dict(extended=True)}.items():
    te = split(build_features(simulate(n_cards=1500, days=60, seed=0, events_per_pattern=40, **kw)))[2]
    out[name] = {"old_bundle": metrics(old, te), "new_bundle": metrics(new, te)}
print(json.dumps(out, indent=1))
