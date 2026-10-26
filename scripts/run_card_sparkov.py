"""Step 6: run the card track on the real(ish) Sparkov dataset (fraudTrain.csv / fraudTest.csv, simulated by Sparkov but with real-world geography).
Usage: python scripts/run_card_sparkov.py DIR_WITH_CSVS"""
import json, os, sys, time, warnings, tempfile
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from v2.card.features import build_features, FEATURES, RELATIVE
from v2.card.rules import rules_cp, rules_naive
from v2.card.sim import haversine_km

d = sys.argv[1]; t0 = time.time()
a = pd.read_csv(os.path.join(d, "fraudTrain.csv"), index_col=0); b = pd.read_csv(os.path.join(d, "fraudTest.csv"), index_col=0)
a["part"] = "train"; b["part"] = "test"; r = pd.concat([a, b], ignore_index=True)
df = pd.DataFrame(dict(card=r.cc_num.astype("int64"), ts=r.unix_time.astype(float), amount=r.amt, mcc=r.category, merchant=r.merchant,
                       channel=np.where(r.category.str.endswith("_net"), "ECOM", "POS"), lat=r.merch_lat, lon=r.merch_long, declined=0,
                       is_fraud=r.is_fraud.astype(int), part=r.part, home_lat=r.lat, home_lon=r.long, dob=r.dob, city_pop=r.city_pop, pattern="x", event=-1))
df["pattern"] = np.where(df.is_fraud == 1, "FRAUD", "LEGIT")
cache = os.path.join(tempfile.gettempdir(), "spk_feat.pkl")
if os.path.exists(cache): f = pd.read_pickle(cache)
else:
    f = build_features(df); del df, r, a, b
    f = f.drop(columns=["merchant", "city", "declined", "pattern", "event"], errors="ignore")
    for c in f.select_dtypes("float64").columns:
        if c not in ("ts", "lat", "lon", "home_lat", "home_lon"): f[c] = f[c].astype("float32")
    f.to_pickle(cache)
print("features", f.shape, round(time.time() - t0), "s", flush=True)
f["cat_code"] = f.mcc.astype("category").cat.codes
f["age"] = 2020 - pd.to_datetime(f.dob).dt.year
f["dist_home_decl_km"] = haversine_km(f.home_lat, f.home_lon, f.lat, f.lon)
EXTRA = ["cat_code", "age", "city_pop", "dist_home_decl_km"]; F = FEATURES + EXTRA
REL = RELATIVE + ["dist_home_decl_km"]
tr_all = f[f.part == "train"]; te = f[f.part == "test"]
cut = tr_all.ts.quantile(0.85); tr = tr_all[tr_all.ts < cut]; va = tr_all[tr_all.ts >= cut]
print("train", len(tr), tr.is_fraud.sum(), "val", len(va), va.is_fraud.sum(), "test", len(te), te.is_fraud.sum(), flush=True)
rng = np.random.default_rng(0)
trs = pd.concat([tr[tr.is_fraud == 1], tr[tr.is_fraud == 0].sample(250000, random_state=0)])           # legit subsample for speed
res, sc = {}, {}
xg = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=0).fit(trs[F], trs.is_fraud)
sc["xgb"] = (xg.predict_proba(va[F])[:, 1], xg.predict_proba(te[F])[:, 1]); print("xgb", round(time.time() - t0), flush=True)
xg2 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, tree_method="hist", n_jobs=1, random_state=0).fit(trs[FEATURES], trs.is_fraud)
sc["xgb_no_catcode"] = (xg2.predict_proba(va[FEATURES])[:, 1], xg2.predict_proba(te[FEATURES])[:, 1])
mlp = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                    MLPClassifier((64, 32), max_iter=60, early_stopping=True, random_state=0)).fit(trs[F], trs.is_fraud)
sc["mlp"] = (mlp.predict_proba(va[F])[:, 1], mlp.predict_proba(te[F])[:, 1]); print("mlp", round(time.time() - t0), flush=True)
leg = tr[tr.is_fraud == 0].sample(120000, random_state=0); imp = SimpleImputer(strategy="constant", fill_value=0).fit(leg[REL])
iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=1).fit(imp.transform(leg[REL]))
sc["percard_iforest"] = (-iso.score_samples(imp.transform(va[REL])), -iso.score_samples(imp.transform(te[REL])))
y = te.is_fraud.values; prev = y.mean()
def row(flag, s=None):
    tpr = flag[y == 1].mean(); fpr = flag[y == 0].mean()
    o = dict(recall=float(tpr), fpr=float(fpr), precision=float(flag[flag].size and y[flag].mean()))
    if s is not None: o.update(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)))
    return o
for nm, (sv, st) in sc.items():
    for fpr in (0.01, 0.001):
        t = np.quantile(sv[va.is_fraud.values == 0], 1 - fpr)
        res[f"{nm}@{fpr:.1%}fpr"] = row(st >= t, st)
res["rules_cp"] = row(rules_cp(te).values == 1); res["rules_naive"] = row(rules_naive(te).values == 1)
res["_meta"] = dict(test_rows=int(len(te)), test_fraud=int(y.sum()), prevalence=float(prev), seconds=round(time.time() - t0))
imp_ = pd.Series(xg.feature_importances_, F).sort_values(ascending=False).head(10).round(3).to_dict(); res["_xgb_top_features"] = imp_
os.makedirs("reports/v2", exist_ok=True); json.dump(res, open("reports/v2/phase10_sparkov.json", "w"), indent=1)
print(json.dumps(res, indent=1))
