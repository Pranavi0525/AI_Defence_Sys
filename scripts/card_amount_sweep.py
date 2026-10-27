"""Monotonicity / saturation diagnostic: score vs amount with every other input held fixed (SYNTHETIC model; INR).

Usage: python scripts/card_amount_sweep.py [--bundle models/card] [--old DIR_WITH_OLD_BUNDLE] [--out reports/card/amount_sweep.md]
The scenario history is 7 ordinary POS purchases (INR 380-1500) in Chennai. The sweep bypasses the business limit on purpose
(CARD_MAX_TXN_AMOUNT is raised for this script only) so the raw model behaviour above the limit is visible.
Non-monotonic behaviour is NOT a bug by itself: it is a tree model on interacting features, and we do not enforce monotonicity.
"""
import argparse, os, sys, time
os.environ["CARD_MAX_TXN_AMOUNT"] = "1e12"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2.card.serving import CardScorer

DAY = 86400; T0 = 20 * DAY + 10 * 3600
M = ["M5411_3", "M5812_2", "M5411_3", "M5411_1", "M5812_2", "M5411_3", "M5541_4"]; MCC = [5411, 5812, 5411, 5411, 5812, 5411, 5541]; AMT = [640, 380, 910, 455, 520, 700, 1500]
hist = [dict(card="C-1001", ts=T0 + i * DAY + i * 1237, amount=AMT[i], mcc=MCC[i], channel="POS", lat=13.0827, lon=80.2707, merchant=M[i], declined=0) for i in range(7)]
def tx(a, **kw):
    d = dict(card="C-1001", ts=T0 + 7 * DAY + 900, amount=a, mcc=5411, channel="ECOM", lat=13.0827, lon=80.2707, merchant="M5411_3", declined=0); d.update(kw); return d
CTX = {"A familiar ECOM": lambda a: tx(a),
       "B new merchant + risky MCC, home": lambda a: tx(a, mcc=5947, merchant="M5947_fx3"),
       "C new merchant + risky MCC, Mumbai": lambda a: tx(a, mcc=5947, merchant="M5947_fx3", lat=19.076, lon=72.8777),
       "D familiar POS": lambda a: tx(a, channel="POS")}
AMTS = [100, 560, 2000, 5000, 10000, 20000, 50000, 1e5, 3e5, 1e6, 1e8, 1e10]

ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default="models/card"); ap.add_argument("--old", default=None); ap.add_argument("--out", default="reports/card/amount_sweep.md")
a = ap.parse_args()
def table(sc, title):
    L = [f"### {title}", f"threshold = {sc.threshold:.5f}; `*` = FRAUD", "", "| amount (INR) | " + " | ".join(CTX) + " |", "|---:|" + "---:|" * len(CTX)]
    for amt in AMTS:
        cells = []
        for f in CTX.values():
            r = sc.score_one(f(amt), hist); cells.append(f"{r['model_score']:.2e}{'*' if r['is_fraud'] else ''}")
        L.append(f"| {amt:,.0f} | " + " | ".join(cells) + " |")
    return L
out = ["# Amount sweep (synthetic model; all other inputs fixed)", ""]
new = CardScorer(a.bundle)
if a.old: out += table(CardScorer(a.old), "BEFORE (original Phase 10 bundle)") + [""]
out += table(new, "AFTER (retrained bundle)")
ts = []
for _ in range(30):
    t = time.perf_counter(); new.score_one(tx(560), hist); ts.append((time.perf_counter() - t) * 1000)
ts.sort(); out += ["", f"Latency of score_one with 7 history rows (30 runs, this sandbox): median {ts[15]:.1f} ms, p90 {ts[27]:.1f} ms."]
sup = new.amount_support
if sup: out += ["", f"Training amount range: max {sup['train_max_amount']:,.0f} INR (legit {sup['train_max_amount_legit']:,.0f}, fraud {sup['train_max_amount_fraud']:,.0f}). "
                "Scores are flat above the largest split the trees learned; the API rejects amounts above the business limit instead of extrapolating."]
os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
