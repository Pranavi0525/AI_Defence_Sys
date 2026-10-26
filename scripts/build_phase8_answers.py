#!/usr/bin/env python3
"""
scripts/build_phase8_answers.py -- writes DEFENCE_MODEL_ANSWERS_PHASE8.md from YOUR reports/public/phase8_*.json.

    cmd:  python scripts\\build_phase8_answers.py

Numbers are read from the JSON files; the comparisons ("helped or not", "gap vs spread") are computed, not typed.
Try the six questions at the bottom of PHASE_8_README.md in your own words first.
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f3(x): return f"{x:.3f}"


def build(root: Path = ROOT) -> str:
    pub = root / "reports" / "public"
    for n in ("main", "control", "rolling"):
        if not (pub / f"phase8_{n}.json").exists():
            sys.exit(f"missing {pub / f'phase8_{n}.json'}; run all Phase 8 stages first")
    M, C, R = J(pub / "phase8_main.json"), J(pub / "phase8_control.json")["label_shuffle_time_split"], J(pub / "phase8_rolling.json")["rolling_origin"]
    T, Rn = M["time_split"], M["random_split"]
    tx, rx, tl = T["models"]["xgboost"], Rn["models"]["xgboost"], T["models"]["logreg"]
    ap_t, ap_r = tx["raw"]["average_precision"], rx["raw"]["average_precision"]
    lo, hi = tx["raw"]["average_precision_ci95"]
    rlo, rhi = rx["raw"]["average_precision_ci95"]
    gap = ap_r - ap_t
    aps = [f["average_precision"] for f in R if "average_precision" in f]
    spread = max(aps) - min(aps)
    ratio = gap / spread if spread else float("inf")
    size = "smaller than" if ratio < 0.75 else ("larger than" if ratio > 1.33 else "about the same size as")
    below = all(a < ap_r for a in aps)
    brier = {k: tx[k]["brier"] for k in ("raw", "sigmoid", "isotonic")}
    best = min(brier, key=brier.get)
    helped = best != "raw" and (brier["raw"] - brier[best]) / brier["raw"] >= 0.05
    lb = {k: tl[k]["brier"] for k in ("raw", "sigmoid", "isotonic")}
    lbest = min(lb, key=lb.get)
    lhelped = lbest != "raw" and (lb["raw"] - lb[lbest]) / lb["raw"] >= 0.05
    ceil = lambda S: S["fraud"]["test"] / math.ceil(0.005 * S["sizes"]["test"])
    k_t = math.ceil(0.005 * T["sizes"]["test"])
    ctrl_ratio = C["mean_average_precision"] / C["test_prevalence"]
    lr_ap = tl["raw"]["average_precision"]
    d = M["data"]
    return f"""# Phase 8 -- model answers (try the six questions in PHASE_8_README.md first)
Numbers are read from reports/public/phase8_*.json; comparisons are computed. Dataset: {d['rows']:,} transactions, {d['fraud']} frauds, about {d['time_range_s'][1] / 3600:.0f} hours.

**1. Why "method check" and not "validation of the system"? What can it not test?**
The public data has one row per transaction with anonymised PCA features. My system scores payment SESSIONS with devices, payees, addresses and phones, so this data cannot test: the graph features and entity store; the session-level features, Stage 1 rules and dossiers; the GenAI-era families (the data has none); or whether my simulator is realistic. It can only test generic modelling choices (split type, calibration, XGBoost versus a linear baseline, alert budgets). Say "a method check on a public card-fraud dataset, not the same system".

**2. What does the label-shuffle control prove, and what would make you distrust the pipeline?**
Training on shuffled labels must give average precision near the base rate, because the features then carry no information about the true labels. Result: {C['mean_average_precision']:.4f} on average (repeats {', '.join(f'{v:.4f}' for v in C['per_repeat'])}) against a base rate of {C['test_prevalence']:.4f}, a ratio of {ctrl_ratio:.1f}. {'That is close to the base rate with the noise expected from only a few dozen frauds, so there is no sign of gross label leakage.' if ctrl_ratio < 4 else 'THIS IS FAR ABOVE THE BASE RATE: look for a leak before trusting any other number.'} I would distrust the pipeline if the ratio were several times the base rate (my unit test alarms at 4x). What it does NOT rule out: subtler problems such as the same card on both sides of a split, or duplicated rows.

**3. Time vs random split: what did you see, and why can you not conclude much?**
XGBoost average precision was {f3(ap_t)} on the chronological test block ({T['fraud']['test']} frauds) and {f3(ap_r)} on the random split ({Rn['fraud']['test']} frauds), a gap of {gap:+.3f}. The bootstrap intervals are [{f3(lo)}, {f3(hi)}] and [{f3(rlo)}, {f3(rhi)}], so they {'overlap' if max(lo, rlo) <= min(hi, rhi) else 'do not overlap'}. Across the chronological blocks of the rolling-origin run, average precision ranged {f3(min(aps))} to {f3(max(aps))}, a spread of {f3(spread)}: the time-vs-random gap is {size} the block-to-block spread. All four blocks are {'below' if below else 'not all below'} the random-split value, which is {'consistent with' if below and gap > 0 else 'not consistent with'} an ordering effect. I still call it suggestive, not shown: the two test sets are different samples with different numbers of frauds, only tens of frauds per block, the bootstrap over rows ignores dependence, and two days is a short time for drift.

**4. Did calibration help or hurt on real data? Does it repeat the synthetic finding?**
XGBoost, time split, Brier: raw {brier['raw']:.5f}, sigmoid {brier['sigmoid']:.5f}, isotonic {brier['isotonic']:.5f}. Calibration {'helped' if helped else 'did not help'} XGBoost by the 5% rule; isotonic average precision was {f3(tx['isotonic']['average_precision'])} vs {f3(ap_t)} raw. {'This repeats the direction of the synthetic finding (raw XGBoost was already about as good as the calibrated versions).' if not helped else 'This differs from the synthetic finding.'} ECE (equal-frequency bins) was {tx['raw']['ece_10_quantile_bins']:.5f} raw and {tx['isotonic']['ece_10_quantile_bins']:.5f} isotonic; both are far below the base rate, so differences that small are hard to interpret and should not outweigh Brier and average precision. For logistic regression, calibration {'did' if lhelped else 'did not'} help by the 5% rule (raw {lb['raw']:.5f}, best {lbest} {lb[lbest]:.5f}). Possible reasons for any difference from the synthetic data (hypotheses, not tested): far more training rows, a {d['prevalence']:.2%} base rate, and the validation block holding only {T['fraud']['validation']} frauds to fit a calibrator on.

**5. Why recall at an alert budget instead of accuracy or F1 at a {d['prevalence']:.2%} fraud rate?**
Predicting "never fraud" is {1 - d['prevalence']:.1%} accurate, so accuracy says nothing. An alert budget matches how review teams work: they can look at a fixed share of traffic. Recall at 0.5% alerts was {f3(tx['raw']['recall_at_0.5pct_alerts'])} (time) and {f3(rx['raw']['recall_at_0.5pct_alerts'])} (random). Precision at that budget is capped: {k_t:,} alerts in the time test block can contain at most {T['fraud']['test']} frauds, so the best possible precision is {ceil(T):.3f} (time) and {ceil(Rn):.3f} (random); the observed {f3(tx['raw']['precision_at_0.5pct_alerts'])} and {f3(rx['raw']['precision_at_0.5pct_alerts'])} should be read against those ceilings, not against 1.0. Do not put these numbers next to my synthetic precision at an assumed 0.6% fraud rate as if they measured the same thing: different data, different base rate, different definition.

**6. Why was customer-grouped splitting impossible, and what does that leave unanswered?**
The file has no card or customer id (only Time, V1-V28, Amount, Class). The same card can therefore appear in both train and test, in any split. In my synthetic data customer overlap changed F1 by less than the noise, but nothing here tells us whether that holds on real data. Unanswered: how much of the score on real data comes from card-level memorisation, and everything about sessions, graphs and entity linking.

**Also be ready for: XGBoost vs logistic regression.**
Time-split average precision {f3(ap_t)} (XGBoost) vs {f3(lr_ap)} (logistic regression); the logistic value {'lies inside' if lo <= lr_ap <= hi else 'lies outside'} XGBoost's bootstrap interval, so {'it is not a clear win' if lo <= lr_ap <= hi else 'XGBoost is clearly better by this measure'}. At the very top of the ranking (0.1% alerts) recall was {f3(tx['raw']['recall_at_0.1pct_alerts'])} vs {f3(tl['raw']['recall_at_0.1pct_alerts'])}; at 0.5% it was {f3(tx['raw']['recall_at_0.5pct_alerts'])} vs {f3(tl['raw']['recall_at_0.5pct_alerts'])}.
"""


if __name__ == "__main__":
    out = ROOT / "DEFENCE_MODEL_ANSWERS_PHASE8.md"
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
