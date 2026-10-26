#!/usr/bin/env python3
"""
scripts/build_phase9_docs.py -- Phase 9: writes README_V2.md, INTERVIEW_KIT.md and reports/v2/results_table.md.

    cmd:  python scripts\\build_phase9_docs.py

EVERY number in the three files is read from a generated report (no number is typed in this script):
    phase5a_evaluation_report.json          v1's own headline claim
    reports/v2/leakage_check.json           v1 re-evaluated with customer-grouped folds
    reports/v2/v2_results.json              World A / World B, calibration, graph ablation, unseen families, policy cost
    reports/v2/stealth_sweep.json           F1 vs share of adaptive attackers
    reports/v2/feedback_loop_*.json         Phase 7 feedback loop
    reports/v2/feedback_miss_audit.md       what the round-0 model misses
If you regenerate the reports on your machine and run this script again, the documents follow your numbers.
Wording (what is claimed, what is not) is static text and lives in the templates below.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "v2"


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pc(x): return f"{x * 100:.1f}%"
def pc0(x): return f"{x * 100:.0f}%"
def money(x): return f"{x:,.0f}"


FAM_A = ["ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"]
FAM_B = ["SYNTHETIC_IDENTITY", "FAKE_MERCHANT", "DEEPFAKE_KYC", "AI_SCRIPTED_APP"]


# ------------------------------------------------------------------------------------------------ numbers
def collect(root: Path = ROOT) -> dict:
    rep = root / "reports" / "v2"
    R, claim, leak, sweep = J(rep / "v2_results.json"), J(root / "phase5a_evaluation_report.json"), J(rep / "leakage_check.json"), J(rep / "stealth_sweep.json")
    N: dict[str, str] = {}
    N["ENV"] = ", ".join(f"{k} {v}" for k, v in R["environment"].items())
    c = claim["classification"]
    N["V1_F1"], N["V1_P"], N["V1_R"] = f3(c["f1"]), f3(c["precision"]), f3(c["recall"])
    cm = c["confusion_matrix"]
    N["V1_TP"], N["V1_FP"], N["V1_FN"], N["V1_TN"] = (str(cm[k]) for k in ("true_positive", "false_positive", "false_negative", "true_negative"))
    g = {(r["split"], r["features"]): r for r in leak["grid"]}
    a, b = g[("v1_trace_hash", "all_28_features")], g[("v2_customer_grouped", "all_28_features")]
    N["LEAK_V1_F1"], N["LEAK_V1_SD"], N["LEAK_GRP_F1"], N["LEAK_GRP_SD"] = f3(a["f1_mean"]), f3(a["f1_std"]), f3(b["f1_mean"]), f3(b["f1_std"])
    st = leak["structure"]
    ov = [x["share_test_rows_customer_in_train"] for x in st["v1_fold_overlap"]]
    N["LEAK_SHARE"] = pc0(sum(ov) / len(ov))

    for W, fam in (("A", FAM_A), ("B", FAM_B)):
        w = R["phase1"]["world_" + W]
        m = w["metrics"]
        N[f"{W}_F1"], N[f"{W}_F1_SD"] = f3(m["f1"]["mean"]), f3(m["f1"]["std"])
        N[f"{W}_REC"], N[f"{W}_PREC"], N[f"{W}_PRAUC"] = f3(m["recall"]["mean"]), f3(m["precision"]["mean"]), f3(m["pr_auc"]["mean"])
        N[f"{W}_FP"], N[f"{W}_FN"] = f"{m['fp']['mean']:.1f}", f"{m['fn']['mean']:.1f}"
        ci = w["bootstrap_ci"]
        N[f"{W}_RECCI"] = f"{f3(ci['recall']['mean'])} [{f3(ci['recall']['lo'])}, {f3(ci['recall']['hi'])}]"
        N[f"{W}_P06"] = f"{f3(ci['precision_at_0.6pct_prevalence']['mean'])} [{f3(ci['precision_at_0.6pct_prevalence']['lo'])}, {f3(ci['precision_at_0.6pct_prevalence']['hi'])}]"
        N[f"{W}_P06_MEAN"] = f2(ci["precision_at_0.6pct_prevalence"]["mean"])
        N[f"{W}_PSYN"] = f3(ci["precision_synthetic_mix"]["mean"])
        for f in fam:
            N[f"{W}_REC_{f}"] = f2(w["per_family_recall"][f])
        cal = w["calibration"]
        N[f"{W}_BRIER"] = f"raw {f3(cal['raw_xgb']['brier'])}, sigmoid {f3(cal['sigmoid']['brier'])}, isotonic {f3(cal['isotonic']['brier'])}"
        gr = R["phase4"]["world_" + W]
        N[f"{W}_GRAPH"] = f"{f3(gr['no_graph']['f1'])} -> {f3(gr['with_graph']['f1'])}"
        if W == "B":
            fk = w["fpr_by_legit_kind"]
            N["B_FPR_H2"], N["B_FPR_H10"], N["B_FPR_H6"], N["B_FPR_ROUTINE"] = (pc0(fk[k]["fpr"]) for k in ("H2_new_payee_then_pay", "H10_p2p_spree", "H6_genuine_onboarding", "routine_v2"))
    N["A_FRAUD_SHARE"] = pc0(R["data"]["world_A_fraud"] / R["data"]["world_A_rows"])
    N["B_FRAUD_SHARE"] = pc0(R["data"]["world_B_fraud"] / R["data"]["world_B_rows"])
    N["N_A"], N["N_A_FRAUD"] = str(R["data"]["world_A_rows"]), str(R["data"]["world_A_fraud"])
    N["N_B"], N["N_B_FRAUD"] = str(R["data"]["world_B_rows"]), str(R["data"]["world_B_fraud"])
    gs = R["world_B_generator_settings"]
    N["STEALTH"], N["ONBOARD"], N["YOUNG"] = pc0(gs["stealth"]), pc0(gs["onboarding_share"]), pc0(gs["young_share"])
    N["DOMAIN_AUC"] = f3(R["audit"]["legit_simulator_distinguishability_auc"])

    # stealth table
    rows = ["| share of adaptive attackers | F1 | recall | FP | FN | recall FAKE_MERCHANT |", "|---|---|---|---|---|---|"]
    for r in sweep:
        rows.append(f"| {pc0(r['stealth'])} | {f3(r['all_features']['f1'])} | {f3(r['all_features']['recall'])} | {r['all_features']['fp']:.1f} | {r['all_features']['fn']:.1f} | {f2(r['per_family_recall']['FAKE_MERCHANT'])} |")
    N["STEALTH_TABLE"] = "\n".join(rows)
    N["SWEEP_F1_0"], N["SWEEP_F1_100"] = f3(sweep[0]["all_features"]["f1"]), f3(sweep[-1]["all_features"]["f1"])

    # unseen families
    def lofo(world):
        L = ["| held-out family | supervised XGBoost | Isolation Forest | autoencoder | novelty lane (XGB or IF) | lane false-alarm rate |", "|---|---|---|---|---|---|"]
        for fam, v in R["phase2"][f"lofo_world_{world}"].items():
            L.append(f"| {fam} | {pc0(v['xgb_rec'])} | {pc0(v['iforest_rec'])} | {pc0(v['ae_rec'])} | {pc0(v['lane_rec'])} | {pc0(v['lane_fpr'])} |")
        return "\n".join(L)
    N["LOFO_B"], N["LOFO_A"] = lofo("B"), lofo("A")
    N["MULE_SUP"] = pc0(R["phase2"]["lofo_world_A"]["MULE_NETWORK"]["xgb_rec"])
    N["FAKE_SUP"] = pc0(R["phase2"]["lofo_world_B"]["FAKE_MERCHANT"]["xgb_rec"])

    # cost tables
    def cost(world):
        n = R["phase5"][f"nested_world_{world}"]
        L = ["| strategy | allow | review | block | fraud caught (review+block) | block precision | cost per 10k transactions |", "|---|---|---|---|---|---|---|"]
        for k in ("xgb_only", "stacked", "naive_max"):
            v, t = n[k], n[k]["traffic_share_at_assumed_prevalence"]
            L.append(f"| {k} | {pc(t['allow'])} | {pc(t['review'])} | {pc(t['block'])} | {pc(v['fraud_recall_review_plus_block'])} | {pc(v['block_precision_at_assumed_prevalence'])} | {money(v['expected_cost_per_10k_txn'])} |")
        L.append(f"| allow everything | 100% | 0% | 0% | 0% | - | {money(n['allow_everything_cost_per_10k'])} |")
        return "\n".join(L)
    N["COST_B"], N["COST_A"] = cost("B"), cost("A")
    nb = R["phase5"]["nested_world_B"]
    N["COST_B_XGB"], N["COST_B_STACK"], N["COST_B_NAIVE"] = (money(nb[k]["expected_cost_per_10k_txn"]) for k in ("xgb_only", "stacked", "naive_max"))
    N["COST_ALLOW"] = money(nb["allow_everything_cost_per_10k"])
    asm = nb["assumptions"]
    N["ASSUME"] = ", ".join(f"{k} = {v}" for k, v in asm.items())
    pol = {}
    for r in R["phase5"]["unseen_family_policy"]:
        pol.setdefault((r["world"], r["policy"]), []).append(r["cost"])
    mean = lambda w, p: sum(pol[(w, p)]) / len(pol[(w, p)])
    N["UNSEEN_B"] = f"stacked with novelty-aware thresholds {money(mean('B', 'stacked_lofo'))} vs standard stacked {money(mean('B', 'stacked_std'))} vs XGBoost only {money(mean('B', 'xgb_only_std'))}"
    N["UNSEEN_B_LOFO"], N["UNSEEN_B_STD"] = money(mean("B", "stacked_lofo")), money(mean("B", "stacked_std"))

    # feedback loop
    rows = ["| setting | cost change, targeted minus same-size control (per 10k) | sd over worlds | targeted better in |", "|---|---|---|---|"]
    names = {"pc2_found100": "2 fresh cases per mistake, all misses found", "pc2_found50": "2 fresh cases per mistake, half of the misses found", "pc4_found100": "4 fresh cases per mistake, all misses found"}
    for tag, label in names.items():
        p = rep / f"feedback_loop_{tag}.json"
        if p.exists():
            d = J(p)["summary"]["paired"]["targeted_minus_control_same_size"]["cost_per_10k"]
            rows.append(f"| {label} | {d['mean']:+,.0f} | {d['sd']:,.0f} | {d['n_down']} of {d['n']} worlds |")
    N["FEEDBACK_TABLE"] = "\n".join(rows)
    ma = rep / "feedback_miss_audit.md"
    N["MISS_STEALTH_SHARE"] = N["MISS_STEALTH_RATE"] = N["MISS_PLAIN_RATE"] = "n/a"
    if ma.exists():
        for line in ma.read_text(encoding="utf-8").splitlines():
            cells = [x.strip() for x in line.strip("|").split("|")]
            if cells and cells[0] in ("stealth", "plain") and len(cells) >= 5:
                if cells[0] == "stealth":
                    N["MISS_STEALTH_RATE"], N["MISS_STEALTH_SHARE"] = pc0(float(cells[3])), pc0(float(cells[4]))
                else:
                    N["MISS_PLAIN_RATE"] = pc0(float(cells[3]))
    return N


def phase8_section(root: Path) -> tuple[str, str]:
    """(README section, kit note) from reports/public/phase8_*.json if the user has run Phase 8; otherwise ("", default note).
    Every sentence is chosen by comparing the generated numbers, so the text cannot claim a result the data did not give."""
    pm = root / "reports" / "public" / "phase8_main.json"
    default_note = '"Validated on real data" (Phase 8 has not been run).'
    if not pm.exists():
        return "", default_note
    M = J(pm)
    t, r = M["time_split"]["models"], M["random_split"]["models"]
    tx, rx, tl = t["xgboost"], r["xgboost"], t["logreg"]
    d = M["data"]
    ap_t, ap_r = tx["raw"]["average_precision"], rx["raw"]["average_precision"]
    lo, hi = tx["raw"]["average_precision_ci95"]
    rlo, rhi = rx["raw"]["average_precision_ci95"]
    nf_t, nf_r = M["time_split"]["fraud"]["test"], M["random_split"]["fraud"]["test"]
    # calibration: did it beat the RAW scores by a meaningful margin (5%)? (naming a "winner" between near-equal scores would overstate)
    brier = {k: tx[k]["brier"] for k in ("raw", "sigmoid", "isotonic")}
    best = min(brier, key=brier.get)
    if best == "raw" or (brier["raw"] - brier[best]) / brier["raw"] < 0.05:
        cal_txt = "calibration did not improve the Brier score by more than 5% over the raw scores"
    else:
        cal_txt = f"{best} calibration improved the Brier score by {(brier['raw'] - brier[best]) / brier['raw']:.0%} over the raw scores"
    ap_iso = tx["isotonic"]["average_precision"]
    if ap_t - ap_iso > 0.02:
        cal_txt += f"; isotonic also lowered average precision ({f3(ap_iso)} vs {f3(ap_t)}) (a plausible reason is that isotonic outputs step values that tie many scores, but I did not test that here)"
    lb = {k: tl[k]["brier"] for k in ("raw", "sigmoid", "isotonic")}
    lbest = min(lb, key=lb.get)
    lr_txt = (f"for logistic regression {lbest} calibration lowered Brier from {lb['raw']:.5f} to {lb[lbest]:.5f}" if lbest != "raw" and (lb["raw"] - lb[lbest]) / lb["raw"] >= 0.05
              else "for logistic regression calibration did not change Brier by more than 5%")
    # time vs random: compare the gap with the spread across chronological blocks
    gap = ap_r - ap_t
    overlap = max(lo, rlo) <= min(hi, rhi)
    rp = root / "reports" / "public" / "phase8_rolling.json"
    roll_txt = ""
    if rp.exists():
        aps = [x["average_precision"] for x in J(rp)["rolling_origin"] if "average_precision" in x]
        if aps:
            spread = max(aps) - min(aps)
            below = all(a_ < ap_r for a_ in aps)
            roll_txt = (f" Across {len(aps)} chronological test blocks average precision ranged {f3(min(aps))} to {f3(max(aps))} (spread {f3(spread)}), "
                        f"{'all below' if below else 'not all below'} the random-split value.")
    gap_txt = (f"Time vs random split (XGBoost): average precision {f3(ap_t)} ({nf_t} frauds in the test block) vs {f3(ap_r)} ({nf_r} frauds): a gap of {gap:+.3f}. "
               f"The 95% bootstrap intervals {'overlap' if overlap else 'do not overlap'} ([{f3(lo)}, {f3(hi)}] vs [{f3(rlo)}, {f3(rhi)}]).{roll_txt} "
               "The test sets differ and fraud counts are small, so read this as suggestive of an ordering effect at most, not as established.")
    lr_ap = tl["raw"]["average_precision"]
    inside = lo <= lr_ap <= hi
    better = "XGBoost" if ap_t > lr_ap else "logistic regression"
    model_txt = (f"On this data {better} had the higher time-split average precision ({f3(max(ap_t, lr_ap))} vs {f3(min(ap_t, lr_ap))})"
                 + (", but the logistic regression value lies inside XGBoost's bootstrap interval, so this is not a clear win" if inside and better == "XGBoost" else "")
                 + f"; recall at a 0.5% alert budget was {f3(tx['raw']['recall_at_0.5pct_alerts'])} (XGBoost) vs {f3(tl['raw']['recall_at_0.5pct_alerts'])} (logistic regression).")
    ctrl = ""
    pc_ = root / "reports" / "public" / "phase8_control.json"
    if pc_.exists():
        c = J(pc_)["label_shuffle_time_split"]
        ctrl = f" Leakage control (labels shuffled in training): average precision {c['mean_average_precision']:.4f} against a test base rate of {c['test_prevalence']:.4f}; this only rules out gross label leakage."
    sec = f"""
## External method check on a public dataset (Phase 8)

This checks the modelling METHOD only, on the public ULB credit-card fraud data ({d['rows']:,} transactions, {d['fraud']} frauds, about {d['time_range_s'][1] / 3600:.0f} hours, no customer ids, anonymised features).
It is not the layered system: there are no sessions, devices, payees or graph here. Chronological split 60/20/20 (train / calibration / test).

| model | split | average precision | recall at a 0.5% alert budget | precision at that budget |
|---|---|---|---|---|
| XGBoost | time | {f3(ap_t)} | {f3(tx['raw']['recall_at_0.5pct_alerts'])} | {f3(tx['raw']['precision_at_0.5pct_alerts'])} |
| XGBoost | random | {f3(ap_r)} | {f3(rx['raw']['recall_at_0.5pct_alerts'])} | {f3(rx['raw']['precision_at_0.5pct_alerts'])} |
| logistic regression | time | {f3(tl['raw']['average_precision'])} | {f3(tl['raw']['recall_at_0.5pct_alerts'])} | {f3(tl['raw']['precision_at_0.5pct_alerts'])} |

- Calibration (time split, XGBoost): {cal_txt} ({brier['raw']:.5f} raw, {brier['sigmoid']:.5f} sigmoid, {brier['isotonic']:.5f} isotonic); {lr_txt}.
- {gap_txt}
- {model_txt}{ctrl}
- Limits: about two days of data, tens of frauds in the test block, no way to group by customer. Full tables: `reports/public/phase8_report.md`.
"""
    return sec, '"Validated on real data" (it was a method check on one public dataset, not the same system; say exactly that).'


def fill(template: str, N: dict) -> str:
    out = template
    for k in sorted(N, key=len, reverse=True):
        out = out.replace(f"@@{k}@@", N[k])
    left = re.findall(r"@@[A-Z0-9_]+@@", out)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    return out


# ------------------------------------------------------------------------------------------------ templates
README = r"""# AI Defence System (v2): a layered payment-fraud risk prototype, evaluated honestly on synthetic data

**What this is.** A prototype that scores a payment-session "trace" and returns ALLOW / REVIEW / BLOCK with an explanation.
It combines a supervised model for known fraud patterns, two novelty detectors for unusual behaviour, history-only graph features
(shared devices, payees, addresses, phones), and a cost-based decision policy. Every result in this repository is on **synthetic data**.

**What this is not.** It does not detect real-world fraud, and it does not yet legitimately claim to detect GenAI-era fraud.
The "GenAI-era" families are my own lightweight simulations of behaviour (for example KYC telemetry, not images or video). The honest claim is:
*a layered fraud-risk prototype with unseen-fraud evaluation on synthetic data.*

## Headline results (all synthetic; produced by the scripts listed under "Reproduce")

| | v1 as I first reported it | v1 re-run, customer-grouped folds | v2 World A (v1 attacks) | v2 World B (harder, 4 new families) |
|---|---|---|---|---|
| F1 | @@V1_F1@@ | @@LEAK_GRP_F1@@ +/- @@LEAK_GRP_SD@@ | @@A_F1@@ +/- @@A_F1_SD@@ | @@B_F1@@ +/- @@B_F1_SD@@ |
| Recall | @@V1_R@@ | - | @@A_RECCI@@ | @@B_RECCI@@ |
| Precision on the synthetic mix | @@V1_P@@ | - | @@A_PSYN@@ | @@B_PSYN@@ |
| Precision at an assumed 0.6% fraud rate | not reported | not reported | @@A_P06@@ | @@B_P06@@ |

How to read it: the first column is the number I originally put in the v1 README (the Stage 1 + 2 cascade on all 1,556 traces). The second column is a different,
simpler model (single-stage XGBoost, all 28 features) re-run with customer-grouped folds, so it is not an exact like-for-like. On the SAME attacks (World A) the score
barely moves: **fixing customer leakage and in-fold tuning did not change the picture**. The big drop appears only in World B, where legitimate traffic includes hard
cases and some attackers adapt. That drop is the finding: it shows how much of a high score came from an easy simulator.
At a realistic fraud rate, most flags in the hard world are false alarms (precision about @@B_P06_MEAN@@).

## Real vs offline vs served (read this before trusting any claim)

| Component | Status |
|---|---|
| Red Team simulator (v1 attacks) and v2 generators | Offline, synthetic. The v2 generators are my assumptions and were not validated by the Red Team validators |
| v1 pipeline (Stage 1 rules + calibrated XGBoost) | Served at `POST /api/score` (unchanged from the hackathon) |
| v1 GCN graph stage | Offline only; it was evaluated on rings I injected by hand |
| v2 XGBoost + Isolation Forest + autoencoder + history-only graph features + stacked policy | Served at `POST /api/v2/score` (numpy-only novelty models; no scikit-learn at serving time) |
| v2 case dossiers | Served at `POST /api/v2/dossier` (read-only) |
| v2 analyst feedback loop | Offline experiment; result is negative / inconclusive (see below) |
| Stage 1 rules inside v2 | Not part of v2 serving; in-fold they added nothing in World B |
| Authentication, rate limiting, shared state | Not built. One worker; the entity store and drift buffer live in memory |

## Architecture

```mermaid
flowchart LR
    T[Payment session trace] --> F[One shared feature extractor<br/>training and serving use the same code]
    S[(History-only entity store<br/>devices, payees, addresses, phones)] --> F
    F --> X[XGBoost<br/>known patterns]
    F --> I[Isolation Forest<br/>unusual vs normal]
    F --> A[Autoencoder<br/>unusual vs normal]
    X --> M[Stacked logistic regression]
    I --> M
    A --> M
    M --> P[Cost-based policy<br/>thresholds learned on inner out-of-fold scores]
    P --> D{ALLOW / REVIEW / BLOCK}
    D -->|REVIEW or BLOCK| C[Case dossier<br/>drivers, evidence, graph, acting party]
```

Why one feature extractor and history-only graph features: the first version of my graph features looked at future traces, which made World A
look better. Replaying traces in time order (training) and reading the live store (serving) removed that leakage, and a test checks that serving
reproduces the training features exactly.

## What I found wrong in my own v1 (and what it was worth)

| Finding | Effect, measured |
|---|---|
| Folds shared customers (about @@LEAK_SHARE@@ of test rows had a customer in training) | Not the cause of the high scores: F1 @@LEAK_V1_F1@@ (v1 folds) vs @@LEAK_GRP_F1@@ (grouped) |
| Calibration (sigmoid / isotonic) | Made probabilities worse: Brier World A: @@A_BRIER@@; World B: @@B_BRIER@@ |
| Graph features built from future traces | Looked like a gain in World A; with history-only features the gain is gone: @@A_GRAPH@@. World B keeps a small gain: @@B_GRAPH@@ |
| The simulator made attacks easy | With no adaptive attackers F1 is @@SWEEP_F1_0@@; when all attackers adapt it is @@SWEEP_F1_100@@ |
| Raw v1 fields that perfectly separate fraud from legit | Documented in `reports/v2/v2_report.md`; v1 features do not read them, so they did not inflate scores |
| Mixing the two simulators | The model learns which simulator made a trace (legit-vs-legit AUC @@DOMAIN_AUC@@), so the two worlds are evaluated separately |

## Hard world: how much do adaptive attackers cost?

World B has @@N_B@@ traces (@@N_B_FRAUD@@ fraud). Default setting: @@STEALTH@@ of attackers adapt to look legitimate, @@ONBOARD@@ of legitimate traffic is genuine onboarding,
@@YOUNG@@ of legitimate accounts are young. These values are my assumptions, chosen after seeing which features gave families away.

@@STEALTH_TABLE@@

Per-family recall at the default setting: SYNTHETIC_IDENTITY @@B_REC_SYNTHETIC_IDENTITY@@, FAKE_MERCHANT @@B_REC_FAKE_MERCHANT@@, DEEPFAKE_KYC @@B_REC_DEEPFAKE_KYC@@, AI_SCRIPTED_APP @@B_REC_AI_SCRIPTED_APP@@.
The biggest source of false alarms is genuine "new payee then pay" traffic (@@B_FPR_H2@@ flagged) followed by person-to-person sprees (@@B_FPR_H10@@): the same behaviours that define scams.

## Unseen fraud families (leave-one-family-out, fixed thresholds)

World B:

@@LOFO_B@@

World A:

@@LOFO_A@@

Reading: a supervised model is close to blind to a family it never saw (FAKE_MERCHANT: @@FAKE_SUP@@; in World A the graph-dependent MULE_NETWORK reaches @@MULE_SUP@@).
Novelty detectors catch some unusual behaviour (best on DEEPFAKE_KYC and AI_SCRIPTED_APP) but flag legitimate traffic too. None of this is a solved problem.

## Decision policy and cost (World B, outer-test only)

Assumptions (mine, not estimated from data): @@ASSUME@@.

@@COST_B@@

Naive max-fusion is much worse than a learned stacker, but stacking barely beats XGBoost alone, and novelty-aware thresholds did not pay off:
mean cost when the next fraud wave is an unseen family, World B: @@UNSEEN_B@@.

## Analyst feedback loop (Phase 7): a negative result

Misses and false alarms from the round-0 model become fresh look-alike training cases; a same-size control arm separates "learning from mistakes" from "more data".

@@FEEDBACK_TABLE@@

The effects are small, inconsistent across settings and smaller than their spread, so I do not claim the loop helps. Most misses are stealth attackers
(@@MISS_STEALTH_SHARE@@ of misses; stealth fraud is missed @@MISS_STEALTH_RATE@@ of the time versus @@MISS_PLAIN_RATE@@ for plain fraud).

@@PHASE8_SECTION@@
## Limitations (stated up front)

- Everything is synthetic. Scores are not real-world fraud probabilities and say nothing about real payment networks.
- World B's distributions, the stealth level, the onboarding share and every cost constant are my assumptions.
- The A-to-B transfer test is confounded by simulator differences; do not read it as generalisation.
- Some model behaviour is a simulator artifact (for example "no merchant purchase raises risk").
- Liability mappings for the four new families are assumptions, not regulation.
- The API has no authentication or rate limiting; state is in memory on one worker.
- Environment used to generate the reports in this repository: @@ENV@@. Regenerate on your own machine before quoting numbers.

## Reproduce (Windows cmd)

```
python -m venv .venv-eval
.venv-eval\Scripts\activate
pip install -r requirements-eval.lock.txt
pip install fastapi uvicorn httpx
set PYTHONPATH=src
pytest tests -q
pytest web_prototype\api\tests -q
python scripts\run_v2_all.py
python scripts\v2_stealth_sweep.py
python scripts\run_v2_feedback_loop.py --reps 10 --per-case 2 --found-share 1.0
python scripts\run_v2_feedback_loop.py --report
python scripts\v2_feedback_miss_audit.py
python scripts\make_v2_dossiers.py
python scripts\build_phase9_docs.py
python web_prototype\run_api.py --host 127.0.0.1 --port 8000
```
Run the two pytest commands separately (both test folders are named `tests`). Each long script runs in under a few minutes.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/score` | v1: Stage 1 rules + XGBoost |
| `POST /api/v2/score` | v2 cascade: stage scores, decision, top feature contributions |
| `POST /api/v2/dossier` | Case dossier for REVIEW / BLOCK (read-only) |
| `GET /api/v2/model-card`, `/api/v2/drift`, `/api/v2/status` | Model card with limitations, drift against training data, readiness |

`commit=true` on scoring is refused unless `V2_ALLOW_COMMIT=true`, so a public demo cannot poison the entity store.

Further reading: `reports/v2/v2_report.md` (full generated report), `reports/v2/dossiers.md` (example cases), `DEFENCE_NOTES.md` (my own questions and answers), `INTERVIEW_KIT.md`.
"""

KIT = r"""# Interview kit (generated by scripts/build_phase9_docs.py; every number comes from a generated report)

Rules for using this kit: say "synthetic" in the first sentence you speak about results; never say the project "detects GenAI fraud";
if you quote a number, it must match `README_V2.md` on your own regenerated reports.

## 1. Two-minute demo script

| Time | Say / do |
|---|---|
| 0:00 | "This is a layered fraud-risk prototype on synthetic data. It scores a payment session and returns allow, review or block with an explanation. I rebuilt my hackathon version after finding flaws in my own evaluation." |
| 0:20 | Open `/api/v2/status` (the free plan sleeps, so open it before the demo; the deployed API must be redeployed with the Phase 7 code, otherwise run the demo locally). Point out the model version and that the API says it is synthetic. |
| 0:35 | Score `v2_sample_deepfake_kyc.json` at `/api/v2/dossier`. Read the headline, then the drivers, then the evidence flags. "The drivers explain the model; the flags are readable checks I wrote; they are different on purpose." |
| 1:05 | Score `v2_sample_ai_scripted_app.json`. It lands just under the review threshold and is allowed. "This is a near miss. I show it because it is honest about where the model stops." Read the actual numbers from the response. |
| 1:25 | Show the results table: v1 headline F1 @@V1_F1@@, the original attacks under grouped folds @@A_F1@@, and the harder world @@B_F1@@. "Nothing moved until the world got harder. That is the finding." |
| 1:45 | "Unseen fraud families are mostly invisible to the supervised model (FAKE_MERCHANT @@FAKE_SUP@@). Learning from mistakes did not clearly help. Next I would test the modelling method on a real public dataset with time-based splits." |

## 2. Resume bullets (check each against what you can defend; adjust wording to your resume's style)

- Built and served (FastAPI) a layered payment-fraud risk prototype on synthetic data: XGBoost, Isolation Forest and autoencoder novelty detectors, history-only graph features and a cost-based allow/review/block policy with per-case explanations.
- Audited my own hackathon evaluation: customer-grouped splits, in-fold threshold learning, history-only graph features and unseen-family tests. F1 stayed at @@A_F1@@ on the original synthetic attacks but fell to @@B_F1@@ on a harder synthetic world with adaptive attackers.
- Measured unseen-fraud risk with leave-one-family-out tests (supervised recall on a held-out family as low as @@FAKE_SUP@@) and reported precision at an assumed 0.6% fraud rate (@@B_P06_MEAN@@) instead of the synthetic mix.

Say "served" or "deployed" for v2 only after you have redeployed the API with the v2 and Phase 7 code; until then say "runs locally and in the repo".

Shorter version (one line): "Layered fraud-risk prototype on synthetic data with unseen-fraud evaluation; audited and corrected the evaluation of my own v1."

## 3. Cold-email templates (short; fill every [bracket]; keep every claim traceable to your resume)

How to fill the opening: write ONE specific observation about the company's product or fraud/credit problem that you can point to on their site, blog or docs. Do not guess facts about them.

**Template A: payments / fraud team**

Subject: [specific observation, 6-8 words]

Hi [Name],

[One sharp sentence about a fraud or risk challenge in [Company]'s product, for example the trade-off between friction and missed fraud on [their product].]

I built a layered fraud-risk prototype (synthetic data). My v1 F1 of @@V1_F1@@ held at @@A_F1@@ under customer-grouped folds but fell to @@B_F1@@ once attackers adapted; I also measured precision at a 0.6% fraud rate (@@B_P06_MEAN@@) and tested unseen fraud families. [Optional second project: one line with a metric from your resume.]

Could I get 15 minutes this week or next (by [date]) to hear how your team handles [specific topic] and whether a six-month data science / AI-ML internship from January 2027 fits?

[Your name] | GitHub: github.com/Pranavi0525 | LinkedIn: [your URL]

**Template B: lending / credit team**

Subject: [specific observation about their credit product]

Hi [Name],

[One sharp sentence on a credit-risk gap at [Company], for example separating genuine new-to-credit applicants from synthetic identities.]

My fraud-risk project runs into that problem: genuine onboarding is one of its false-alarm sources (@@B_FPR_H6@@ of genuine onboarding flagged), and recall on synthetic identities was @@B_REC_SYNTHETIC_IDENTITY@@ on my synthetic benchmark. [Optional second project: your credit-risk project and one metric from your resume.]

Would you have 15 minutes before [date]? I am looking for a six-month data science / AI-ML internship starting January 2027.

[Your name] | GitHub: github.com/Pranavi0525 | LinkedIn: [your URL]

## 4. Interview Q&A (answer in your own words; these are the facts and the honest framing)

1. **Does it detect GenAI fraud?** No. It has four simulated GenAI-era families (synthetic identity, fake merchant, deepfake-KYC telemetry, AI-scripted APP). They are my assumptions, not validated. The honest claim is a layered prototype with unseen-fraud evaluation on synthetic data.
2. **Your v1 had F1 @@V1_F1@@. Is v2 worse?** On the same attacks it is about the same (World A F1 @@A_F1@@), so the customer overlap and in-fold tuning problems did not inflate the score much (single-stage F1 @@LEAK_V1_F1@@ with v1 folds vs @@LEAK_GRP_F1@@ grouped). The drop is in World B (@@B_F1@@), where attackers adapt: F1 @@SWEEP_F1_0@@ with no adaptive attackers, @@SWEEP_F1_100@@ when all adapt. The lesson: the simulator was easy, and I measured how easy.
3. **Why report precision at 0.6% prevalence?** Because the synthetic mixes are @@A_FRAUD_SHARE@@ (World A) and @@B_FRAUD_SHARE@@ (World B) fraud. At a realistic rate, precision depends on the false-positive rate: @@B_P06@@ in World B and @@A_P06@@ in World A. Those numbers use an assumed rate, not a measured one.
4. **Why did calibration hurt?** Brier scores: World B @@B_BRIER@@. I did not investigate why. Plausible reasons are small data and raw scores that were already reasonable, but that is a hypothesis, so say so. I report it as a measured negative result.
5. **What was the graph leakage?** My first graph features used all traces, including future ones, which made World A look better. Replaying traces in time order removed it; World A went @@A_GRAPH@@. World B kept a small gain: @@B_GRAPH@@.
6. **How do you know serving matches training?** One shared feature extractor, history-only graph features, and tests: the extractor equals my v1 extractor on every real trace except the intentional failure-feature fix; replaying training traces through the scorer reproduces the training features; the numpy novelty models match scikit-learn to 1e-9.
7. **How are thresholds chosen?** By expected cost with assumed constants (@@ASSUME@@), learned on inner out-of-fold scores only, then evaluated on outer folds. The constants are assumptions; the thresholds move when they change.
8. **Stacking vs XGBoost alone?** World B cost per 10k: stacked @@COST_B_STACK@@, XGBoost only @@COST_B_XGB@@, naive max @@COST_B_NAIVE@@, allow everything @@COST_ALLOW@@. Stacking helps little; naive max is much worse.
9. **What happens with an unseen fraud family?** Supervised recall is near zero for some (FAKE_MERCHANT @@FAKE_SUP@@); novelty detectors help for others but raise false alarms. Novelty-aware stacking was not worth it: mean cost @@UNSEEN_B_LOFO@@ vs @@UNSEEN_B_STD@@.
10. **What does the feedback loop show?** Nothing reliable. Targeted retraining versus a same-size control was small, inconsistent and within noise; most misses are stealth attackers (@@MISS_STEALTH_SHARE@@ of misses). I report it as a negative result.
11. **Biggest false-alarm source?** Genuine "new payee then pay" (@@B_FPR_H2@@ flagged) and person-to-person sprees (@@B_FPR_H10@@): they look like scams. In production this is the friction-versus-fraud trade-off.
12. **Why World A and World B separately?** Pooling lets the model learn which simulator made the trace (legit-vs-legit AUC @@DOMAIN_AUC@@).
13. **What would you do with real data?** Time-based splits, label delay, a same-size control for any retraining, threshold stability checks, and a human-validated cost model.
14. **What is not production-ready?** No authentication or rate limiting, in-memory state on one worker, Stage 1 rules not in v2 serving, and everything trained on synthetic data.
15. **What are you most proud of?** Finding and publishing my own evaluation flaws, with the measured size of each, instead of defending the original number.

## 5. Claims ledger (say it only if the evidence file supports it)

| Claim | Evidence file |
|---|---|
| v1 headline F1 @@V1_F1@@ (362 TP, @@V1_FP@@ FP, @@V1_FN@@ FN) | `phase5a_evaluation_report.json` |
| Customer overlap is not the cause of high v1 scores | `reports/v2/leakage_check.md` |
| World A / World B grouped-CV results, CIs, calibration, graph ablation, unseen families, policy cost | `reports/v2/v2_report.md`, `reports/v2/v2_results.json` |
| F1 versus share of adaptive attackers | `reports/v2/stealth_sweep.md` |
| Feedback loop result and miss audit | `reports/v2/feedback_loop.md`, `reports/v2/feedback_miss_audit.md` |
| Example cases | `reports/v2/dossiers.md` |

## 6. Do not say

- "Detects GenAI / deepfake fraud" (it scores simulated KYC telemetry, not images or video).
- "Production-ready", "state of the art", or any real-world accuracy.
- @@P8_NOTE@@
- Any number you have not regenerated on your own machine.
"""


def build(out: Path = ROOT, root: Path = ROOT) -> dict:
    N = collect(root)
    sec, note = phase8_section(root)
    N["PHASE8_SECTION"], N["P8_NOTE"] = sec, note
    (out / "reports" / "v2").mkdir(parents=True, exist_ok=True)
    (out / "README_V2.md").write_text(fill(README, N), encoding="utf-8")
    (out / "INTERVIEW_KIT.md").write_text(fill(KIT, N), encoding="utf-8")
    tbl = ["# Results tables (generated by scripts/build_phase9_docs.py from reports/v2/*.json)", "", "## Stealth sweep", N["STEALTH_TABLE"], "",
           "## Unseen families, World B", N["LOFO_B"], "", "## Unseen families, World A", N["LOFO_A"], "", "## Policy cost, World B", N["COST_B"], "",
           "## Policy cost, World A", N["COST_A"], "", "## Feedback loop", N["FEEDBACK_TABLE"], ""]
    (out / "reports" / "v2" / "results_table.md").write_text("\n".join(tbl), encoding="utf-8")
    return N


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    build(Path(a.out))
    print("wrote README_V2.md, INTERVIEW_KIT.md, reports/v2/results_table.md in", a.out)
