"""Diagnostic: is there an injection-timing artifact in the card simulator, and which patterns does it affect?

For every fraud pattern and for extended=False / extended=True, over many seeds, measure

  * gap_prev_legit : first fraudulent txn of an event  -  last LEGITIMATE txn of the same card before it
  * gap_prev_any   : same, but against the last txn of ANY class (this is what the `gap_s` feature sees)
  * in_window      : share of events whose gap_prev_legit lies in the original artifact window [600 s, 1800 s]
  * AUC (window)   : how well "gap close to 20 min" separates the event's first txn from ordinary legit rows
  * intra-event gaps, within-event ordering, overlap of events on one card, events lost to the day-60 cut
  * DORMANT_REACTIVATION: intended dormant window really empty? first-fraud gap, history left on the card
  * ODD_HOURS_QUIET: hour of day of the first fraud txn (semantics that a timing fix must not move)

`--sim-path` lets the same script run against an older copy of sim.py (before / after comparison).
Writes a JSON file and a markdown table. Uses numpy + pandas only.

    python scripts/diagnose_card_sim_timing.py --seeds 30 --out reports/card/sim_timing_diagnostic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

WIN_LO, WIN_HI = 600.0, 1800.0     # the original 10-30 minute injection window (seconds)
DAY = 86400.0


def load_sim(path: str | None):
    if path is None:
        from v2.card import sim
        return sim
    import v2.card  # noqa: F401  (so the relative import `.limits` resolves)
    spec = importlib.util.spec_from_file_location("v2.card._sim_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "v2.card"
    spec.loader.exec_module(mod)
    return mod


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), rank based; both arrays finite."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return float((s[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def two_sided_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that does not care which direction separates the classes (0.5 = no information, 1.0 = perfect either way)."""
    v = auc(pos, neg)
    return float(max(v, 1 - v))


def window_score(gap: np.ndarray) -> np.ndarray:
    """Higher = closer to 20 minutes on a log scale. This detects the artifact and nothing else."""
    return -np.abs(np.log(np.maximum(gap, 1.0) / 1200.0))


def event_table(df: pd.DataFrame, days: float) -> pd.DataFrame:
    df = df.sort_values(["card", "ts"]).reset_index(drop=True)
    rate_hat = (df[df.is_fraud == 0].groupby("card").size() / days).to_dict()      # legit txns per day, estimated from the card's own rows
    leg_ts = {c: g["ts"].to_numpy() for c, g in df[df.is_fraud == 0].groupby("card")}
    all_ts = {c: g["ts"].to_numpy() for c, g in df.groupby("card")}
    out = []
    fr = df[df.is_fraud == 1]
    for ev, g in fr.groupby("event"):
        c = int(g["card"].iloc[0]); t = np.sort(g["ts"].to_numpy()); first, last = float(t[0]), float(t[-1])
        lt = leg_ts.get(c, np.empty(0)); at = all_ts[c]
        i = int(np.searchsorted(lt, first, "left")); j = int(np.searchsorted(at, first, "left"))
        prev_legit = float(lt[i - 1]) if i > 0 else np.nan
        prev_any = float(at[j - 1]) if j > 0 else np.nan
        # legit history strictly before the first fraud txn, and legit rows inside the 15-day dormant window
        n_leg_15d = int(((lt >= first - 15 * DAY) & (lt < first)).sum())
        n_any_15d = int(((at >= first - 15 * DAY) & (at < first)).sum())
        out.append(dict(event=int(ev), card=c, pattern=g["pattern"].iloc[0], first=first, last=last, n=len(t),
                        gap_prev_legit=first - prev_legit, gap_prev_any=first - prev_any, hist_legit=i, hist_any=j,
                        gap_prev_any_norm=(first - prev_any) / DAY * rate_hat.get(c, np.nan),
                        n_legit_15d=n_leg_15d, n_any_15d=n_any_15d,
                        intra_min=float(np.diff(t).min()) if len(t) > 1 else np.nan,
                        intra_med=float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
                        intra_max=float(np.diff(t).max()) if len(t) > 1 else np.nan,
                        first_hour=(first % DAY) / 3600.0))
    ev = pd.DataFrame(out)
    # events on the same card whose [first, last] intervals overlap
    ev = ev.sort_values(["card", "first"]).reset_index(drop=True)
    prev_last = ev.groupby("card")["last"].shift(1)
    ev["overlaps_prev_event"] = (ev["first"] < prev_last).fillna(False)
    return ev


def legit_gaps(df: pd.DataFrame, days: float) -> dict:
    d = df.sort_values(["card", "ts"]).reset_index(drop=True)
    gap_any = d.groupby("card")["ts"].diff()
    legit = d[d.is_fraud == 0]
    gap_legit_prev_legit = legit.groupby("card")["ts"].diff()
    rate = (legit.groupby("card").size() / days)
    norm_any = (gap_any[d.is_fraud == 0] / DAY * d.loc[d.is_fraud == 0, "card"].map(rate))
    hour = (legit["ts"] % DAY) / 3600.0
    night = (hour >= 2) & (hour < 2.5)
    return dict(any=gap_any[d.is_fraud == 0].dropna().to_numpy(), legit=gap_legit_prev_legit.dropna().to_numpy(),
                night=gap_legit_prev_legit[night].dropna().to_numpy(), any_norm=norm_any.dropna().to_numpy(),
                night_norm=(gap_legit_prev_legit[night] / DAY * legit.loc[night, "card"].map(rate)).dropna().to_numpy())


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a); b = np.sort(b); v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, side="right") / len(a) - np.searchsorted(b, v, side="right") / len(b))))


def summ(x):
    x = np.asarray(x, float); nan = int(np.isnan(x).sum()); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, n_nan=nan)
    q = np.quantile(x, [0.1, 0.5, 0.9])
    return dict(n=int(len(x)), n_nan=nan, min=float(x.min()), p10=float(q[0]), median=float(q[1]), mean=float(x.mean()), p90=float(q[2]), max=float(x.max()))


_SIM = None
_ARGS = None


def _one(task):
    seed, ext = task
    cache = os.path.join(_ARGS.cache_dir, f"seed{seed}_ext{int(ext)}.pkl") if _ARGS.cache_dir else None
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    df = _SIM.simulate(n_cards=_ARGS.n_cards, days=_ARGS.days, seed=seed, events_per_pattern=_ARGS.events_per_pattern, novel=True, extended=ext)
    ev = event_table(df, _ARGS.days); ev["seed"] = seed; ev["extended"] = ext
    lg = legit_gaps(df, _ARGS.days); lg.update(seed=seed, extended=ext)
    if cache:
        os.makedirs(_ARGS.cache_dir, exist_ok=True)
        pd.to_pickle((ev, lg), cache)
    return ev, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--sim-path", default=None, help="path to an alternative sim.py (e.g. the pre-fix copy)")
    ap.add_argument("--out", default="reports/card/sim_timing_diagnostic")
    ap.add_argument("--n-cards", type=int, default=1500)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--events-per-pattern", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one task per seed and mode)")
    ap.add_argument("--extended-only", action="store_true", help="skip the extended=False runs (its output is unchanged by the timing fix)")
    ap.add_argument("--cache-dir", default=None, help="keep per-seed results here so an interrupted run can resume")
    a = ap.parse_args()
    sim = load_sim(a.sim_path)
    pats = list(sim.PATTERNS) + list(sim.NOVEL) + list(sim.EXTENDED)

    global _SIM, _ARGS
    _SIM, _ARGS = sim, a
    modes = (True,) if a.extended_only else (False, True)
    tasks = [(seed, ext) for seed in range(a.seeds) for ext in modes]
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs) as pool:
            results = pool.map(_one, tasks, chunksize=1)
    else:
        results = [_one(t) for t in tasks]
    ev_all = [r[0] for r in results]; leg_all = [r[1] for r in results]
    ev = pd.concat(ev_all, ignore_index=True)
    res = {"seeds": a.seeds, "n_cards": a.n_cards, "days": a.days, "events_per_pattern": a.events_per_pattern, "patterns": {}}
    for ext in modes:
        lg = [x for x in leg_all if x["extended"] == ext]
        base_any = np.concatenate([x["any"] for x in lg]); base_leg = np.concatenate([x["legit"] for x in lg]); base_night = np.concatenate([x["night"] for x in lg])
        base_any_norm = np.concatenate([x["any_norm"] for x in lg]); base_night_norm = np.concatenate([x["night_norm"] for x in lg])
        res[f"legit_baseline_extended={ext}"] = dict(
            gap_prev_any=summ(base_any), gap_prev_legit=summ(base_leg), in_window_share_legit=float(((base_leg >= WIN_LO) & (base_leg <= WIN_HI)).mean()),
            night_2to2h30_gap_prev_legit=summ(base_night), night_in_window_share=float(((base_night >= WIN_LO) & (base_night <= WIN_HI)).mean()))
        for pat in pats:
            e = ev[(ev.extended == ext) & (ev.pattern == pat)]
            if e.empty:
                continue
            gl = e["gap_prev_legit"].to_numpy(); ga = e["gap_prev_any"].to_numpy()
            ok = ~np.isnan(gl)
            per_seed_in = e.groupby("seed")["gap_prev_legit"].apply(lambda s: float(((s >= WIN_LO) & (s <= WIN_HI)).mean()))
            r = dict(events=int(len(e)), events_per_seed_mean=float(len(e) / a.seeds),
                     gap_prev_legit=summ(gl), gap_prev_any=summ(ga),
                     in_window_share=float(((gl[ok] >= WIN_LO) & (gl[ok] <= WIN_HI)).mean()) if ok.any() else None,
                     in_window_share_any=float(((ga[~np.isnan(ga)] >= WIN_LO) & (ga[~np.isnan(ga)] <= WIN_HI)).mean()),
                     in_window_per_seed_sd=float(per_seed_in.std()),
                     auc_indicator_vs_legit=(lambda pf, pl: float(0.5 + 0.5 * (pf - pl)))(
                         float(((ga[~np.isnan(ga)] >= WIN_LO) & (ga[~np.isnan(ga)] <= WIN_HI)).mean()),
                         float(((base_any >= WIN_LO) & (base_any <= WIN_HI)).mean())),
                     auc_window_vs_legit=auc(window_score(ga[~np.isnan(ga)]), window_score(base_any)),
                     auc_loggap_two_sided=two_sided_auc(np.log(np.maximum(ga[~np.isnan(ga)], 1.0)), np.log(np.maximum(base_any, 1.0))),
                     auc_loggap_rate_normalised=two_sided_auc(np.log(np.maximum(e["gap_prev_any_norm"].dropna().to_numpy(), 1e-6)), np.log(np.maximum(base_any_norm, 1e-6))),
                     intra_event_gap=dict(min=float(np.nanmin(e.intra_min)) if e.intra_min.notna().any() else None,
                                          median=float(np.nanmedian(e.intra_med)) if e.intra_med.notna().any() else None,
                                          max=float(np.nanmax(e.intra_max)) if e.intra_max.notna().any() else None),
                     negative_intra_gaps=int((e.intra_min < 0).sum()),
                     events_overlapping_previous_event_on_card=int(e.overlaps_prev_event.sum()),
                     first_txn_with_no_prior_legit=int(e.hist_legit.eq(0).sum()),
                     first_hour_of_day=dict(min=float(e.first_hour.min()), median=float(e.first_hour.median()), max=float(e.first_hour.max())),
                     hist_legit_at_first_fraud=summ(e.hist_legit.to_numpy()))
            if pat == "ODD_HOURS_QUIET":
                gn = e["gap_prev_legit"].to_numpy(); gn = gn[~np.isnan(gn)]
                nn = (e["gap_prev_any_norm"].to_numpy()); nn = nn[~np.isnan(nn)]
                r["vs_legit_rows_at_02_00_to_02_30"] = dict(
                    ks_raw=ks(gn, base_night), ks_raw_5pct_critical=float(1.36 * np.sqrt((len(gn) + len(base_night)) / (len(gn) * len(base_night)))),
                    ks_rate_normalised=ks(nn, base_night_norm),
                    median_min_odd=float(np.median(gn) / 60), median_min_legit_night=float(np.median(base_night) / 60))
            if pat == "DORMANT_REACTIVATION":
                r["dormant"] = dict(
                    events_with_any_legit_in_prior_15d=int((e.n_legit_15d > 0).sum()),
                    events_with_any_txn_in_prior_15d_incl_other_fraud=int((e.n_any_15d > 0).sum()),
                    min_gap_prev_legit_days=float(np.nanmin(gl) / DAY), first_fraud_min_day=float(e["first"].min() / DAY))
            res["patterns"].setdefault(pat, {})[f"extended={ext}"] = r
    # events lost to the `ts < days*DAY` cut (compare against the non-extended event count per pattern is not meaningful for EXTENDED)
    for pat in pats:
        for ext in modes:
            r = res["patterns"].get(pat, {}).get(f"extended={ext}")
            if r:
                e = ev[(ev.extended == ext) & (ev.pattern == pat)]
                r["events_with_rows_cut_by_window_end_proxy"] = int((e["last"] > (a.days - 1) * DAY).sum())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as fp:
        json.dump(res, fp, indent=1)

    m = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v / 60:,.1f}"
    L = [f"# Card simulator timing diagnostic ({a.seeds} seeds, n_cards={a.n_cards}, days={a.days}, events_per_pattern={a.events_per_pattern}, novel=True)\n",
         "Gap = first fraudulent txn of an event minus the last LEGITIMATE txn of the same card before it. Minutes.",
         f"`in-window` = share of events whose gap is in [{WIN_LO/60:.0f}, {WIN_HI/60:.0f}] min (the original artifact). `AUC(ind)` = AUC of the in-window yes/no indicator on the `gap_s` feature (the measure used in reports/card/large_amount_investigation.md: 0.946 before, 0.513 after); "
         "`AUC(win)` = AUC of a continuous 'closeness to 20 min' score (more sensitive). 0.5 = no information.\n"]
    for ext in modes:
        b = res[f"legit_baseline_extended={ext}"]
        L += [f"## extended={ext}\n",
              f"Legit baseline: gap-to-previous-legit median {m(b['gap_prev_legit']['median'])} min, mean {m(b['gap_prev_legit']['mean'])}, in-window share {b['in_window_share_legit']:.3f} "
              f"(legit rows between 02:00 and 02:30: {b['night_in_window_share']:.3f}).\n",
              "| pattern | events | min | p10 | median | mean | p90 | max | no prior legit | in-window | AUC(ind) | AUC(win) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for pat in pats:
            r = res["patterns"].get(pat, {}).get(f"extended={ext}")
            if not r:
                continue
            g = r["gap_prev_legit"]
            L.append(f"| {pat} | {r['events']} | {m(g.get('min'))} | {m(g.get('p10'))} | {m(g.get('median'))} | {m(g.get('mean'))} | {m(g.get('p90'))} | {m(g.get('max'))} | "
                     f"{g.get('n_nan', 0)} | {r['in_window_share'] if r['in_window_share'] is None else format(r['in_window_share'], '.3f')} | {r['auc_indicator_vs_legit']:.3f} | {r['auc_window_vs_legit']:.3f} |")
        L.append("")
    with open(a.out + ".md", "w") as fp:
        fp.write("\n".join(L))
    print("\n".join(L))
    print(f"\nSaved {a.out}.json / .md")


if __name__ == "__main__":
    main()
