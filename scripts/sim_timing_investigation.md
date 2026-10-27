# Card simulator: extended-mode event timing (SYNTHETIC data; 30 seeds)

Reproduce: `python scripts/diagnose_card_sim_timing.py --seeds 30 [--extended-only] [--sim-path OLD_sim.py]`. Tests: `tests/test_card_sim_timing.py`.
Gap = first fraud txn of an event minus the last LEGIT txn of the same card. "In window" = gap in 10-30 min (the original injection artifact).

## Which patterns had the artifact (extended=True, original code -> fixed code)
| pattern | in window | indicator AUC | log-gap AUC | events/30 seeds |
|---|---|---|---|---|
| six base patterns | 3.3-3.9% (unchanged) | 0.50 | 0.57-0.59 | 1165-1176 / 1200 |
| FAMILIAR_MERCHANT_DRAIN | **98.2% -> 3.2%** | **0.976 -> 0.502** | 0.966 -> 0.581 | 1200 -> 1171 |
| ODD_HOURS_QUIET | 0.3% (no artifact) | 0.49 | 0.58 (0.52 rate-normalised) | 1199 |
| DORMANT_REACTIVATION | 0.0% (min gap 15.0003 d) | - | 0.995 (by design) | 1197 |
extended=False: all six base patterns and DRAIN are 97.6-98.5% in window (original behaviour, unchanged).

## Decision
Only FAMILIAR_MERCHANT_DRAIN is re-timed (`RETIMED_PATTERNS`, helper `_retime_event`). ODD_HOURS_QUIET (02:00-02:30 clock time) and
DORMANT_REACTIVATION (absolute start, emptied 15-day window) never had the gap; re-timing them breaks the pattern (5 tests fail if tried).
Verified by output hashes (seeds 0-3, novel/extended all combinations): only DRAIN rows in extended+novel changed; extended=False,
extended+novel=False, every other pattern and all legit rows are bit-identical. The committed `models/card` and `phase10_card.*` are unaffected;
`phase10_card_novel.*` was produced with the old DRAIN timing and should be regenerated.

## Open findings (NOT changed)
1. DORMANT in extended mode: 46/1197 events (3.8%) sit on cards with no prior legit history (0 in original mode); 122/1197 (10%) have other fraud
   rows inside the "dormant" 15 days (only legit rows are removed). Intended semantics are weakened, not the window itself.
2. Extended mode drops 2-3% of fraud events past day 60 (organic gaps push them off the end), biasing fraud slightly earlier.
3. Residual gap signal for ALL re-timed patterns (pre-existing): log-gap AUC 0.57-0.60 (0.63-0.66 rate-normalised). Consistent with anchor + exponential gap
   giving the minimum of two exponentials (theory 0.667), not proven. A cleaner design would draw the start uniformly over the card's activity span.
4. ODD gaps differ from legit rows at 02:00-02:30 (KS 0.092 raw, 0.068 rate-normalised, 5% critical 0.041): card selection explains part, not all.
