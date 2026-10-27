# Card model: why INR 10,000,000,000 scored 1.85e-5 (LEGITIMATE) -- investigation and fix

All numbers are from SYNTHETIC simulator data. None is evidence about real fraud. Reproduce with the scripts named below.

## Confirmed root cause
1. **Training range.** The original simulator never produced an amount above 18,313 (no row above 20,000 in any class; fraud max 11,522).
   The trees' largest amount split is at 10,000 (`amount`), 2e4 (`sum_24h`), 8.5 (`log_amount`), 3.22 (`amt_z`), 1.76 (`amt_ratio_max`).
   A tree cannot extrapolate: every amount above its largest split lands in the same leaf. The old sweep is flat from 5,000 INR to 1e10 INR
   (`reports/card/amount_sweep.md`, BEFORE table). The request's features were finite and correct (amt_z 38.7, amt_ratio_max 6.7e6);
   they were simply beyond anything the model was fitted on.
2. **No large legitimate purchases and few large-spend fraud scenarios** in the simulator, so "large" carried no learned meaning
   (only 20 legit rows > 10,000 of 112,142).
3. **Timing leakage in the simulator (found during the investigation).** Every fraud event started 10-30 min after one of the card's own
   transactions: 91% of event-first rows had a gap in that window vs 2.9% of legit rows; that one indicator alone gives AUC 0.946, and
   `gap_s` was the 3rd most important feature. The old 98% recall was partly this shortcut.
4. **Not causes (checked):** the feature code is causal and uses only the card's prior rows; feature order at train and serve is identical
   (`policy.features == FEATURES == booster.feature_names`); the threshold was chosen on a validation block disjoint from the fit block and
   achieved its 1% FPR target; the pydantic schema was not responsible. The threshold was NOT lowered.

## What changed
- `v2/card/sim.py` `extended=True` (default off, `extended=False` is bit-identical to the original -- verified): legit big-ticket purchases
  incl. premium cards, new cards, six extra fraud scenarios whose amounts overlap legit big tickets (HIGH_VALUE_BUSTOUT half absolute / half
  card-relative, SPEND_SPIKE_FAMILIAR in an otherwise ordinary context, RAPID_REPEAT, UNUSUAL_MERCHANT, DECLINE_THEN_APPROVE,
  NEW_CARD_FIRST_USE), organic event start times (gap AUC 0.513 afterwards). Amount cap = shared business-limit default.
- `v2/card/limits.py`: INR, `CARD_MAX_TXN_AMOUNT` (default 1,000,000), `MAX_TS_SECONDS`. Validation, not classification.
- `v2/card/train.py`: extended data by default; records PR-AUC/ROC-AUC, validation-block results, per-pattern and per-slice recall/FPR,
  training amount/feature support in `policy.json`. Same XGBoost params, same split, same FPR-based threshold rule.
- `v2/card/serving.py`: amount limit, ts bound, MCC format, history cleaning (other-card / future / duplicate / self rows dropped and
  counted), `features`, `features_outside_training_range` + warning, error rows attributed to transaction vs history.
- API: NaN/Infinity no longer produces a 500 (shared 422 handler now JSON-safe); status/model-card publish the limit and units.
- Dashboard: shows history rows used/supplied and attributes errors to "Transaction" / "History row n". Nothing else changed.

## Behaviour you should expect (not hidden)
- Scores are still flat above ~50,000 INR (largest learned splits); hence the business limit and the out-of-range warning.
- The model is NOT an amount rule: the same amount scores differently by context (sweep table). It is also not monotone in amount
  (e.g. context B at 5,000 vs 10,000) because of feature interactions; monotonicity is not enforced.
- Counter-intuitive and caused by simulator assumptions: large legit purchases in the simulator are mostly at *new* merchants, while
  the SPEND_SPIKE_FAMILIAR fraud is at familiar merchants, so at huge amounts "familiar merchant" scores higher than "new merchant".
  Whether real data looks like that is unknown.
- On the extended synthetic test block 25 of 215 legit purchases >= 20,000 INR (11.6%) are still flagged (FPR 0.65% overall).
