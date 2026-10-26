# Phase 10 -- card-transaction track (SYNTHETIC; seeds=3, mean ± sd over simulator seeds)

Fixed operating point: model thresholds set at 1% false-positive rate on a validation block of legitimate traffic (lane: 0.5% each). Rules have fixed thresholds.

## Time split (train first 30 days, validate 30-42, test 42-60)

| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |
|---|---|---|---|---|
| rules_naive | 0.66 ± 0.04 | 0.28 ± 0.01 | 0.02 ± 0.00 | 0.08 ± 0.01 |
| rules_cp | 0.41 ± 0.06 | 0.20 ± 0.02 | 0.00 ± 0.00 | 0.93 ± 0.04 |
| xgb | 0.99 ± 0.01 | 0.99 ± 0.00 | 0.01 ± 0.00 | 0.43 ± 0.01 |
| mlp | 1.00 ± 0.01 | 0.97 ± 0.01 | 0.01 ± 0.00 | 0.42 ± 0.03 |
| percard_iforest | 0.79 ± 0.01 | 0.67 ± 0.02 | 0.01 ± 0.00 | 0.30 ± 0.01 |
| lane_xgb_or_percard | 0.99 ± 0.01 | 0.97 ± 0.00 | 0.01 ± 0.00 | 0.43 ± 0.04 |

## Unseen pattern (leave-one-pattern-out): EVENT recall on the held-out pattern

| held-out pattern | rules_naive | rules_cp | xgb | mlp | percard_iforest | lane_xgb_or_percard |
|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 0.99 ± 0.01 | 0.78 ± 0.03 | 1.00 ± 0.00 | 0.97 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CARD_TESTING | 1.00 ± 0.00 | 0.98 ± 0.02 | 1.00 ± 0.00 | 0.98 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CNP_SPREE | 0.78 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.94 ± 0.02 | 1.00 ± 0.00 |
| ATM_CASHOUT | 1.00 ± 0.00 | 0.93 ± 0.02 | 1.00 ± 0.00 | 0.74 ± 0.30 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| SLOW_DRIP | 0.13 ± 0.05 | 0.00 ± 0.00 | 0.83 ± 0.03 | 0.88 ± 0.05 | 0.07 ± 0.04 | 0.63 ± 0.05 |
| LIMIT_SPLIT | 0.17 ± 0.03 | 0.01 ± 0.01 | 0.99 ± 0.01 | 0.92 ± 0.08 | 0.94 ± 0.05 | 0.97 ± 0.05 |

(legit FPR for each model in the LOPO runs is the same as in the time split.)
