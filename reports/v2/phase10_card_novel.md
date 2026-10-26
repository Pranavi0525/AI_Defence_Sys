# Phase 10 -- card-transaction track (SYNTHETIC; seeds=3, mean ± sd over simulator seeds)

Fixed operating point: model thresholds set at 1% false-positive rate on a validation block of legitimate traffic (lane: 0.5% each). Rules have fixed thresholds.

## Time split (train first 30 days, validate 30-42, test 42-60)

| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |
|---|---|---|---|---|
| rules_naive | 0.41 ± 0.02 | 0.19 ± 0.00 | 0.02 ± 0.00 | 0.06 ± 0.00 |
| rules_cp | 0.25 ± 0.02 | 0.14 ± 0.00 | 0.00 ± 0.00 | 0.90 ± 0.04 |
| xgb | 0.98 ± 0.01 | 0.92 ± 0.00 | 0.01 ± 0.00 | 0.42 ± 0.01 |
| mlp | 0.89 ± 0.02 | 0.81 ± 0.03 | 0.01 ± 0.00 | 0.41 ± 0.03 |
| percard_iforest | 0.52 ± 0.02 | 0.46 ± 0.00 | 0.01 ± 0.00 | 0.23 ± 0.01 |
| lane_xgb_or_percard | 0.97 ± 0.03 | 0.88 ± 0.01 | 0.01 ± 0.00 | 0.41 ± 0.04 |

## Unseen pattern (leave-one-pattern-out): EVENT recall on the held-out pattern

| held-out pattern | rules_naive | rules_cp | xgb | mlp | percard_iforest | lane_xgb_or_percard |
|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 0.98 ± 0.01 | 0.77 ± 0.04 | 1.00 ± 0.00 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CARD_TESTING | 1.00 ± 0.00 | 0.98 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CNP_SPREE | 0.78 ± 0.02 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.94 ± 0.02 | 1.00 ± 0.00 |
| ATM_CASHOUT | 1.00 ± 0.00 | 0.93 ± 0.02 | 1.00 ± 0.00 | 0.68 ± 0.14 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| SLOW_DRIP | 0.13 ± 0.05 | 0.00 ± 0.00 | 0.73 ± 0.10 | 0.66 ± 0.03 | 0.05 ± 0.02 | 0.56 ± 0.13 |
| LIMIT_SPLIT | 0.16 ± 0.02 | 0.01 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.93 ± 0.03 | 1.00 ± 0.00 |
| DORMANT_REACTIVATION | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.61 ± 0.22 | 0.50 ± 0.11 | 0.14 ± 0.06 | 0.56 ± 0.21 |
| FAMILIAR_MERCHANT_DRAIN | 0.09 ± 0.01 | 0.00 ± 0.00 | 0.85 ± 0.04 | 0.80 ± 0.05 | 0.12 ± 0.03 | 0.72 ± 0.04 |
| ODD_HOURS_QUIET | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.38 ± 0.17 | 0.44 ± 0.10 | 0.26 ± 0.08 | 0.28 ± 0.10 |

(legit FPR for each model in the LOPO runs is the same as in the time split.)
