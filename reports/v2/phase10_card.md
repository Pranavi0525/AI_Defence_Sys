# Phase 10 -- card-transaction track (SYNTHETIC; seeds=3)

Fixed operating point: model thresholds calibrated at 1% FPR on validation legitimate traffic; lane uses 0.5% per component. Rules use fixed thresholds.

## Time split (train first 30 days, validate 30-42, test 42-60)

| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |
|---|---|---|---|---|
| rules_naive | 0.509 ± 0.020 | 0.261 ± 0.009 | 0.025 ± 0.001 | 0.060 ± 0.003 |
| rules_cp | 0.490 ± 0.021 | 0.250 ± 0.007 | 0.004 ± 0.000 | 0.259 ± 0.028 |
| xgb | 0.898 ± 0.025 | 0.859 ± 0.020 | 0.006 ± 0.000 | 0.455 ± 0.002 |
| mlp | 0.880 ± 0.036 | 0.817 ± 0.024 | 0.009 ± 0.002 | 0.367 ± 0.048 |
| percard_iforest | 0.634 ± 0.027 | 0.534 ± 0.028 | 0.009 ± 0.001 | 0.256 ± 0.014 |
| lane_xgb_or_percard | 0.867 ± 0.015 | 0.829 ± 0.025 | 0.007 ± 0.001 | 0.408 ± 0.028 |

## Unseen pattern (leave-one-pattern-out): EVENT recall

| held-out pattern | rules_naive | rules_cp | xgb | mlp | percard_iforest | lane_xgb_or_percard |
|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 0.103 ± 0.073 | 0.077 ± 0.063 | 0.944 ± 0.079 | 0.741 ± 0.156 | 1.000 ± 0.000 | 0.972 ± 0.039 |
| CARD_TESTING | 0.979 ± 0.029 | 0.958 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| CNP_SPREE | 0.058 ± 0.041 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.893 ± 0.095 | 0.611 ± 0.098 | 1.000 ± 0.000 |
| ATM_CASHOUT | 0.825 ± 0.046 | 0.825 ± 0.046 | 1.000 ± 0.000 | 0.411 ± 0.339 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| SLOW_DRIP | 0.024 ± 0.034 | 0.000 ± 0.000 | 0.343 ± 0.112 | 0.464 ± 0.162 | 0.000 ± 0.000 | 0.258 ± 0.078 |
| LIMIT_SPLIT | 0.056 ± 0.079 | 0.000 ± 0.000 | 0.770 ± 0.114 | 0.778 ± 0.151 | 0.639 ± 0.160 | 0.726 ± 0.061 |
| HIGH_VALUE_BUSTOUT | 0.954 ± 0.035 | 0.954 ± 0.035 | 0.759 ± 0.125 | 0.824 ± 0.080 | 0.769 ± 0.173 | 0.796 ± 0.086 |
| SPEND_SPIKE_FAMILIAR | 0.807 ± 0.072 | 0.807 ± 0.072 | 0.443 ± 0.011 | 0.594 ± 0.097 | 0.418 ± 0.155 | 0.394 ± 0.027 |
| RAPID_REPEAT | 0.623 ± 0.096 | 0.563 ± 0.122 | 0.950 ± 0.036 | 0.842 ± 0.067 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| UNUSUAL_MERCHANT | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.495 ± 0.169 | 0.510 ± 0.159 | 0.024 ± 0.034 | 0.324 ± 0.054 |
| DECLINE_THEN_APPROVE | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| NEW_CARD_FIRST_USE | 0.764 ± 0.086 | 0.764 ± 0.086 | 0.542 ± 0.156 | 0.583 ± 0.118 | 0.222 ± 0.039 | 0.514 ± 0.119 |

(LOPO test sets contain future legitimate transactions and future fraud transactions from the held-out pattern.)
