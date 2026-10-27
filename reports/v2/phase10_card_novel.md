# Phase 10 -- card-transaction track (SYNTHETIC; seeds=3)

Fixed operating point: model thresholds calibrated at 1% FPR on validation legitimate traffic; lane uses 0.5% per component. Rules use fixed thresholds.

## Time split (train first 30 days, validate 30-42, test 42-60)

| model | event recall | txn recall | legit FPR | precision at 0.6% fraud |
|---|---|---|---|---|
| rules_naive | 0.404 ± 0.005 | 0.209 ± 0.004 | 0.025 ± 0.001 | 0.049 ± 0.001 |
| rules_cp | 0.373 ± 0.015 | 0.197 ± 0.004 | 0.004 ± 0.001 | 0.215 ± 0.021 |
| xgb | 0.897 ± 0.010 | 0.813 ± 0.017 | 0.006 ± 0.000 | 0.439 ± 0.010 |
| mlp | 0.851 ± 0.015 | 0.734 ± 0.016 | 0.007 ± 0.002 | 0.394 ± 0.050 |
| percard_iforest | 0.497 ± 0.018 | 0.430 ± 0.006 | 0.009 ± 0.001 | 0.217 ± 0.013 |
| lane_xgb_or_percard | 0.848 ± 0.013 | 0.769 ± 0.021 | 0.008 ± 0.001 | 0.381 ± 0.022 |

## Unseen pattern (leave-one-pattern-out): EVENT recall

| held-out pattern | rules_naive | rules_cp | xgb | mlp | percard_iforest | lane_xgb_or_percard |
|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 0.103 ± 0.073 | 0.077 ± 0.063 | 1.000 ± 0.000 | 0.656 ± 0.136 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| CARD_TESTING | 0.979 ± 0.029 | 0.958 ± 0.059 | 1.000 ± 0.000 | 0.958 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| CNP_SPREE | 0.058 ± 0.041 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.863 ± 0.033 | 0.611 ± 0.098 | 1.000 ± 0.000 |
| ATM_CASHOUT | 0.825 ± 0.046 | 0.825 ± 0.046 | 0.821 ± 0.056 | 0.070 ± 0.050 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| SLOW_DRIP | 0.024 ± 0.034 | 0.000 ± 0.000 | 0.321 ± 0.084 | 0.409 ± 0.169 | 0.000 ± 0.000 | 0.200 ± 0.123 |
| LIMIT_SPLIT | 0.056 ± 0.079 | 0.000 ± 0.000 | 0.948 ± 0.037 | 0.925 ± 0.059 | 0.663 ± 0.174 | 0.921 ± 0.068 |
| HIGH_VALUE_BUSTOUT | 0.954 ± 0.035 | 0.954 ± 0.035 | 0.759 ± 0.125 | 0.741 ± 0.148 | 0.722 ± 0.079 | 0.796 ± 0.086 |
| SPEND_SPIKE_FAMILIAR | 0.807 ± 0.072 | 0.807 ± 0.072 | 0.454 ± 0.066 | 0.601 ± 0.162 | 0.273 ± 0.049 | 0.478 ± 0.066 |
| RAPID_REPEAT | 0.623 ± 0.096 | 0.563 ± 0.122 | 0.978 ± 0.031 | 0.925 ± 0.056 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| UNUSUAL_MERCHANT | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.462 ± 0.102 | 0.419 ± 0.117 | 0.024 ± 0.034 | 0.371 ± 0.062 |
| DECLINE_THEN_APPROVE | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| NEW_CARD_FIRST_USE | 0.764 ± 0.086 | 0.764 ± 0.086 | 0.486 ± 0.086 | 0.542 ± 0.059 | 0.208 ± 0.059 | 0.486 ± 0.086 |
| DORMANT_REACTIVATION | 0.019 ± 0.026 | 0.000 ± 0.000 | 0.470 ± 0.100 | 0.507 ± 0.153 | 0.037 ± 0.026 | 0.451 ± 0.074 |
| FAMILIAR_MERCHANT_DRAIN | 0.139 ± 0.095 | 0.000 ± 0.000 | 0.550 ± 0.144 | 0.574 ± 0.213 | 0.117 ± 0.089 | 0.407 ± 0.121 |
| ODD_HOURS_QUIET | 0.033 ± 0.047 | 0.000 ± 0.000 | 0.171 ± 0.126 | 0.105 ± 0.029 | 0.176 ± 0.079 | 0.148 ± 0.123 |

(LOPO test sets contain future legitimate transactions and future fraud transactions from the held-out pattern.)
