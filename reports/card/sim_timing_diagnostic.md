# Card simulator timing diagnostic (30 seeds, n_cards=1500, days=60, events_per_pattern=40, novel=True)

Gap = first fraudulent txn of an event minus the last LEGITIMATE txn of the same card before it. Minutes.
`in-window` = share of events whose gap is in [10, 30] min (the original artifact). `AUC(ind)` = AUC of the in-window yes/no indicator on the `gap_s` feature (the measure used in reports/card/large_amount_investigation.md: 0.946 before, 0.513 after); `AUC(win)` = AUC of a continuous 'closeness to 20 min' score (more sensitive). 0.5 = no information.

## extended=False

Legit baseline: gap-to-previous-legit median 705.0 min, mean 1,136.4, in-window share 0.028 (legit rows between 02:00 and 02:30: 0.005).

| pattern | events | min | p10 | median | mean | p90 | max | no prior legit | in-window | AUC(ind) | AUC(win) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 1200 | 0.1 | 12.0 | 19.5 | 202.5 | 28.0 | 26,957.2 | 0 | 0.976 | 0.972 | 0.969 |
| CARD_TESTING | 1200 | 0.0 | 11.8 | 20.1 | 67.9 | 28.0 | 17,171.6 | 0 | 0.981 | 0.975 | 0.978 |
| CNP_SPREE | 1200 | 0.2 | 11.6 | 20.0 | 94.8 | 27.7 | 22,980.6 | 0 | 0.979 | 0.975 | 0.976 |
| ATM_CASHOUT | 1200 | 0.3 | 11.8 | 20.2 | 96.8 | 28.1 | 21,528.3 | 0 | 0.982 | 0.976 | 0.976 |
| SLOW_DRIP | 1200 | 0.7 | 11.7 | 19.2 | 107.8 | 27.9 | 19,899.2 | 0 | 0.978 | 0.975 | 0.976 |
| LIMIT_SPLIT | 1200 | 0.7 | 11.9 | 20.2 | 117.6 | 28.5 | 20,661.9 | 0 | 0.979 | 0.974 | 0.975 |
| DORMANT_REACTIVATION | 1200 | 21,600.5 | 21,714.3 | 22,513.7 | 23,189.1 | 25,453.4 | 48,076.0 | 0 | 0.000 | 0.486 | 0.006 |
| FAMILIAR_MERCHANT_DRAIN | 1200 | 0.5 | 11.8 | 20.3 | 81.0 | 28.3 | 23,810.8 | 0 | 0.985 | 0.978 | 0.979 |
| ODD_HOURS_QUIET | 1200 | 1.2 | 277.7 | 807.3 | 1,587.8 | 3,512.7 | 22,566.9 | 0 | 0.005 | 0.489 | 0.416 |

## extended=True

Legit baseline: gap-to-previous-legit median 682.3 min, mean 1,092.9, in-window share 0.029 (legit rows between 02:00 and 02:30: 0.005).

| pattern | events | min | p10 | median | mean | p90 | max | no prior legit | in-window | AUC(ind) | AUC(win) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLONED_CARD_POS | 1166 | 1.0 | 59.4 | 461.4 | 875.1 | 1,877.5 | 34,424.9 | 0 | 0.035 | 0.504 | 0.570 |
| CARD_TESTING | 1165 | 0.9 | 64.8 | 446.0 | 780.0 | 1,711.7 | 18,378.8 | 0 | 0.033 | 0.503 | 0.582 |
| CNP_SPREE | 1170 | 0.9 | 57.7 | 461.6 | 810.7 | 1,800.3 | 23,609.1 | 0 | 0.039 | 0.505 | 0.575 |
| ATM_CASHOUT | 1174 | 0.9 | 58.3 | 439.3 | 759.0 | 1,694.8 | 22,810.2 | 0 | 0.037 | 0.504 | 0.588 |
| SLOW_DRIP | 1172 | 0.1 | 56.5 | 418.4 | 722.3 | 1,667.7 | 14,869.0 | 0 | 0.039 | 0.506 | 0.593 |
| LIMIT_SPLIT | 1176 | 0.6 | 57.0 | 426.0 | 842.1 | 1,748.7 | 26,496.5 | 0 | 0.037 | 0.504 | 0.583 |
| DORMANT_REACTIVATION | 1197 | 21,600.5 | 21,709.1 | 22,462.5 | 23,099.1 | 25,219.3 | 48,076.0 | 46 | 0.000 | 0.486 | 0.004 |
| FAMILIAR_MERCHANT_DRAIN | 1171 | 0.6 | 56.4 | 463.9 | 739.5 | 1,597.6 | 13,909.7 | 0 | 0.032 | 0.502 | 0.581 |
| ODD_HOURS_QUIET | 1199 | 1.2 | 276.3 | 776.5 | 1,490.2 | 3,435.4 | 22,566.9 | 1 | 0.003 | 0.488 | 0.420 |
| HIGH_VALUE_BUSTOUT | 1161 | 0.6 | 47.4 | 398.8 | 757.4 | 1,755.9 | 21,959.3 | 0 | 0.035 | 0.503 | 0.596 |
| SPEND_SPIKE_FAMILIAR | 1177 | 0.5 | 65.3 | 438.6 | 790.9 | 1,767.7 | 20,080.4 | 0 | 0.031 | 0.501 | 0.577 |
| RAPID_REPEAT | 1176 | 1.0 | 53.4 | 427.6 | 768.7 | 1,617.4 | 21,941.2 | 0 | 0.046 | 0.509 | 0.584 |
| UNUSUAL_MERCHANT | 1174 | 0.4 | 62.0 | 414.1 | 806.9 | 1,770.0 | 20,000.5 | 0 | 0.037 | 0.504 | 0.587 |
| DECLINE_THEN_APPROVE | 1165 | 0.5 | 56.0 | 458.3 | 840.3 | 1,808.4 | 25,118.6 | 0 | 0.040 | 0.506 | 0.576 |
| NEW_CARD_FIRST_USE | 1200 | 1.8 | 75.3 | 492.9 | 604.5 | 1,327.0 | 1,798.5 | 145 | 0.027 | 0.502 | 0.602 |
