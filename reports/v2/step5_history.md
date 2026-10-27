# Step 5 -- per-customer history in World B (SYNTHETIC; seeds [2026, 2027, 2028])

Unseen-family recall (held-out family never in training) at the fixed operating points; `lane` = XGBoost OR IsolationForest. Format mean±sd over world seeds.


## Held-out: SYNTHETIC_IDENTITY

| setting | xgb recall | iforest recall | lane recall | lane false alarms | in-dist F1 |
|---|---|---|---|---|---|
| original world (no history) | 0.06±0.05 | 0.14±0.02 | 0.19±0.05 | 0.05±0.00 | 0.840 |
| strength 0.0, no history features | 0.05±0.05 | 0.15±0.02 | 0.19±0.04 | 0.05±0.00 | 0.837 |
| strength 0.5, no history features | 0.04±0.03 | 0.13±0.02 | 0.16±0.03 | 0.05±0.00 | 0.833 |
| strength 1.0, no history features | 0.04±0.03 | 0.12±0.03 | 0.15±0.02 | 0.05±0.00 | 0.828 |
| strength 0.0, WITH history features | 0.13±0.03 | 0.17±0.03 | 0.26±0.01 | 0.06±0.00 | 0.842 |
| strength 0.5, WITH history features | 0.13±0.03 | 0.17±0.03 | 0.26±0.01 | 0.05±0.01 | 0.868 |
| strength 1.0, WITH history features | 0.19±0.02 | 0.16±0.02 | 0.29±0.00 | 0.05±0.00 | 0.901 |

## Held-out: FAKE_MERCHANT

| setting | xgb recall | iforest recall | lane recall | lane false alarms | in-dist F1 |
|---|---|---|---|---|---|
| original world (no history) | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.05±0.00 | 0.840 |
| strength 0.0, no history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.05±0.00 | 0.837 |
| strength 0.5, no history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.06±0.00 | 0.833 |
| strength 1.0, no history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.06±0.00 | 0.828 |
| strength 0.0, WITH history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.05±0.00 | 0.842 |
| strength 0.5, WITH history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.05±0.00 | 0.868 |
| strength 1.0, WITH history features | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.05±0.00 | 0.901 |

## Held-out: DEEPFAKE_KYC

| setting | xgb recall | iforest recall | lane recall | lane false alarms | in-dist F1 |
|---|---|---|---|---|---|
| original world (no history) | 0.20±0.01 | 0.41±0.08 | 0.43±0.08 | 0.06±0.01 | 0.840 |
| strength 0.0, no history features | 0.21±0.03 | 0.41±0.08 | 0.43±0.07 | 0.06±0.00 | 0.837 |
| strength 0.5, no history features | 0.19±0.02 | 0.39±0.09 | 0.42±0.08 | 0.06±0.00 | 0.833 |
| strength 1.0, no history features | 0.15±0.00 | 0.36±0.06 | 0.38±0.06 | 0.06±0.00 | 0.828 |
| strength 0.0, WITH history features | 0.20±0.02 | 0.30±0.07 | 0.35±0.06 | 0.06±0.00 | 0.842 |
| strength 0.5, WITH history features | 0.17±0.01 | 0.27±0.07 | 0.32±0.06 | 0.05±0.00 | 0.868 |
| strength 1.0, WITH history features | 0.14±0.01 | 0.21±0.08 | 0.27±0.06 | 0.05±0.00 | 0.901 |

## Held-out: AI_SCRIPTED_APP

| setting | xgb recall | iforest recall | lane recall | lane false alarms | in-dist F1 |
|---|---|---|---|---|---|
| original world (no history) | 0.02±0.01 | 0.25±0.01 | 0.26±0.01 | 0.05±0.00 | 0.840 |
| strength 0.0, no history features | 0.02±0.01 | 0.25±0.01 | 0.25±0.01 | 0.05±0.00 | 0.837 |
| strength 0.5, no history features | 0.01±0.01 | 0.26±0.01 | 0.26±0.01 | 0.05±0.01 | 0.833 |
| strength 1.0, no history features | 0.01±0.01 | 0.23±0.02 | 0.23±0.02 | 0.05±0.01 | 0.828 |
| strength 0.0, WITH history features | 0.05±0.02 | 0.28±0.01 | 0.28±0.02 | 0.05±0.00 | 0.842 |
| strength 0.5, WITH history features | 0.05±0.03 | 0.23±0.01 | 0.25±0.01 | 0.05±0.01 | 0.868 |
| strength 1.0, WITH history features | 0.10±0.03 | 0.26±0.02 | 0.28±0.01 | 0.05±0.00 | 0.901 |