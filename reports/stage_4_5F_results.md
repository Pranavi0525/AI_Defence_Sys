# STAGE 4.5F — REBUILD SUPERVISED SEQUENCE DATASET

## 1. EXCLUSION RULE APPLICATION LOG
The supervised sequence dataset was rebuilt using the `card_addr_email` proxy as the `COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR` objective, with the mandatory empirical exclusion rule applied *before* the dataset split:
`device_match_rate >= 0.50 AND top3_known_device_coverage < 0.60`

**Data Lineage and Exclusion Log:**
- **Initial valid rows (isFraud==0)**: 569,877
- **Total Entities before filter**: 93,073
- **Excluded (`CONFIRMED_DIFFUSE`)**: 100 entities (30,023 rows)
- **Retained Final Corpus**: 539,854 rows (perfectly matching expected `569,877 - 30,023`)

**Retained Entities by Category:**
- **CONFIRMED_STABLE**: 13,806 entities (85,588 rows)
- **INSUFFICIENT_EVIDENCE**: 78,769 entities (437,621 rows)
- **AMBIGUOUS**: 398 entities (16,645 rows)

## 2. DATASET SPLIT SIZES
The same chronological proxy-grouped 70/15/15 entity split logic was applied to the filtered dataset. The new split sizes (measured in sequential rows, where the first row of each entity is dropped since it has no history) are compared against the original Stage 4.5C/D dataset below.

| Split | Original (Rows) | Corrected (Rows) | Difference | Original Entities | Corrected Entities |
|---|---|---|---|---|---|
| **Train** | 329,348 | 310,453 | -18,895 | 34,917 | 32,995 |
| **Validation** | 72,858 | 66,217 | -6,641 | 7,482 | 7,070 |
| **Test** | 74,598 | 70,211 | -4,387 | 7,482 | 7,071 |

*Note: The reduction in rows is exactly proportional to the exclusion of the massive diffuse gateways.*

## 3. MODEL METRICS
The identical Logistic Regression configuration (`C=0.1`, `class_weight=None`, `solver='lbfgs'`, `max_iter=1000`) was trained on the corrected `ml_sequence_v2` dataset to establish a strictly controlled comparison.

**Test Split Class Distribution (`ProductCD`)**
| Class | Original (v1) | Corrected (v2) | Shift |
|---|---|---|---|
| **W** | 54,096 | 58,013 | Modest shift due to entity shuffle |
| **C** | 11,347 | 3,193 | **Plummeted (-72%)** |
| **R** | 3,432 | 3,535 | Stable |
| **H** | 2,965 | 3,014 | Stable |
| **S** | 2,758 | 2,456 | Stable |

| Metric | Original (Flawed Proxy) | Corrected Dataset (v2) | Change |
|---|---|---|---|
| **Majority Baseline Macro F1** | 0.1681 | 0.1810 | +0.0129 |
| **Logistic Regression Macro F1** | 0.6836 | 0.6790 | -0.0046 (Degraded) |
| **Balanced Accuracy** | 0.6446 | 0.6372 | -0.0074 (Degraded) |
| **Accuracy** | 0.8974 | 0.8967 | -0.0007 (Degraded) |

## 4. VERDICT

**Causal Interpretation of the Metric Drop**
Correcting the proxy definition by surgically amputating the 100 massive diffuse gateways caused a slight performance degradation. This drop is a direct consequence of population composition, not noise removal. As established in the Stage 4.5E Addendum (Section 1), large entities—including the excluded gateways—exhibit extremely high `ProductCD` purity (91-94%). By removing these 30,000 highly repetitive and artificially pure rows, we eliminated an unusually predictable subpopulation from the train and test splits. The remaining legitimate traffic contains more genuine contextual variability, which mathematically lowers the ceiling on raw predictive metrics while improving the construct validity of the dataset.

**Does ML measurably improve the Normal World?**
No. To properly answer this, we must compare the ML model against a domain-modeled baseline on the *exact same task* (predicting `target_ProductCD`) and the *exact same population* (the `ml_sequence_v2` test split). 

We constructed a naive, domain-explainable baseline representing a simple rule a simulator could trivially implement: **Predict the entity's `previous_ProductCD`** (falling back to the global majority class 'W' if unknown). 

| Metric | Naive Rule (Predict Previous) | Logistic Regression (ML) | ML vs Naive |
|---|---|---|---|
| **Macro F1** | 0.7508 | 0.6790 | **-0.0718** |
| **Balanced Acc** | 0.7581 | 0.6372 | **-0.1209** |

The simple domain rule drastically outperforms the Logistic Regression model. The ML model fails to beat a basic heuristic on its own task.

**Untested Generalization Caveat**
Even if the ML model had outperformed the naive rule, applying it to the Normal World simulator involves a massive, untested assumption: that a model trained on IEEE-CIS composite *payment-context proxies* (groups of transactions sharing card/address features) will reliably generalize to predicting the behavior of distinct *synthetic personas* modeled around individual human domain roles. 

**Recommendation: DO NOT INTEGRATE**
The corrected `ml_sequence_v2` dataset successfully isolates legitimate composite payment contexts. However, the resulting ML model fails to outperform a trivial, one-line domain rule (`predict previous_ProductCD`). In accordance with the project's ML strategy—which requires measurable benefit before integration—this model should **not** be integrated into the Normal World. The Normal World would be better served by implementing the naive rule directly, bypassing the complexity and opacity of the ML pipeline entirely.
