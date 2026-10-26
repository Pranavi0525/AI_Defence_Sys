# STAGE 4.5D — IEEE-CIS PROXY SENSITIVITY ANALYSIS

## Overview
This report formalizes the evidence for Stage 4.5D, replacing the previous reliance on "historical reconstruction." The analysis evaluates how sensitive the baseline Logistic Regression model's performance is to the presence of extremely large entities in the IEEE-CIS proxy grouping (`card_addr_email`).

## Scope & Reproducibility
- **Script:** `src/red_team/ml/analyze_sensitivity.py`
- **Output Data:** `data/reference/ml_sequence/sensitivity_results/res_{a,b,c,d}.json`
- **Integrity Check:** The script has been rerun and strictly reproduces the recorded JSON metrics exactly, confirming zero drift.

## Methodology
The analysis evaluates the original baseline model (`models/normal_behavior/logistic_regression/model.joblib`) on the Stage 4.5C `ml_sequence` test split under four different inclusion conditions based on global transaction volume percentiles.

**Global Entity Volume Distribution:**
- **Median:** 3.0 transactions
- **P90:** 18.0 transactions
- **P99:** 120.0 transactions
- **P99.9:** 584.4 transactions
- **Max:** 3922 transactions

## Results

### Condition A (Baseline - All Entities)
- **Entities:** 7,086 | **Rows:** 74,598
- **Macro F1:** 0.6836

### Condition B (Exclude > P99)
- *Excludes the top 1% of highly active entities (>120 txns).*
- **Entities:** 7,017 | **Rows:** 50,493 (Removed 24,105 rows, ~32% of test set)
- **Macro F1:** 0.7279 (**+0.0443** vs A)

### Condition C (Exclude > P99.5)
- *Excludes the top 0.5% (>197 txns).*
- **Entities:** 7,051 | **Rows:** 55,357
- **Macro F1:** 0.7219 (**+0.0383** vs A)

### Condition D (Exclude > P99.9)
- *Excludes only the top 0.1% (>584 txns).*
- **Entities:** 7,078 | **Rows:** 63,492 (Removed ~11,000 rows)
- **Macro F1:** 0.7100 (**+0.0264** vs A)

## Per-Class Sensitivity (Baseline vs Condition D)
Comparing Condition A (all entities) to Condition D (removing just the top 0.1% largest entities):
- **C:** Rec=0.9986->0.9973 (-0.0013), F1=0.9987->0.9975 (-0.0012)
- **H:** Rec=0.3275->0.3373 (+0.0098), F1=0.3904->0.4021 (+0.0116)
- **R:** Rec=0.4400->0.4467 (+0.0067), F1=0.4862->0.4927 (+0.0065)
- **S:** Rec=0.4924->0.7175 (**+0.2251**), F1=0.6006->0.7077 (**+0.1071**)
- **W:** Rec=0.9647->0.9641 (-0.0005), F1=0.9420->0.9500 (+0.0080)

*Note: Removing just the top 0.1% largest entities causes the Recall for class `S` to jump by 22.5 points, indicating that the model's performance on minority classes was heavily suppressed by massive entities dominating the sequence distributions.*

## Conclusion
The model is extremely sensitive to proxy definition size cutoffs. A minuscule fraction of entities (the top 0.1% to 1%) account for up to 32% of the total transaction volume. Because these massive entities are structurally different (likely corporate gateways or IP collisions rather than individual behavior), they aggressively skew the model's predictive capabilities, especially suppressing minority classes like `S`. This justifies the need for the Stage 4.5E behavioral proxy redesign to separate genuine coherent behavior from diffuse gateway collisions.
