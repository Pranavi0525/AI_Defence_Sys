# STAGE 4.5E ADDENDUM: COMPOSITE PAYMENT CONTEXT VALIDATION

## 1. RECURRENCE
- **Median**: 2.0
- **Mean**: 6.12
- **Std**: 34.40
- **P90**: 10.0
- **P95**: 19.0
- **P99**: 71.0
- **P99.9**: 362.86
- **Max**: 3923

**Comparison to Prior Record**: 
The prior individual-proxy analysis reported slightly different extreme percentiles (e.g., P99=120) because it was computed over the temporally partitioned sequence dataset splits (which aggregated all historical events leading up to a target). This metric was computed independently on the raw `train_transaction.csv` filtered strictly to `isFraud == 0`, representing the true stationary distribution of the raw proxy.

## 2. SEMANTIC COHERENCE
We evaluated the coherence of the proxy using two signals:
- **Product Purity**: The maximum probability of a single `ProductCD` within the entity (1.0 = completely homogeneous, lower = mixed).
- **Amount CV**: The Coefficient of Variation (Std / Mean) of `TransactionAmt` within the entity.

**Findings across size buckets**:
- **Size 2-5**: Purity = 0.927, Amount CV = 0.436
- **Size 11-50**: Purity = 0.919, Amount CV = 0.841
- **Size 51-100**: Purity = 0.916, Amount CV = 1.126
- **Size 101-500**: Purity = 0.920, Amount CV = 1.291
- **Size 501-1000**: Purity = 0.960, Amount CV = 1.280
- **Size 1000+**: Purity = 0.942, Amount CV = 1.203

**Conclusion**: Semantic coherence remains remarkably stable and high across all sizes. Even mega-entities (1000+ transactions) exhibit >94% purity in `ProductCD`. While amount variation increases for larger entities, it stabilizes around 1.2, which is highly consistent with a coherent recurring payment context (e.g., corporate purchasing or localized institutional traffic) rather than an incoherent mix of random traffic.

## 3. COLLISION CHARACTERISTICS
For entities above the median count (>2), the device collision footprint is:
- **Mean Unique Devices**: 1.00
- **P50 Unique Devices**: 0.0 (mostly null/missing `DeviceInfo`)
- **P90 Unique Devices**: 2.0
- **P99 Unique Devices**: 7.0
- **Max Unique Devices**: 400.0

**Average Unique Devices by Bucket**:
- **Size 11-50**: 1.17
- **Size 51-100**: 3.02
- **Size 101-500**: 8.26
- **Size 1000+**: 98.37

**Conclusion**: The proxy definitively collapses unrelated devices at high frequencies. However, under the "payment context" interpretation, this is explicitly a "shared proxy value colliding across multiple devices" (e.g., a corporate gateway or regional BIN slice), not a single individual using 100 devices.

## 4. TEMPORAL PERSISTENCE
- **Size 2-5**: Avg Span = 52.8 days, Avg Gap = 31.0 days
- **Size 11-50**: Avg Span = 143.3 days, Avg Gap = 8.5 days
- **Size 101-500**: Avg Span = 175.5 days, Avg Gap = 1.07 days
- **Size 1000+**: Avg Span = 181.6 days, Avg Gap = 0.12 days (approx. 2.8 hours)

**Conclusion**: High-frequency entities demonstrate perfect temporal persistence across the entire 6-month window. The gaps shrink smoothly to a continuous stream (every few hours). They are decidedly *not* burst/collision artifacts (which would clump into implausibly short windows), but rather represent a steady, continuous real-world traffic context.

## 5. THRESHOLD NECESSITY
For an *individual-customer* objective, a threshold (e.g., P99) was mandatory because 100+ devices fundamentally violates the human construct.
However, for the **COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR** objective, extreme entities do not violate the construct. They possess exceptionally high semantic purity (>94% ProductCD consistency) and smooth, continuous temporal persistence. The high device count correctly reflects the reality of the shared payment instrument slice.

**Conclusion**: NO threshold is required or justified for this objective. Filtering extreme entities would artificially truncate valid, highly coherent high-volume payment contexts.

---

**ORIGINAL VERDICT**: APPROPRIATE

---

# REVISION (Addendum Update)

## 1. BASELINE / NULL COMPARISON FOR COHERENCE
To rigorously test coherence, we computed the global `ProductCD` distribution on all legitimate transactions (`W`: 75.6%, `C`: 10.6%, `R`: 6.4%, `H`: 5.5%, `S`: 1.9%). We then used Monte Carlo simulation to calculate the expected purity if transactions were randomly grouped into entities of the same size. 

*Observed Purity vs Expected-Under-Random:*
- **Size 2-5** (30,098 entities): **92.7%** observed vs **77.7%** expected
- **Size 6-20** (12,775 entities): **92.0%** observed vs **75.6%** expected
- **Size 21-100** (3,784 entities): **91.6%** observed vs **75.5%** expected
- **Size 101-1000** (563 entities): **92.3%** observed vs **75.5%** expected
- **Size 1000+** (16 entities): **94.3%** observed vs **75.6%** expected

**Meaningful margin**: Across all buckets, observed purity exceeds the random baseline by +15 to +19 percentage points. Furthermore, the P50 purity for almost all buckets is **100%**. This confirms that the proxy entities represent highly coherent contextual boundaries, crushing the null hypothesis.

## 2. RECONCILE AMOUNT CV WITH THE COHERENCE CLAIM
- **Global Baseline Amount CV**: 1.78
- **Bucket CV Means (P10 - P50 - P90)**:
  - **Size 2-5**: 0.44 (0.00 - 0.38 - 0.98)
  - **Size 6-20**: 0.70 (0.27 - 0.62 - 1.24)
  - **Size 21-100**: 0.99 (0.50 - 0.89 - 1.64)
  - **Size 101-1000**: 1.29 (0.81 - 1.18 - 1.89)
  - **Size 1000+**: 1.20 (0.88 - 1.28 - 1.68)

**Implication**: Amount CV *supports* the coherence claim, but offers a separate signal. At all entity sizes, the mean CV is substantially lower (tighter) than the global baseline of 1.78. As bucket size increases, the CV rises to ~1.2 but never reaches global variance. This indicates that while larger entities are categorically homogeneous (ProductCD), their transaction amounts behave like a bounded pool of diverse purchases, exactly as expected for a composite payment context (e.g., a corporate BIN).

## 3. DEVICE COLLISION: STABILITY TEST
For mega-entities (Size 1000+), we tested whether the device set is stable by calculating the percentage of transactions covered by the top 3 devices per entity.
- **Mean Top-3 Coverage**: 84.9%
- **P50 Top-3 Coverage**: 96.1%
- **P10 - P90**: 54.4% - 98.2%

**Implication**: Despite averaging 98 unique devices, the median mega-entity routes **96.1% of its traffic through just 3 devices**. This proves the device footprint is heavily STABLE. It reflects a shared instrument where a few primary gateways/systems handle the vast majority of volume, rather than a diffuse, random collision of unrelated devices.

## 4. CONFIRM RECURRENCE COMPUTATION INDEPENDENCE
As verified in the script execution log:
- The recurrence statistics were computed **FRESH** in this run directly from `train_transaction.csv`.
- **Data Lineage**: 590,540 input rows → 569,877 rows (isFraud==0) → 93,073 unique proxy entities.
- Zero dependency on prior individual-proxy artifacts or thresholds.

## 5. THRESHOLD NECESSITY & FINAL VERDICT
With baseline-adjusted coherence crushing the null hypothesis, amount variance bounded below global levels, and mega-entity device footprints proving highly stable, there is no empirical justification to filter or cap extreme entities under the `COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR` objective.

## 6. 101-1000 BUCKET DEVICE DRILLDOWN (Boundary Verification)
To determine if the diffuse device collision pattern is unique to mega-entities (size > 1000) or if it exists at smaller sizes, we applied the exact same strict device analysis to the 101-1000 bucket (563 entities) and classified them using the following criteria (mirroring the mega-bucket clusters):
- **CONFIRMED_DIFFUSE**: `device_match_rate` >= 50% AND `top3_known_coverage` < 60%
- **CONFIRMED_STABLE**: `device_match_rate` >= 20% AND `top3_known_coverage` >= 80%
- **INSUFFICIENT_EVIDENCE**: `device_match_rate` < 20%
- **AMBIGUOUS**: Any other combination

**Findings in the 101-1000 Bucket:**
- **INSUFFICIENT_EVIDENCE**: 460 entities (101,960 rows)
- **CONFIRMED_DIFFUSE**: 42 entities (16,116 rows)
- **CONFIRMED_STABLE**: 32 entities (6,020 rows)
- **AMBIGUOUS**: 29 entities (6,216 rows)

**Inflection Point**: The `CONFIRMED_DIFFUSE` pattern (heavily populated device fields but highly fragmented coverage) appears starting at size **103** in this bucket, and further analysis shows it extends all the way down to size **12** in smaller buckets. 

**Conclusion**: The diffuse gateway behavior is NOT a function of raw size; it is a structural anomaly of specific internet gateways or payment processors. A size cutoff of 1000 is entirely arbitrary and fails to catch 16,116 rows of known diffuse traffic in the 100-999 range.

## 7. RECOMMENDATION & VERDICT UPDATE
Because diffuse entities appear across multiple size buckets, any raw size cutoff is empirically unsupportable. Instead, the boundary should be redefined by a **DIRECT RULE**:

> **Rule-Based Exclusion**: Exclude any entity where `device_match_rate >= 50%` AND `top3_known_coverage < 60%`, regardless of size.

Applying this rule dataset-wide surgically targets exactly **100 entities** (out of 93,073) representing **30,023 rows**, cleaning the dataset of confirmed messy gateways while preserving all stable and data-sparse traffic.

### Verdict Table

| Entity Size Range | Recommendation | Evidence Basis |
|---|---|---|
| **2-5** | APPROPRIATE | 0 diffuse entities found. Coherence margins crush null baseline. |
| **6-20** | CONDITIONAL (Rule-Exclusion) | 17 diffuse entities found (217 rows). Remaining traffic is stable. |
| **21-100** | CONDITIONAL (Rule-Exclusion) | 36 diffuse entities found (1,987 rows). Remaining traffic is stable. |
| **101-1000** | CONDITIONAL (Rule-Exclusion) | 42 diffuse entities found (16,116 rows). Remaining traffic is stable/sparse. |
| **1000+** | CONDITIONAL (Rule-Exclusion) | 5 diffuse entities found (11,703 rows). Remaining 11 entities are sparse or stable. |

**REVISED VERDICT**: CONDITIONAL. The `COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR` proxy is a defensible objective *only if* the 100 empirically-identified `CONFIRMED_DIFFUSE` entities are excluded using the direct rule above. Raw size cutoffs should be abandoned.
