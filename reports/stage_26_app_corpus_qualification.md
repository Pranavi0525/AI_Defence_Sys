# Stage 26 Qualification: AUTHORIZED PUSH PAYMENT (APP)

## 1. Objective
Determine whether the APP family is capable of producing a high-quality, diverse, difficulty-aware attack corpus suitable for future Blue Team evaluation.

## 2. Methodology
- Generative load testing to establish raw pool structural capabilities.
- Fingerprint isolation to identify unique structural variations.
- Diversity testing across bounded novelty constraints (max 20 attempts per target trace).
- Mathematical modeling of entropy ceilings given APP's rigid semantics.
- Full regression testing suite execution.

## 3. Structural Entropy Ceiling Discovery (The Bottleneck)
The audit identified a fundamental, mathematically demonstrable entropy ceiling built into the APP signature logic.

**Analysis of APP:**
By design, the APP signature correctly implements authorized push payment fraud by coercing the customer to use their own legitimate device and session. 
This means for the `APPAttackFingerprint`:
- `device_continuity` is ALWAYS `True`.
- `session_continuity` is ALWAYS `True`.
- New device anomalies are ALWAYS `0`.

With these axes frozen, the structural variation must rely entirely on:
- **Timing** (`RAPID`, `NORMAL`, `SLOW`, `BURSTY`)
- **Transaction Splitting** (`splits`)
- **Outcome Buckets** (Amount relative to balance)
- **Beneficiary Novelty** (New vs Reused)

**The `EASY` Difficulty Collapse:**
For an `EASY` difficulty APP attack, the configuration enforces:
- `timing_type` = `RAPID` (Fixed = 1 choice)
- `splits` = 1 (Fixed = 1 choice)
- `beneficiary_new_prob` = 1.0 (Fixed = 1 choice)
- `amount_buckets` = 3 to 4 typical variations (e.g., small, medium, large)
- `phase_sequence` = ~3 viable loop structures without triggering early ends.

**Mathematical Ceiling:**
Total combinations for EASY APP = 1 x 1 x 1 x 4 x 3 = **12 unique fingerprints.**
Because the structural space of an EASY APP trace is strictly limited to around a dozen unique forms, the Novelty Engine correctly rejects anything beyond the first dozen variants as non-novel duplicates. Therefore, attempting to generate a quota of 100 `EASY` APP traces mathematically exhausts the attempt budget and hits `BLOCKED`.

## 4. Raw Pool Results & Fingerprint Diversity
When generating thousands of traces with `use_novelty=False`:
- **Accepted:** The generator successfully produces raw traces, but exact-duplicate and near-duplicate rates skyrocket. 
- **Behavioral vs Numeric:** While absolute amounts and UUIDs vary (Numeric Noise), the underlying behavioral structures (Structural Diversity) collapse into identical patterns across the corpus, especially at lower difficulties.

## 5. Difficulty Analysis
- **EASY:** Collapses into rapid, single-transfer, full-balance depletion. Lack of splitting creates an immediate entropy bottleneck.
- **MEDIUM / HARD:** Gains some entropy via splitting (1-3 splits) and reused beneficiaries.
- **ADVANCED:** Achieves the most diversity through heavy loops, bursty timing, and split transactions, but still fundamentally underperforms ATO's structural diversity due to lacking device/session variations. 

## 6. Novelty Saturation & Sensitivity
Attempting to generate 100 novel traces per difficulty bucket:
- **EASY:** `BLOCKED` (Exhausts budget due to the ~12 fingerprint ceiling).
- **MEDIUM:** Highly saturated, severe attempt inflation.
- **HARD / ADVANCED:** Succeeds but with elevated rejection rates compared to ATO.

Controlled mutations verify that the Novelty Engine correctly ignores numerical noise (UUIDs, $1 offsets) and successfully recognizes splits and timing adjustments as novel. The limitation is entirely within the Generative constraint, not the Validator.

## 7. Realism Preservation
For generated accepted traces:
- `balance_valid`: Maintained strictly (failed traces correctly do not alter balances; completed traces deduct correctly).
- `beneficiary_order_valid`: Enforced (payee exists prior to execution).
- `session_order_valid`: Enforced. 
- **Adversarial Mutations:** Any mutation causing a corrupted balance transition or chronological reversal was successfully caught and rejected by the Realism Validator.

## 8. APP vs ATO Comparison
The two families are distinct and realistic:
- **APP Signal:** Valid device + legitimate session + anomalous payee setup + velocity spike + high-value depletion.
- **ATO Signal:** Unrecognized device + parallel login/anomaly + rapid modification + broad depletion.
A detector can cleanly segregate these behaviors without relying on ground-truth `attack_family` flags.

## 9. Performance & Reproducibility
- Performance suffers exponentially as `target_count` increases with `use_novelty=True` because the attempt budget is rapidly consumed by non-novel duplicates hitting the entropy ceiling.
- Reproducibility is 100% exact given a fixed `master_seed`.

## 10. Regression
`PYTHONPATH=src pytest tests/ -v -W error::DeprecationWarning`
**Result:** 464 / 464 tests passed (0 failures).

## 11. Limitations & Future Requirements
To resolve the APP entropy ceiling and allow 100+ novel variations across all difficulties, the simulator requires new behavioral dimensions explicitly modeled for social engineering.
**Required Dimensions for Future Work:**
- **Hesitation / Interaction Metrics:** Behavioral biometrics (e.g., time-on-page, back-and-forth navigation, repeated app opens) mimicking victim duress or live-call guidance.
- **Concurrent Channel Activity:** Explicitly modeling SMS OTP, MFA fatigue, or concurrent phone call indicators in the `NormalWorld`.
- **Pre-staging Velocity:** Tracking the speed at which funds are consolidated from secondary accounts into the primary checking account before the final push.

## 12. Final Decision
**QUALIFIED_WITH_LIMITATIONS**

The implementation is functionally perfect, realistic, logically sound, and distinct from ATO. However, due to the mathematically constrained nature of legitimate-device push payments, its structural entropy is capped, particularly at the `EASY` difficulty. It is certified for use in the corpus but cannot currently satisfy high-volume novelty targets without introducing new behavioral dimensions.
