# Stage 27: APP Social-Engineering Behavioral Dimension Design

## 1. Executive Summary
The Stage 26 audit demonstrated an entropy bottleneck in the `AUTHORIZED_PUSH_PAYMENT` (APP) signature, particularly at the `EASY` difficulty tier. Because APP enforces legitimate-device and legitimate-session continuity, its observable behavioral variance is highly constrained compared to `ACCOUNT_TAKEOVER`. 
Following a detailed investigation into the semantic representation capabilities of the existing system, this design document proposes the addition of **hesitation tracking**, **outcome retries**, and **amount escalation trends** to the simulator and fingerprint.

**Verdict:** `NO_SCHEMA_CHANGE_REQUIRED`. The current event schemas perfectly support these behavioral dimensions via timestamps, transaction statuses, and amount sequences.

## 2. Current APP Model & Entropy Bottleneck
**Currently Observable State Lifecycle:**
1. `SESSION_LOGIN` (Known Device)
2. `BENEFICIARY_ADDITION` (Usually New)
3. `TRANSACTION` (Usually Completed)

**The Bottleneck:**
Because the device and session are always legitimate, they contribute `0` variation to the APP fingerprint. For an `EASY` attack, the profile restricts splitting (`splits=1`), restricts timing to `RAPID`, and forces `beneficiary_new_prob=1.0`. The remaining combinatorial space (essentially just amount scaling) creates fewer than 15 unique structural fingerprints. This mathematically prevents the Novelty Engine from generating 100 uniquely novel traces without exhausting attempts.

## 3. Real APP Variation Dimensions Identified
We can vastly expand APP structural entropy by explicitly modeling human social-engineering behavior:

### A. Payment Urgency / Hesitation (Temporal Social Engineering)
- **Concept:** Victims under duress often hesitate on the confirmation screen or take time to comply with the scammer's verbal instructions.
- **Observable via:** The precise timestamp delta between `BENEFICIARY_ADDITION` and `TRANSACTION`.
- **Classification:** OBSERVABLE NOW (No Schema Extension Required).

### B. Transaction Outcome Patterns (Decline & Retry)
- **Concept:** High-value APP transactions often trigger bank risk engines (declines). The victim, prompted by the scammer, retries the transfer with a smaller amount until it passes.
- **Observable via:** Generating a `failed` transaction event followed by a `completed` transaction event within the same session.
- **Classification:** OBSERVABLE NOW (No Schema Extension Required).

### C. Amount Escalation Trends
- **Concept:** When splitting transactions, scammers may test the waters with a small transfer before executing a massive transfer (Escalation), or drain limits sequentially (Fragmentation).
- **Observable via:** The numeric sequence of transaction amounts.
- **Classification:** OBSERVABLE NOW (No Schema Extension Required).

## 4. APP vs ATO Boundary Safety
It is critical that these dimensions do not blur the line between APP and ATO:

| Dimension | APP Meaning | ATO Meaning | Safe? |
| :--- | :--- | :--- | :--- |
| **Hesitation Gap** | Victim coercion/reluctance | Script timeout / C2 lag | **YES.** Human hesitation is fundamentally different from script timing. |
| **Decline & Retry** | Scammer negotiating with victim limits | Script bruteforcing balance limits | **YES.** Contextualized by the legitimate device, this strongly signals APP. |
| **Amount Escalation** | Test payment followed by drain | Automated limit extraction | **YES.** Combined with the above, it forms a distinct behavioral signature. |

## 5. Fingerprint Extension Design
To recognize these new dimensions, the `APPAttackFingerprint` will be updated to include:
1. `hesitation_category`: Derived from the time gap between beneficiary setup and the first payment (`"immediate"`, `"hesitant"`, `"delayed"`).
2. `outcome_pattern`: Derived from transaction success states (`"clean"`, `"failed_then_success"`, `"multiple_failures"`).
3. `amount_trend`: Derived from split transactions (`"single"`, `"escalating"`, `"fragmented"`, `"decreasing"`).

These purely structural metrics ignore numerical noise (UUIDs, $1 offsets) and map perfectly to genuine social-engineering variance.

## 6. Entropy Analysis
By introducing these three dimensions, the multiplicative entropy of `EASY` APP expands from ~12 fingerprints to:
- Hesitation (3 states)
- Outcome Pattern (3 states)
- Amount Buckets (4 states)
- Beneficiary (1 state)
Total = 3 * 3 * 4 * 1 = **36 unique fingerprints**. 

For `MEDIUM` (which allows 1-2 splits):
- Amount Trend (4 states)
- Outcome Pattern (3 states)
- Hesitation (3 states)
- Amount Buckets (4 states)
- Beneficiary (2 states)
Total = 4 * 3 * 3 * 4 * 2 = **288 unique fingerprints**.

This resolves the novelty saturation bottleneck.

## 7. Ground-Truth Isolation
No new internal properties will leak into the `ObservableAttackTrace`. `hesitation_category` is derived mathematically from the `timestamp` values already present in the trace. The `AttackPlan` and `WorldState` remain completely opaque to the detector.

## 8. Test Plan
- **Deterministic Generation:** Ensure the new probabilities respect `seed`.
- **APP/ATO Separation:** Verify that ATO fingerprints do not attempt to extract `hesitation_category`.
- **Novelty Sensitivity:** Assert that identical hesitation/retry patterns hash to the same value, while switching from "immediate" to "hesitant" generates a novel hash.
- **Realism Preservation:** Ensure `RealismValidator` correctly processes the new `failed` transaction events without failing `BALANCE_CONSTRAINT_VIOLATION`. 
- **Full Regression:** 100% test pass rate required.

## 9. Verdict
**NO_SCHEMA_CHANGE_REQUIRED**

The existing schemas are fully robust and capable of representing the complex temporal and outcome-based behaviors of human social engineering. The modifications will be localized strictly to `DIFFICULTY_PROFILES`, `StatefulSimulator`, and `APPAttackFingerprint`.
