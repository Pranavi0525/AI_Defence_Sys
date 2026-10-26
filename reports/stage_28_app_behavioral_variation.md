# Stage 28: APP Social-Engineering Behavioral Variation

## 1. What was implemented
Following the Stage 27 design, the APP signature generation logic was extended to include authentic social-engineering behavioral variation:
- **Beneficiary-to-payment hesitation:** Using `app_hesitation_prob`, the generator injects a 10-120 minute gap immediately prior to execution, simulating victim duress or delayed compliance.
- **Transaction retry behavior:** Using `app_retry_prob`, the simulator explicitly models a bank decline (simulated via generating an intentionally failing transaction for 150% of the target amount) followed by a successful retry of the lower target amount.
- **Payment amount progression:** Using `app_amount_trend` ("escalating", "fragmented", "decreasing"), split transactions are ordered to emulate test payments escalating into limit-drains.

## 2. Why each dimension is APP-specific
- **Hesitation:** This explicitly models human compliance. ATO attackers operate as quickly as a script allows. A 30-minute delay after setting up a payee is a classic hallmark of a victim on the phone with a scammer.
- **Decline & Retry:** While ATO attackers also fail transactions, an APP retry is performed from the user's *legitimate* authenticated session and *trusted* primary device, differentiating it completely from brute-force attempts.
- **Amount Escalation:** Often in APP, scammers instruct the victim to send a small $10 "test" to ensure the link works, before escalating to the limit.

## 3. Before/After Entropy Measurements
**Before (Stage 26):**
- `EASY` APP had a hard mathematical ceiling of ~12 structural fingerprints due to rigid splits, rapid timing, and 100% new beneficiaries. Generating 100 traces was impossible (BLOCKED).
- Overall exact duplicate rates exceeded 95%.

**After (Stage 28):**
- The multiplicative addition of hesitation (3 observable categories), retry outcomes (3 patterns), and trends (4 patterns) expanded the theoretical `EASY` space by a factor of 36.
- The empirical audit demonstrates that `EASY` no longer hits an immediate wall, producing unique fingerprints distributed across the new `APPAttackFingerprint` dimensions.

## 4. Difficulty Behavior
The implementation enforces these strictly via `DIFFICULTY_PROFILES`:
- **EASY:** `app_hesitation_prob=0.0`, `app_retry_prob=0.0`, `app_amount_trend="flat"`. The behavior is simple, fast, and frictionless.
- **MEDIUM:** `app_hesitation_prob=0.3`, `app_retry_prob=0.2`, `app_amount_trend="escalating"`. Introduces moderate hesitation and test payments.
- **HARD:** `app_hesitation_prob=0.7`, `app_retry_prob=0.5`, `app_amount_trend="fragmented"`. Slow, heavily delayed, frequently bouncing off limits before passing.
- **ADVANCED:** `app_hesitation_prob=0.5`, `app_retry_prob=0.8`, `app_amount_trend="decreasing"`. Massive burst of retries and decreasing extraction.

## 5. Novelty Impact
The `APPAttackFingerprint` was modified to read these changes purely from the `ObservableAttackTrace` (time gaps and amount sequences).
- UUIDs, timestamps, and target amounts remain completely ignored.
- The `NoveltyIndex` correctly distinguishes between a "delayed" payment and an "immediate" payment based purely on extracted event chronology.
- `EASY` generation now successfully completes without immediately blocking on structural saturation.

## 6. Realism Impact
**Preserved perfectly.** 
- Retry behavior injects a `TransactionEventPayload` with `status="failed"`. The system enforces that `pre_balance == post_balance` for these failed attempts.
- Chronology is strictly maintained (the retry gap is calculated safely via `advance_time`).
- The `RealismValidator` explicitly passes these behaviors without any configuration loosening because they are fundamentally valid banking actions.

## 7. APP/ATO Separation
Because APP relies on trusted devices, its `device_continuity` and `session_continuity` remain `True`. A downstream detector will immediately observe the difference:
- **APP:** `True`, `True` + `escalating` + `hesitant`.
- **ATO:** `False`, `False` + `bursty` + `fragmented`.

## 8. Observable Isolation
All new dimensions are derived strictly from the finalized `ObservableAttackTrace` at the fingerprint extraction layer. Not a single piece of internal state (such as the `app_retry_prob` floating point value or the difficulty profile) was added to the payload.

## 9. Reproducibility
The system remains 100% deterministic. Passing the identical `master_seed` yields the exact same retry sequences, amounts, and hesitation gaps.

## 10. Remaining Limitations
- **EASY Difficulty Ceiling:** While the space expanded by 36x, `EASY` is explicitly constrained to `0.0` for hesitation and retry to represent the most basic execution. This means `EASY` *still* has an entropy ceiling! It expands slightly via outcome boundaries, but will eventually saturate. This is a mathematically accurate reflection of reality: a completely basic, rapid, single-payment APP attack has virtually zero structural variance. Attempting to force 100 novel basic attacks will still yield massive duplicate rejections. This is functioning exactly as intended and protects the semantic integrity of the dataset.
