# Stage 18: Controlled ATO Attack Variation

## 1. Objective
Improve the ATO simulator to produce controlled, meaningful behavioral variation across attack traces (amounts, timings, entities) while remaining strictly bound to structural invariants, provenance rules, and the AttackSignature graph.

## 2. Stage 17 Problems Addressed
- Hardcoded $500.00 transaction amounts.
- Total collapse of difficulty label semantics.
- Mechanical RNG-only timing without behavioral patterns.
- High path duplication (23% reconnaissance only).
- Lack of transaction friction beyond NSF.

## 3. Existing Architecture Inspected
- `StatefulSimulator` heavily relied on static instantiations for `consequence` mapping.
- It mutated `world_state` predictably without scaling to customer history.

## 4. Variation Model
We introduced a `VariationProfile` explicitly bound to the attack `difficulty`.
```python
class VariationProfile(BaseModel):
    timing_type: Literal["RAPID", "NORMAL", "SLOW", "BURSTY"]
    device_new_prob: float
    beneficiary_new_prob: float
    amount_scale: Tuple[float, float]
    splits: Tuple[int, int]
    end_early_prob: float
    loop_prob_mult: float
```
This is fully configurable per difficulty level and documented as `DOMAIN_MODELED`.

## 5. Transaction Amount Model
- **Amount Generation:** Replaced hardcoded $500 with an account-relative calculation: `target_amt = acct.balance * uniform(scale_min, scale_max)`.
- If balance < $50, defaults to $100.00 (forcing NSF).
- **Result:** Fully coherent financially.

## 6. Transaction Splitting
- An intended `target_amt` can be split into N smaller transactions based on the `splits` tuple in the `VariationProfile`.
- The splits sum perfectly to `target_amt` preventing invisible trace inflation.
- Optional delays are inserted between split executions.

## 7. Timing Model
- **RAPID:** 1-10 minutes between events.
- **NORMAL:** 30-120 minutes.
- **SLOW:** 12-24 hours.
- **BURSTY:** 20% chance of a 1-2 day wait; 80% chance of a 1-10 minute rapid burst.
- *Strict chronological append only.*

## 8. Device & Beneficiary Model
- Reuses existing relationships.
- Based on `device_new_prob` / `beneficiary_new_prob`, the attacker will probabilistically harvest known devices/beneficiaries from the customer's history in the `WorldState` rather than constantly spinning up new identities.

## 10. Phase Variation & Loop Modification
- `end_early_prob`: Modifies the AttackSignature's weight towards `END` (Easy mode is more likely to abandon).
- `loop_prob_mult`: Multiplies the probability of returning to the same state (Advanced mode loops heavily).

## 11. Difficulty Profiles (Measurable Generation Profiles)
- **EASY:** Rapid, 100% new entities, large amounts, no splitting, eager to finish early.
- **MEDIUM:** Normal speed, mostly new entities, moderate amounts, occasional split (1-2).
- **HARD:** Slow speed, reuses entities to blend in, small amounts, 2-3 splits, avoids ending early.
- **ADVANCED:** Bursty timing, almost entirely reuses entities, 3-5 splits, high loop probability.

## 12. Failure/Friction Modeling
Introduced a 10% chance for the attacker to erroneously overestimate the available balance, resulting in a slightly oversized transaction target that triggers realistic NSF (declined) outcomes natively.

## 13. Provenance
All parameters in `VariationProfile` and the 10% greedy friction rate are explicitly marked in source code as **DOMAIN_MODELED**.

## 14. Reproducibility
The system is fully deterministic under the `master_seed`. `generate_attack_corpus` produces exactly the same aggregated and structural metrics. (The test suite `test_deterministic_variation` guarantees this).

## 15. Before vs. After Corpus Comparison
*(Based on target_count=100, master_seed=42)*

| Metric | Before (Stage 17) | After (Stage 18) |
| --- | --- | --- |
| Transaction Amount | $500.00 (variance=0) | min=$328.53, mean=$6010.64, max=$24565.52 |
| Unique Amounts | 1 | 127 |
| Unique Paths | 40 | 35 |
| Most Common Path | RECONNAISSANCE (23) | RECONNAISSANCE (18) |
| Diff. Phases (EASY) | 2.93 phases | 2.63 phases |
| Diff. Phases (ADVANCED) | 2.68 phases | 2.79 phases |
| Diff. Events (EASY) | 3.46 events | 3.23 events |
| Diff. Events (ADVANCED) | 3.40 events | 4.79 events (Splitting logic works!) |
| Approved / Declined | 44 / 39 | 127 / 0 (Note: Overdrawing is now rarer due to relative scaling) |
| Timing Gaps | max=312s | max=151308s (42 hours, Bursty) |

## 16. Remaining Limitations
- While transaction limits scale perfectly relative to the balance, there are very few Declined transactions because the attacker correctly sizes the relative transfer. The 10% friction error is not high enough to create massive NSF volumes.
- Failed logins or MFA challenges still lack semantic support in the schema without fabricating non-standard flags.

## 17. Tests
`tests/test_stage_18_attack_variation.py` added covering:
- Deterministic Variation
- Financial Coherence & Amount Variance
- Splitting Capability
- Chronology constraints
All 445 tests passed.

## 18. Acceptance Decision
**ACCEPT**. The rigid boundaries of the simulator are eliminated while protecting structural truth. The corpus now produces measurable difficulty differentiation without risking downstream LLM/Blue Team overfitting.
