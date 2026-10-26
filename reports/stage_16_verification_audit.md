# Stage 16 Verification Audit

## 1. Scope
This audit independently verifies the changes and claims made in Stage 16 ("Transaction Outcome & Balance-State Modeling"). We will verify whether transaction semantics (APPROVED vs. DECLINED) behave safely, whether balance invariants were protected, and whether the 2 remaining Stage 16 rejections were truly test-harness false positives. We will also perform negative testing and corpus integrity checks.

## 2. Stage 16 Claims Being Audited
- 29 of the original 31 Stage 15 `BALANCE_CONSTRAINT_VIOLATION`s were converted to structurally valid failed/declined transactions.
- The remaining 2 rejections are false positives due to `generate_attack_corpus` passing a post-attack mutated WorldState to the validator.
- APPROVED transactions mutate balance precisely once.
- DECLINED transactions do not mutate balance.
- Rejection rates dropped safely without weakening validation rules.

## 3. Actual Implementation Inspected
I thoroughly reviewed the modified files:
- `src/red_team/attacks/simulator.py`: Logic previously injected artificial `acct.balance += amt * 2` top-ups when `acct.balance < amt`. The Stage 16 patch removed this and correctly assigned `status = "failed"` while leaving `acct.balance` untouched.
- `src/red_team/validation/realism.py`: The validator was updated to conditionally branch on `transaction_status`. If `completed`, it rigorously enforces `amt <= tracked_balance` and deducts the amount. If `failed`, it skips the balance deduction and allows the trace to continue. 
- `scripts/run_stage_16_experiment.py`, `scripts/investigate.py`, `tests/test_stage_16_transaction_outcome.py`: Appropriately measure and test these properties.

## 4. Transaction Outcome Verification
By verifying the direct Python object mutations in `StatefulSimulator._synthesize_events_for_consequence`:
- **COMPLETED**: `acct.balance -= amt` executes, and `post_balance` captures the modified balance. `pre_balance - amt == post_balance`.
- **FAILED**: `acct.balance` is explicitly untouched. `post_balance` captures the unmutated balance. `pre_balance == post_balance`.
There are no code paths in the simulator where a failed transaction mutates the internal WorldState.

## 5. Balance Mutation Verification
- The `WorldState` is correctly mutated precisely once per `completed` transaction via standard in-place assignment (`-=`).
- `TransactionEventPayload` strictly carries the `pre_balance` and `post_balance` that mirror the exact physical WorldState before and after execution.

## 6. Detailed Analysis of the Two Remaining Rejections
I authored `investigate.py` to capture and inspect the exact traces of the 2 rejected attempts.
Both traces exhibit the exact same pattern:
- The synthetic customer successfully executed 4 identical `status="completed"` transfers of `$500.00` each.
- The customer began with sufficient funds (e.g., `$2288.00`), ending up with a valid final `WorldState` balance (e.g., `$288.00`).
- The validator initialized its internal `balances[acct]` tracker using the **final** WorldState balance (`$288.00`). 
- When evaluating the first `completed` $500 transfer chronologically, it failed the check `500 > 288`, triggering a `BALANCE_CONSTRAINT_VIOLATION`.

**Conclusion:** Both rejections are definitively **C. corpus harness bug (test-harness false positives).** The traces themselves are genuinely valid.

## 7. Pre/Post Attack WorldState Analysis
Tracing the object lifecycle from `src/red_team/attacks/corpus.py`:
1. `world_state` is instantiated by `NormalWorld`.
2. Passed by reference to `sim = StatefulSimulator(world_state, ...)`
3. `sim.generate_attack(...)` intrinsically mutates `world_state` in-place.
4. The exact mutated `world_state` reference is then passed to `validate_attack_realism(trace, gt, signature, world_state)`.
Because Python passes object references, the validator receives the POST-attack state. The smallest safe fix is to inject a `deepcopy` when initializing the simulator, or caching a `deepcopy` to pass to the validator.

## 8. Validator Correctness
The validator correctly became more permissive *only* to structurally sound DECLINED transactions. It strictly retains its mathematical boundaries for COMPLETED transactions. The increase in acceptance rate (98.04%) genuinely reflects the elimination of the simulator's unrealistic "auto-top-up" bug that previously broke temporal state checks.

## 9. Negative Tests
I authored negative unit tests in `tests/test_stage_16_verification_audit.py` proving the validator and simulator schemas continue to strictly reject:
1. Completed transactions overdrawing the account.
2. Completed transactions with mathematically inconsistent `post_balance`s.
3. Failed transactions mutating balance.
4. Phantom account references.
5. Chronologically impossible event ordering.

## 10. Reproducibility
I executed `scripts/investigate_reproducibility.py` which runs the full 100-trace corpus generation twice. 
- Accepted counts, rejected counts, and transaction outcomes were perfectly identical (100 vs 100).
- The internal properties of the attack execution are 100% reproducible based on `master_seed=42`. 
- *(Note: `customer_id` UUIDs differ across runs due to `uuid4()` entity initialization in NormalWorld, but control flow remains tightly deterministic.)*

## 11. Corpus API Impact
Stage 16 added `validation_metadata` and the observable `trace` to the `rejected_attempts` dictionary returned by `generate_attack_corpus`.
- It does **not** leak Ground Truth.
- However, storing complete trace and metadata objects for every rejected attempt significantly changes memory scalability. If a signature generates 100,000 rejections, the process will OOM.
- **Recommendation:** This is an investigation-only convenience that should either be formalized into a strict `RejectedAttemptRecord` schema (and pruned), or removed entirely to preserve memory footprint before scaling generation.

## 12. Defects Discovered
1. **Corpus Harness Initialization Bug:** Evaluates validity against POST-attack state.
2. **Memory Scalability Risk:** Saving complete rejected traces unconditionally.

## 13. Fixes Made, if any
None during this audit. The system remains precisely in its Stage 16 state to preserve the integrity of the audit.

## 14. Tests
Executed the entire suite (`pytest tests/ -v`).
Total Tests: 435
Passed: 435
Failed: 0
Warnings: 0

## 15. Final Verdict
**ACCEPT**

The logic introduced in Stage 16 correctly represents transaction outcomes and enforces strict boundaries. The remaining rejections are conclusively isolated to an infrastructural test harness bug rather than adversarial defect. 
