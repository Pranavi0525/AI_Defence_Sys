# Stage 15 — ATO Realism Validator Audit & Rejection Analysis

## 1. Executive Summary
The Stage 15 audit successfully refactored the Realism Validator from a binary black box into a fully observable, multi-failure reporting system with structured reason codes. An experiment over the Stage 12 ATO corpus revealed that 100% of trace rejections are caused by a single strict rule: `BALANCE_CONSTRAINT_VIOLATION`. This reveals an over-restrictive constraint in our simulation where synthetic attacks are penalized for draining accounts that do not have active synthetic income streams replenishing them.

## 2. Verdict
**ACCEPT_WITH_LIMITATIONS**
The realism validator now perfectly supports structured, auditable rejection reporting. However, the simulation environment's strict balance constraint causes false positives for valid attack behavior, limiting the generator's ability to produce complex, multi-transfer attack paths without failing validation.

## 3. Audit Context & Goal
Previously, the Realism Validator returned generic boolean failures (e.g. `Failures: ["Constraint: Impossible balance transition (e.g. overspend)."]`). This lack of observability prevented researchers from understanding why the generator was rejecting ~24% of generated traces. The goal of Stage 15 was to implement a rigorous, machine-readable rejection taxonomy and use it to audit the Stage 12 ATO corpus.

## 4. Original Validator Architecture
The original architecture accumulated string-based reasons for failures and aggressively aborted validation when hard gates were hit. The `RealismReport` returned a single list of strings for `failures`, masking the underlying structural issues and failing to distinguish between primary violations and cascading secondary faults.

## 5. Refactoring Accomplished
We entirely rewrote `src/red_team/validation/realism.py` to support the new `RejectionReason` schema. `RealismCheckResult` was upgraded to include specific reason codes, categories, and severities. The `RealismReport` was extended to export a `primary_failure` and a list of `secondary_failures`.

## 6. Structured Reason Code Taxonomy
The following taxonomy was implemented:
- `DUPLICATE_EVENT_ID`: Trace contains duplicate event identifiers.
- `GROUND_TRUTH_MISMATCH`: Observable events diverge from ground truth.
- `TEMPORAL_ORDER_VIOLATION`: Events are not strictly chronologically ordered.
- `PHANTOM_ENTITY_REFERENCE`: An event references a non-existent account or entity.
- `BALANCE_CONSTRAINT_VIOLATION`: A transaction attempts to overspend the account's available balance.
- `INVALID_RELATIONSHIP`: A transfer references a beneficiary before it was added.
- `INVALID_EVENT_SEQUENCE`: A session references a device before it was registered.
- `INVALID_PHASE_TRANSITION`: An attack action violates the state transitions allowed by the ATO signature.

## 7. Multi-Failure Processing Mechanism
The validator now executes all non-fatal checks sequentially regardless of previous failures. The first `HARD` failure encountered is elected as the `primary_failure`, ensuring deterministic ordering, while all subsequent failures are collected into `secondary_failures`.

## 8. Rejection Statistics & Acceptance Rate
Rerunning the Stage 12 configuration (`target_count=100`, `master_seed=42`, `max_attempts=1000`) yielded identical macroscopic results:
- **Attempts:** 131
- **Accepted:** 100
- **Rejected:** 31
- **Acceptance Rate:** 76.34%

## 9. Primary Rejection Concentration
100% of the primary rejections (31 out of 31) were classified as `BALANCE_CONSTRAINT_VIOLATION`.

## 10. Phase Path Analysis of Rejections
Because the rejections are exclusively balance-related, they occur strictly during the `EXPLOITATION` phase (specifically during `TRANSFER` events). Attacks correctly navigate `RECONNAISSANCE` and `ACCOUNT_ACCESS`, but fail when attempting to execute multiple fund transfers that exceed the static initial balance provided to the persona.

## 11. Secondary Failure Insights
The experiment recorded 0 secondary failures. The trace generation cleanly halts or aborts without violating temporal order or graph integrity when a balance violation occurs, meaning the simulator is structurally sound but resource-starved.

## 12. Difficulty vs Rejection Correlation
High-difficulty attacks require more exploitation phases and thus more `TRANSFER` events. Given the static balance constraint, there is a direct correlation between attack difficulty and rejection probability.

## 13. False Positive Assessment
The `BALANCE_CONSTRAINT_VIOLATION` gate is currently over-restrictive. In the real world, an attacker attempting to transfer more money than available would simply result in a declined transaction, which itself is a valuable adversarial signal. Dropping the entire trace as "unrealistic" represents a false positive rejection, masking legitimate fraud attempt indicators from the Blue Team.

## 14. Threshold Audit & Provenance Verification
All existing thresholds in `src/red_team/validation/realism.py` (e.g. `temporal_spacing`, `behavioral_deviation`) were audited. They are currently uncalibrated placeholders. They have been explicitly tagged with `DOMAIN_MODELED` comments to enforce that they are not empirically backed.

## 15. Observability Schema Modifications
`RealismReport` now cleanly exposes:
- `primary_failure: Optional[RejectionReason]`
- `secondary_failures: List[RejectionReason]`
- `rejection_reasons: List[RejectionReason]`
- `checks_run`, `checks_passed`, `checks_failed`
- Fully populated `available_metrics` and `unavailable_metrics` arrays.

## 16. Regression Tests Authored
Authored `tests/test_realism_rejections.py`, which validates:
- Multi-failure reporting capabilities via deterministic ordering.
- `TEMPORAL_ORDER_VIOLATION` alongside `PHANTOM_ENTITY_REFERENCE`.
- Schema compliance with the Pydantic observables (`ObservableTransactionEvent`).

## 17. Boundary Condition Compliance
No modifications were made to the ATO signature, the behavioral simulator's underlying mechanics, or the ML infrastructure. The scope was strictly limited to validator observability and reporting.

## 18. Known Limitations
The validator currently lacks empirical statistical data to validate behavioral deviations. All behavioral checks return dummy `INFO` passes or `NOT_AVAILABLE`.

## 19. Recommended Next Stage
**Stage 16: Balance Constraint Relaxation & Failed Transaction Modeling**
The generator should be updated to model `DECLINED` transactions rather than outright rejecting attack traces when a balance constraint is hit. This will fix the 24% synthetic rejection rate and provide the Blue Team with crucial "NSF (Non-Sufficient Funds)" fraud signals.
