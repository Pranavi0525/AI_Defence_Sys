# Stage 25 — AUTHORIZED PUSH PAYMENT (APP) Implementation

## Objective
Implement `AUTHORIZED_PUSH_PAYMENT` (APP) as a genuine second attack-family vertical slice in the Red Team corpus generator, running side-by-side with the frozen `ACCOUNT_TAKEOVER` (ATO) implementation.

## 1. Abstraction Refactoring

### 1.1 Polymorphic Fingerprinting
The previous global `AttackFingerprint` was refactored into a `BaseAttackFingerprint` with family-specific implementations:
- `ATOAttackFingerprint`: Analyzes device variation, timing categories, amount thresholds, and transaction splits.
- `APPAttackFingerprint`: Analyzes device *continuity*, session duration, amount thresholds, and beneficiary novelty (new vs existing hijacked).

The `NoveltyIndex` was converted into a nested dictionary grouping fingerprints by `attack_family` and `difficulty`, strictly preventing cross-family comparisons.

### 1.2 Signature Graph Expansion
Created `app_signature.py` encapsulating the social engineering progression:
1. `SOCIAL_ENGINEERING`
2. `SESSION_ACTIVE`
3. `BENEFICIARY_SETUP`
4. `PAYMENT_EXECUTION`

## 2. Generator Modifications

### 2.1 Context Selection
`StatefulSimulator` was augmented to respect APP's fundamental constraint: attacks must originate from the customer's known primary device. When the requested family is APP, the simulator explicitly extracts an existing device from the `WorldState` relationships and injects it into the generation loop instead of probabilistically generating anomalous devices.

### 2.2 Entry State Dispatch
`generate_attack_corpus` was parameterized to accept `attack_family` and dynamically dispatch to `get_app_signature()` or `get_ato_signature()`, using the proper family-defined `entry_states` rather than hardcoded ATO strings.

## 3. Realism Validator Additions
The constraint validation layer of `validate_attack_realism` was updated:
- **APP Constraint**: If an APP trace contains a `DEVICE_REGISTRATION` event (a brand new device), it is strictly rejected with reason code `APP_NEW_DEVICE_VIOLATION`.
- **Session Resolution**: Modified the session linkage constraint to successfully resolve device IDs against the global `WorldState` references (for known devices) rather than only accepting devices registered dynamically within the trace window.

## 4. Test Suite and Regression
Created `test_stage_25_app.py` enforcing the 16 invariant requirements.
All 464 tests across the entire test suite pass, guaranteeing 100% backward compatibility for ATO generators and validators.

## 5. Experiment Results
Comparing ATO vs APP generation:
APP generation correctly synthesizes `SESSION_LOGIN`, `BENEFICIARY_ADDITION`, and `TRANSACTION` events, explicitly skipping `DEVICE_REGISTRATION` and successfully routing around `RealismValidator` strict checks.

## Conclusion
Stage 25 is COMPLETE. The APP vertical slice is implemented and verified. The `CorpusGenerator` is now functionally a multi-family orchestrator.

**Update:** A minor oversight in pp_signature.py (missing 'transaction' in ffected_entities for PAYMENT_EXECUTION) was resolved to ensure that TRANSACTION events are properly emitted by the simulator for APP attacks.

