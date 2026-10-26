# Ground-Truth Integrity Root-Cause Investigation

## 1. Original Forensic Finding
The forensic reconciliation identified a BLOCKING FAILURE in Ground-Truth Integrity. Specifically, an Account Takeover (ATO) attack and an Authorized Push Payment (APP) attack, despite being fundamentally distinct simulation paths, generated the exact same `attack_id` (`atk-bdd640fb`) and the exact same initial phase entry/exit timestamps when provided the same global seed.

## 2. Exact Evidence
*   **Duplicate attack_id:** Both ATO(seed=42) and APP(seed=42) produced `attack_id = 'atk-bdd640fb'`.
*   **Duplicate timestamp:** Both ATO(seed=42) and APP(seed=42) produced phase 1 timestamps of `entered_at=2025-01-01 23:51:16`, `exited_at=2025-01-02 16:32:16`.
*   **Source of Executions:** These were produced by identical, independent calls to the production `StatefulSimulator.generate_attack()` API using `master_seed=42`.
*   **Comparison Method:** The comparison was performed programmatically on the actual returned `AttackGroundTruth` objects, not hardcoded strings or logs.

## 3. attack_id Implementation Path
```python
# src/red_team/attacks/simulator.py
self.seed = seed
self.rng = random.Random(seed)

def _generate_event_id(self) -> str:
    return str(uuid.UUID(int=self.rng.getrandbits(128)))

# In generate_attack():
attack_id = f"atk-{self._generate_event_id()[:8]}"
```
*Finding:* The `attack_id` is **B. Seed-derived**. It exclusively consumes the first 128 bits from the deterministic `random.Random(seed)` state without context or salting.

## 4. Phase Timestamp Implementation Path
```python
# src/red_team/attacks/simulator.py
gap = self.rng.randint(720, 1440) * 60
self.state.advance_time(gap)
```
*Finding:* The phase timestamps (`entered_at`, `exited_at`) are derived directly from the **random seed** interacting with the shared `WorldState` clock. Because the seed is unsalted by attack type, the identical sequence of initial random float/integer queries generates identical time progressions.

## 5. Four Fresh Runtime Executions
*   **A: ACCOUNT_TAKEOVER (seed=42)**
    *   `attack_id`: `atk-bdd640fb`
    *   Phase 1: `2025-01-01 23:51:16` -> `2025-01-02 16:32:16`
*   **B: ACCOUNT_TAKEOVER (seed=43)**
    *   `attack_id`: `atk-c33f4584`
    *   Phase 1: `2025-01-02 02:51:21` -> `2025-01-02 17:18:21`
*   **C: AUTHORIZED_PUSH_PAYMENT (seed=44)**
    *   `attack_id`: `atk-fde0f5cd`
    *   Phase 1: `2025-01-01 11:27:16` -> `2025-01-02 07:11:16`
*   **D: AUTHORIZED_PUSH_PAYMENT (seed=45)**
    *   `attack_id`: `atk-d876a3e2`
    *   Phase 1: `2025-01-02 14:14:21` -> `2025-01-03 00:58:21`

*(Full raw JSONs for observable and ground truth omitted for brevity but strictly generated via `model_dump()` in the forensic script).*

## 6. Cross-Run Comparison
```
A.attack_id != B.attack_id : PASS (True)
A.attack_id != C.attack_id : PASS (True)
A.attack_id != D.attack_id : PASS (True)
B.attack_id != C.attack_id : PASS (True)
B.attack_id != D.attack_id : PASS (True)
C.attack_id != D.attack_id : PASS (True)
```
When uniquely seeded, all IDs and timestamps strictly diverge.

## 7. Observable Isolation Check
Observable records physically share linkage with the ground truth via `event_id` mapping to `linked_event_ids`, ensuring they belong to the exact same simulator execution.
*   `attack_family`: NOT LEAKED
*   `difficulty`: NOT LEAKED
*   `attack_id`: PRESENT AS `trace_id` (by design, primary key integration)
*   `planner_metadata`: NOT LEAKED
*   `variation_profile`: NOT LEAKED
*   `internal simulation state`: NOT LEAKED
*   *Conclusion: Isolation mechanism perfectly restricts metadata.*

## 8. Exact Root Cause
**CONCLUSION B & C: Genuine production bugs in `attack_id` and timestamp generation.**
The root cause is a systemic failure to salt the deterministic PRNG sequence in the production code. Both `corpus.py` (which yields identical `child_seed` sequences for different families given the same `master_seed`) and `StatefulSimulator` (which initializes `random.Random(seed)`) fail to inject context like the `attack_family` string into the seed hash. Consequently, when two different attack types are queued under the same seed conditions, their simulators initialize in identical states, pull the exact same 128 random bits for their `attack_id`, and pull identical initial random integers for phase timing gaps.

## 9. Blocking Determination
**Ground Truth Integrity = CONFIRMED PRODUCTION BUG (Status = BLOCKING)**
This is not a false positive or an artifact of the forensic harness. The production APIs intrinsically generate massive structural collisions across completely separated attack vectors when using standard reproducible seeds.

## 10. Recommended Next Action
Apply a context salt to the simulator PRNG initialization. For example, modify `StatefulSimulator.__init__` to hash the provided seed together with `signature.attack_family` before passing it to `random.Random()`. Do NOT apply fixes during this audit phase.
