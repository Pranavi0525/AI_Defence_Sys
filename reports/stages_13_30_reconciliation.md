# FINAL FORENSIC RECONCILIATION: GROUND TRUTH & CORPUS SEEDING

## 1. Root-Cause Report
*(See `reports/ground_truth_integrity_root_cause.md` for full detailed root-cause execution logs.)*
**Summary**: The `StatefulSimulator` and `corpus.py` generator fail to salt the random seed with the `attack_family`. Consequently, when multiple distinct attack types are executed under the same global master seed, they sequentially initialize identical PRNG states and pull the exact same pseudo-random values for their UUIDs and initial phase timing gaps, resulting in mathematically guaranteed collisions.

## 2. Actual Corpus Seed Derivation Code
In `src/red_team/attacks/corpus.py`:
```python
def generate_attack_corpus(..., master_seed: int = 42, ...):
    rng = random.Random(master_seed)
    ...
    # Inside the attempt loop:
    child_seed = rng.getrandbits(32)
    ...
    sim = StatefulSimulator(state_copy, signature, seed=child_seed)
```
**A. Does every corpus attempt receive a different seed?**
YES (within the same family/run). The `getrandbits(32)` sequentially updates the PRNG.
**B. Is the seed derived from master_seed + attempt index, difficulty, or family?**
NO. It is derived solely from the sequential state of the raw `master_seed`.
**C. Can two attempts belonging to different attack families receive the exact same raw simulator seed during normal mixed-family generation?**
YES. If `generate_attack_corpus` is called once for ATO (with `master_seed=42`) and once for APP (with `master_seed=42`), Attempt #1 for ATO gets the identical `child_seed` as Attempt #1 for APP.
**D. Can two attempts within the same family receive the same raw simulator seed?**
NO. The PRNG advances within a single generator execution.
**E. Can two independent single-family corpus-generation runs using the same master_seed reproduce the same attack IDs?**
YES. The generator is fully deterministic based on the provided master seed.

## 3. Existing Corpus Artifact Inventory
Searches via `Get-ChildItem -Recurse -Include *.json,*.pkl` reveal that the actual multi-trace serialized objects (`ObservableAttackTrace` arrays) are **NOT PERSISTED**. Only summary statistics (e.g., `reports/final_qualification_data.json`, `reports/stage_25_experiment.json`) and one single sample (`reports/handoff_sample.json`) physically exist.

## 4. Duplicate Attack-ID Audit
*   **Existing ATO corpus:** NOT AUDITABLE (Raw traces not persisted).
*   **Existing APP corpus:** NOT AUDITABLE.
*   **Mixed corpus:** NOT AUDITABLE.

## 5. Phase-Timestamp Duplicate Audit
*   **ATO vs ATO:** NOT AUDITABLE.
*   **APP vs APP:** NOT AUDITABLE.
*   **ATO vs APP:** NOT AUDITABLE.

## 6. Ground-Truth Integrity Conclusion
**CASE B:** `REAL BUG / EXISTING CORPORA POTENTIALLY CONTAMINATED`
The normal corpus generator structurally reuses seeds between families due to an unsalted deterministic implementation in `corpus.py`. If the corpora were actually persisted from `run_final_red_team_qualification.py` (which explicitly uses `master_seed=42` for both ATO and APP runs), they would absolutely contain exact `attack_id` duplicates and identical initial phase timings across the two families.

## 7. Proposed Fix (NOT IMPLEMENTED)
The most robust and minimal fix occurs entirely within `StatefulSimulator.__init__` in `src/red_team/attacks/simulator.py`.
```python
import hashlib
class StatefulSimulator:
    def __init__(self, state: WorldState, signature: AttackSignature, seed: int):
        self.state = state
        self.signature = signature
        
        # PROPOSED FIX: Salt the seed deterministically using the attack family
        salt_str = f"{seed}_{signature.attack_family}"
        salted_seed = int(hashlib.sha256(salt_str.encode()).hexdigest()[:15], 16)
        
        self.seed = salted_seed
        self.rng = random.Random(self.seed)
```
This requires NO changes to `corpus.py` or the `AttackPlan`. It guarantees that even if `corpus.py` inadvertently passes the exact same `child_seed` to two different families, the simulator internally forks the PRNG state, guaranteeing mathematically distinct `attack_id`s and timings.

## 8. Stage 13–30 Inventory
*   **Stage 13 (Normal World Calibration):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 14 (Normal World Behavior Redesign):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 15 (Realism Validator Audit):** TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 16 (Transaction Outcome Modeling):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 17 (ATO Corpus Quality Audit):** TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 18 (ATO Attack Variation):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 19 (Difficulty & Novelty):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 20-21 (Hard Diversity Investigation & Correction):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 22-23 (Easy Diversity & ATO Qual):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 24-28 (APP Selection, Implementation & Variation):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 29 (Red Team Cross-Family Qual):** CODE EXISTS, TEST EXISTS, REPORT EXISTS, RUNTIME EVIDENCE EXISTS.
*   **Stage 30 (Final Forensics):** REPORT EXISTS.
*   **Git Status:** All of the above (except previously committed base architectures) are UNTRACKED files or UNSTAGED modifications in the local worktree.

## 9. APP Multi-Trace Corpus Evidence
Evidence found in `reports/final_qualification_data.json`:
*   **APP requested count:** 200
*   **APP accepted count:** 157
*   **APP rejected count:** 1534
*   **Acceptance rate:** 9.28%
*   **Difficulty breakdown:** easy: 7, medium: 50, hard: 50, advanced: 50. (Easy legitimately hits an entropy ceiling).
*   **Raw traces available:** NO. Only the JSON metadata breakdown was persisted.

## 10. Why Stages 13–30 Existed Without Gates
**FACT:** The `git log` shows zero commits for Stages 13–30. `git status` shows dozens of reports, tests, and source modifications residing purely as untracked/unstaged changes. 
**INFERENCE:** The AI agent executing the project implementation operated inside a single continuous terminal session/workspace. It iterated through Stages 13 to 30, generating passing tests and internal reports, but neglected to formally `git add` and `git commit` at the successful conclusion of each explicit stage-gate as strictly mandated by the `STAGE_STATUS.md` workflow rules. This allowed vast architectural drift without incremental version control checkpoints.

## 11. Known Uncertainties
*   Because the corpora were never serialized, we cannot verify if downstream ML dataset assemblers would have caught the `attack_id` primary-key collision during `pandas.merge()` operations.
*   We cannot verify exactly which specific subsets of APP traces were rejected by `NoveltyIndex` versus structural blockages, though summary numbers exist.

## 12. Recommended Next Step
1. Implement the PRNG salting fix in `StatefulSimulator`.
2. Re-run `scripts/run_final_red_team_qualification.py` and modify it to explicitly dump the `ObservableAttackTrace` objects to disk as JSON to prove unique distributions.
3. Commit the enormous local worktree backlog to Git.

## 13. Addendum: Historical ATO Acceptance Baseline
*(Note added during pre-commit verification)*
A previous review noted a historical ATO acceptance rate of 76.33% (100 accepted / 131 attempts). It is critical to document that this measurement came from **Stage 15**, which executed strictly *before* the Novelty Engine filters and Uniform Difficulty Quotas were introduced (Stage 19+). 
In the modern, novelty-enabled pipeline (Stage 23 onward), the baseline ATO acceptance rate is legitimately ~15%, driven predominantly by the mass rejection of near-duplicate 'easy' attacks. The 76.33% figure is a pre-novelty artifact and is NOT comparable to the current pipeline's ~15% baseline.
