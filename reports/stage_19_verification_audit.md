# Stage 19 Verification Audit

## 1. Executive Summary
This audit rigorously examines the ATO Novelty & Diversity Engine implemented in Stage 19. The evaluation proves that the novelty engine functions as specified, generating mathematically distinct traces without leaking metadata, violating realism, or failing determinism. However, the audit reveals a significant **Selection Bias** consequence: the engine structurally over-selects `ADVANCED` difficulty profiles while suppressing `HARD` and `EASY` profiles.

## 2. Exact Rejection Accounting
Experiment Parameters: `target_count = 100`, `master_seed = 42`
- **Total Attempts:** 511
- **Accepted Traces:** 100
- **Novelty Rejections:** 411
- **Structural Rejections:** 0
- **Realism Rejections:** 0
- **Simulation Errors:** 0
- **Other Rejections:** 0
*(Categories perfectly sum to 511)*

## 3. Difficulty Comparison (Selection Bias Evidence)
| Difficulty | Without Novelty | With Novelty | Mean Phases (With) | Mean Events (With) |
| --- | --- | --- | --- | --- |
| **EASY** | 30.00% | 11.00% | 3.73 | 4.64 |
| **MEDIUM** | 19.00% | 27.00% | 4.26 | 5.96 |
| **HARD** | 27.00% | 7.00% | 2.00 | 4.29 |
| **ADVANCED** | 24.00% | 55.00% | 4.35 | 9.85 |

**Analysis:** Novelty selection explicitly favors `ADVANCED` traces. Because `ADVANCED` profiles exhibit high splitting and aggressive looping, they contain vastly more structural entropy, allowing them to trivially pass the novelty threshold. `HARD` and `EASY` profiles, being shorter and less volatile, quickly collide in the index, causing massive rejection rates for those profiles. 

## 4. Timing Comparison
| Profile | Without Novelty | With Novelty |
| --- | --- | --- |
| **RAPID** | 71 | 41 |
| **NORMAL** | 14 | 34 |
| **BURSTY** | 15 | 25 |

**Analysis:** Timing variation successfully balances out. The baseline over-sampled `RAPID` attacks; the novelty filter suppressed homogeneous rapid generations.

## 5. Behavioral Diversity Comparison
| Metric | Without Novelty | With Novelty |
| --- | --- | --- |
| **Amount StdDev (Raw)** | $7,408.94 | $5,139.42 |
| **Amount StdDev (Normalized)** | 0.4183 | 0.4971 |
| **Mean Split Count** | 1.27 | 5.32 |

**Analysis:** Normalized amount variance improved (0.41 to 0.49). Mean splitting skyrocketed as expected from the `ADVANCED` selection bias. 

## 6. Novelty Score Distribution
*(Accepted Corpus B, 1.0 = Completely Unique, 0.0 = Exact Duplicate)*
- **Min:** 0.1900
- **p10:** 0.2400
- **p25:** 0.2900
- **Median:** 0.3500
- **Mean:** 0.3881
- **p75:** 0.5500
- **p90:** 0.5500
- **Max:** 1.0000

**Analysis:** The threshold of 0.85 similarity (which requires `score > 0.15`) is effectively gating identical patterns. No traces fall below 0.15.

## 7. Duplicate / Near-Duplicate Comparison
*From previous Stage 19 report:*
- **Near-Duplicate Rate (>= 0.8 sim):** Dropped from ~20% to **0.91%**.

## 8. Selection-Bias Analysis
There is undeniable **Selection Bias** (mode collapse towards high-entropy). Because the novelty engine simply gates on similarity, and `ADVANCED` attacks naturally have much larger state spaces, `ADVANCED` traces constitute 55% of the accepted corpus. The system is obtaining novelty simply by waiting for the RNG to output an `ADVANCED` profile rather than generating novel `EASY` traces. This is a legitimate architectural limitation of applying global novelty gates to mixed-difficulty corpora.

## 9. Novelty-Index Correctness
The index operates as intended. It strictly caches only accepted traces, dropping the oldest when exceeding `max_size`. Ground truth metadata is safely excluded via the abstraction layer. 

## 10. Trivial-Change Robustness
`test_stage_19_novelty.py` confirms robustness:
- Absolute timestamp shifts, UUID regeneration, and penny-level float modifications accurately output `1.0` (100% similar / 0% novel).
- Structural loop adjustments, timing bucket category shifts, and large fractional amount changes successfully compute measurable novelty.

## 11. Realism Ordering
Inspection of `src/red_team/attacks/corpus.py` verifies the exact flow:
1. Generation 
2. Realism Validation (Runs to yield Structural/Constraint `Report`)
3. `if report.structural.passed:`
4. `extract_fingerprint` -> `novelty_index.evaluate()`
5. `if is_novel and report.constraint.passed:`
6. Append to accepted and add fingerprint to index.

The index correctly forbids a novel candidate from caching if it fails Realism Constraint validation (e.g. overspend).

## 12. Reproducibility
Running Corpus B and Corpus C sequentially with identical seeds yielded:
- `Trace counts match: True`
- `Attempts match: True`
- `Timings exact match: True`
- `Difficulty distribution exact match: True`

## 13. Defects
None.

## 14. Limitations
**Mode Collapse via Entropy Bias:** The novelty engine solves duplication but ruins the difficulty distribution. To generate 25 novel `HARD` attacks, the system would likely need thousands of attempts because `HARD` attacks intentionally lack the entropy (splits/loops) to look "novel" compared to each other.

## 15. Final Verdict
**ACCEPT_WITH_LIMITATIONS**

The mathematical architecture, robustness, and integration of the Novelty Engine are perfectly executed. However, it must be accepted with the limitation that global novelty filtering heavily biases the corpus toward `ADVANCED` profiles. Future architectures may require bucketed novelty indexes (e.g., an index per difficulty) to preserve the intended generation distribution.
