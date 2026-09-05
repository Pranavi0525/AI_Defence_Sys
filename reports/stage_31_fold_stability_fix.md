# Stage 31 -- Identity-Stable K-Fold Fix (Round1 -> Round2 Phantom-Regression Bug)

Status: FIXED (code already in place; this document was the missing stage
write-up, added retroactively)
Where: `blue_team_pipeline.py` (`stable_fold_id`, `stable_kfold_split`)
Consumed by: `blue_team_pipeline.cross_validated_evaluate`,
`blue_team_pipeline.compute_stage_1_2_cascade`,
`blue_team_pipeline.EvaluationHarness.run`, and -- transitively, by reusing
the same `folds` list -- `cascade_with_graph.py`, `cascade_with_autoencoder.py`,
`risk_fusion.py`, and `retrain_round2.py`.

## The bug

`sklearn.model_selection.StratifiedKFold.split(X, y)` assigns fold
membership by a row's **position** in the array, not by any property of
the record itself. That's invisible as long as the dataframe never
changes shape or order between runs. It stops being invisible the moment
new rows get appended -- which is exactly what `retrain_round2.py` does:
Round 2 = Round 1's corpus + N Stage-2-validated ATO hard examples,
concatenated on the end.

Appending rows shifts every subsequent row's array position, which
reshuffles which fold each *pre-existing* row lands in. Concretely: a
transaction that was in fold 3 (held out, i.e. test) under Round 1 could
land in fold 1 (used for training) under Round 2, or vice versa, purely
because of where the new rows were inserted -- nothing about that
transaction changed. Any precision/recall/FP delta measured between
Round 1 and Round 2 was then a mix of two effects tangled together:
genuine model improvement from the new training data, and pure fold
reshuffling noise. A false positive count that grew from 7 to 10
(42.9% relative) could be reported as a real regression when some
fraction of it was actually rows changing which side of the train/test
split they were on for reasons that have nothing to do with the model.
This is the "phantom regression" the fix is named for -- a regression
that looks real in the numbers but isn't attributable to anything the
retrain actually did.

## The fix

`stable_fold_id(trace_id, random_state)` hashes each record's own
`trace_id` (SHA-256, salted with `random_state` so different seeds still
give different-but-reproducible partitions) instead of relying on array
position. `stable_kfold_split(df, y_col, n_splits, random_state)` is a
drop-in replacement for `StratifiedKFold(...).split(X, y)` -- same
`(train_idx, test_idx)` positional-index return contract every existing
caller already expects -- but fold membership is derived purely from
that per-row hash. Stratification is preserved separately: within each
class, rows are ranked by their stable hash and assigned
`fold = rank % n_splits`, keeping per-fold class balance close to what
`StratifiedKFold` guarantees without depending on row order. A row's
fold assignment is now a pure function of `(trace_id, random_state)` --
it cannot change no matter what gets appended, removed, or reordered
elsewhere in the dataframe.

A safety check is built in: if any fold's fraud rate drifts more than
10 percentage points from the true overall rate (possible with very
small classes, since identity-stability trades a small amount of
stratification precision for that guarantee), `stable_kfold_split`
prints a `[stable_fold WARNING]` naming the fold and the drift so the
trade-off stays visible rather than silent.

## Why this needed to be a shared, single source of truth

Stage 1+2 (`compute_stage_1_2_cascade`), the GCN (`cascade_with_graph.py`),
the autoencoder (`cascade_with_autoencoder.py`), and Risk Fusion
(`risk_fusion.py`) all score the same rows and need to agree on which
rows were held out for which model, or their OOF scores can't legitimately
be stacked as meta-features. `risk_fusion.py` doesn't call
`stable_kfold_split` itself -- it reuses the exact `folds` list returned
by `btp.compute_stage_1_2_cascade(...)`, which is the single source of
truth for the partition (see `risk_fusion.py`'s own module docstring,
"IDENTICAL fold partition"). That means the identity-stability guarantee
propagates for free through every downstream stage without each one
needing its own copy of the logic. `retrain_round2.py` gets the same
guarantee the same way: `evaluate_round()` calls
`cross_validated_evaluate()` and `EvaluationHarness.run()`, both of which
call `stable_kfold_split` internally, so Round 1 and Round 2 evaluations
are automatically comparable on a row-by-row basis.

## What this fix does NOT address (separate, already-documented limitation)

Identity-stable folds fix *positional* instability -- they do not change
the fact that the split is still row-level, not customer/entity-level.
`BLUE_TEAM_INTEGRATION_SPEC.md` Section 9 calls for splitting by
customer when one customer can contribute multiple session-windowed
rows; that has not been implemented, and a customer can still have some
rows in the train fold and others in the test fold for the same split.
This is called out explicitly in `risk_fusion.py`'s module docstring
("CARRIED-FORWARD LIMITATION") and is tracked as a separate, open item
-- see the Leakage Audit (`reports/stage_leakage_audit_risk_fusion_decision_policy.md`)
for the full writeup of that limitation and why it wasn't fixed as part
of this stage.

## Verification

- `python3 -m py_compile blue_team_pipeline.py risk_fusion.py
  retrain_round2.py` -- syntax-valid.
- `PYTHONPATH=src python3 -m pytest tests/ -q` -- 468 passed (run in this
  session with the real dependency stack installed; see handoff note for
  the exact command).
- `PYTHONPATH=src python3 test_risk_fusion.py` -- 7 passed, including
  `test_deterministic_given_fixed_seed`, which is the direct regression
  guard for this fix: it asserts that the fused OOF scores are bit-identical
  across two runs with the same seed, which would break immediately if
  fold assignment depended on anything positional or otherwise
  non-deterministic.
