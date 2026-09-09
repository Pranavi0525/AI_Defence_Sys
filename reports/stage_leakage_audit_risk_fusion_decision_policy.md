# Leakage Audit -- risk_fusion.py / decision_policy.py

Status: AUDIT COMPLETE (original pass). Two findings below were
documented but NOT code-fixed in the original audit pass. Finding 1 was
subsequently remediated in **Phase 4A** (see the "Phase 4A Remediation"
section immediately following Finding 1, added afterward -- this does
NOT rewrite the original audit text below, which is preserved verbatim
as the historical record of what was found and why it was left unfixed
at the time). Finding 2 remains undisposed as of Phase 4A; it was out of
Phase 4A's approved scope (risk_fusion.py only) and has not been touched.

Scope: every place either file touches labels, computes a statistic
across rows, or picks a value by looking at outcomes, checked for
whether test-fold information (or future/target information) leaks into
something used at train or decision time. Stage 1+2's own cascade
(`blue_team_pipeline.compute_stage_1_2_cascade`) was re-checked as a
baseline for comparison but is out of scope for this audit -- it's an
XGBoost model on raw features with no standardization step, so it isn't
exposed to Finding 1 at all.

## Finding 1 -- Global feature standardization computed before the fold split (risk_fusion.py)

**Where:** `compute_base_scores()`, `risk_fusion.py` lines ~124-138.

```python
X_raw = df[feature_cols].fillna(0).values.astype(float)
...
mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
X_std = (X_raw - mu) / sigma
...
stage_1_2_proba, escalate, folds = btp.compute_stage_1_2_cascade(...)
...
for fold, (train_idx, test_idx) in enumerate(folds, start=1):
    ...
    gcn = OneLayerGCN(...); train_gcn(gcn, M, y.astype(float), train_mask, ...)
    ...
    train_ae(ae, X_std[legit_train_idx], ...)
```

`mu` and `sigma` are computed once, over every row in `df` -- test rows
from every fold included -- **before** the fold loop starts. Every
fold's GCN and Autoencoder are then trained on `X_std` (and the graph
message-passed `M = A_hat @ X_std`), which was standardized using
statistics that included that very fold's held-out rows.

**Why it's leakage, not just style:** a fold's training features are
supposed to depend only on information available from the training
split. Here, every row's standardized value carries a small amount of
information about the mean/variance of the *whole* dataset, including
its own fold's test rows. In effect, each fold gets a faint, indirect
signal about its own held-out distribution baked into its training
inputs.

**Severity, honestly assessed:** low-to-moderate, not "the numbers are
fake." `mu`/`sigma` are dataset-level location/scale constants, not
label-derived -- they don't encode fraud/legit information directly,
and with a large-enough dataset one fold's contribution to a global
mean/std is small. It will not produce a headline-grabbing inflated
AUC. But it is a real violation of fold isolation, it compounds with
every stage that inherits `X_std` (GCN *and* Autoencoder both use it),
and it's the kind of leakage that becomes worse, not better, as the
corpus grows more skewed (e.g. once real fraud rates are much lower
than this red-team corpus, or once `retrain_round2.py` appends more
rows and shifts the global mean further).

**What the correct fix looks like** (not applied in this pass -- flagged
for a follow-up, since it requires re-deriving `mu`/`sigma` per fold and
re-verifying `test_risk_fusion.py`'s `test_deterministic_given_fixed_seed`
still holds under a per-fold standardizer):
```python
for fold, (train_idx, test_idx) in enumerate(folds, start=1):
    mu = X_raw[train_idx].mean(axis=0)
    sigma = X_raw[train_idx].std(axis=0) + 1e-8
    X_std_fold = (X_raw - mu) / sigma   # apply train stats to ALL rows,
                                         # fit only on train_idx
    M_fold = A_hat @ X_std_fold
    ...
```
This does mean re-computing `M` (the message-passed features) inside
the fold loop instead of once outside it -- a real cost increase (5x
the matrix multiplies instead of 1x) that's worth flagging to whoever
picks this up, since it changes the function's current "compute M once,
reuse across folds" performance characteristic.

**Original disposition (audit pass):** documented, not fixed in this
pass. Fixing it safely means touching the GCN/Autoencoder training loop
and re-validating `test_risk_fusion.py`, which is more than a
documentation-only change should carry. Recorded here as the concrete
next step.

## Phase 4A Remediation -- Finding 1 (risk_fusion.py)

**1. Original finding:** as documented above -- `mu`/`sigma` for feature
standardization were computed once over the full dataset (`X_raw.mean(axis=0)`,
`X_raw.std(axis=0)`), before the fold loop, in `compute_base_scores()`.

**2. Exact location:** `risk_fusion.py`, `compute_base_scores()`,
originally lines 124-138 in the pre-remediation file.

**3. Evidence:** confirmed independently at the start of this
remediation pass by re-reading the live file (`git diff` showed zero
changes to `risk_fusion.py` versus the commit that introduced it,
`7a8054d`) -- the leaking code was still present, unchanged, at the
start of Phase 4A. The identical pattern also exists, unmodified, in
`cascade_with_graph.py:400` and `cascade_with_autoencoder.py:101-102`
(see item 12).

**4. Why it is preprocessing leakage:** every fold's held-out (test)
rows contributed to the `mu`/`sigma` used to standardize those same
rows before scoring them -- a violation of fold isolation. Not a label
leak (`mu`/`sigma` never touch `y`), but a genuine violation of the
out-of-fold principle for the GCN and Autoencoder's input features.

**5. Original disposition:** documented, not fixed (see above).

**6. New remediation:** `mu`/`sigma` are now computed per fold, from
`X_raw[train_idx]` only, and applied (frozen) to every row scored in
that fold -- including its own held-out rows. `M` (`A_hat @ X_std`) is
recomputed per fold from the fold-local `X_std_fold` rather than once
outside the loop. The existing `+ 1e-8` zero-std guard is preserved,
now applied per-fold instead of once globally. No other numerical
policy (NaN handling via `.fillna(0)`, dtype, inf behavior) was changed.

**7. Corrected data flow:**
```python
for fold, (train_idx, test_idx) in enumerate(folds, start=1):
    mu = X_raw[train_idx].mean(axis=0)
    sigma = X_raw[train_idx].std(axis=0) + 1e-8
    X_std_fold = (X_raw - mu) / sigma   # frozen train-only stats,
                                          # applied to ALL rows this fold touches
    M_fold = A_hat @ X_std_fold          # recomputed per fold (5x cost, same math)
    ...
    train_gcn(gcn, M_fold, ...)
    train_ae(ae, X_std_fold[legit_train_idx], ...)
    ae.reconstruction_error(X_std_fold[test_idx])
```
No changes to GCN/AE architecture, hyperparameters, epochs, hidden
dims, random seeds, adjacency construction, Stage 1/2 fold assignment,
the fusion meta-model, or decision thresholds. `decision_policy.py`,
`blue_team_pipeline.py`, and `retrain_round2.py` were not touched.

**8. Regression test:** `test_risk_fusion_leakage.py` (new). Calls
`risk_fusion.compute_base_scores()` itself (not a reimplementation of
the standardization math) with the base-model internals (GCN,
Autoencoder, Stage 1+2 cascade) monkeypatched to transparent recorders
that capture the exact arrays production code passes them. A synthetic
dataset places an extreme value (10,000.0) on one feature, in exactly
one held-out row. Checks:
   - training-fold mean/std do not shift when that held-out extreme
     value is introduced (properties 1-3 from the remediation spec);
   - the held-out row is transformed using the frozen training
     mu/sigma, verified against an independently-reconstructed
     train-only mu/sigma computed straight from raw data (property 4).

**9. Proof the test catches the old bug:** the identical test class was
run against the actual pre-fix `risk_fusion.py` (reverted via `git
stash`, not a reimplementation) -- both checks **failed**, with the
held-out extreme value visibly shifting the "training" standardized
values (max abs diff ~23,498, i.e. the injected outlier leaking straight
through). The fix was then restored via `git stash pop` and the same
two checks **passed**. A third test in the file additionally runs the
same property check against a byte-for-byte reconstruction of the old
function body as a standalone regression guard (so the distinction is
covered even if someone re-runs only this file without git access).

**10. Verification results:**
   - `pytest test_risk_fusion_leakage.py` -> **3 passed**
   - `pytest test_risk_fusion.py` -> **7 passed** (pre-existing, unaffected)
   - `pytest web_prototype/api/tests` -> **12 passed**
   - `pytest -q` (repo root, default `testpaths=["tests"]`) -> **468 passed**
   - Known limitation, reconfirmed: `pyproject.toml`'s
     `testpaths=["tests"]` means the default run does NOT include
     `test_risk_fusion.py`, `test_risk_fusion_leakage.py`, or
     `web_prototype/api/tests` -- each was run explicitly, above, since
     the default command is not the complete test suite. Additionally,
     running `tests/`, `test_risk_fusion.py`, `test_risk_fusion_leakage.py`,
     and `web_prototype/api/tests` together in one pytest invocation
     fails at collection (`web_prototype/api/tests` is a package
     literally named `tests`, colliding with the root `tests/`
     directory) -- a pre-existing repo-structure issue, not introduced
     here, worth a real fix in a later phase.

**11. Metric impact (real corpus, before -> after, precision/recall/f1):**
   - `stage_1_2_only`: unchanged (0.989 / 0.963 / 0.976)
   - `stage_1_2_plus_gcn_max`: unchanged (0.989 / 0.968 / 0.978)
   - `stage_1_2_plus_autoencoder_max`: precision 0.8605 -> 0.8545
     (delta -0.0061), recall unchanged (0.973), f1 0.9134 -> 0.9100.
     Confusion matrix FP 59 -> 62.
   - `naive_max_all_three`: precision 0.8612 -> 0.8551 (delta -0.0060),
     recall unchanged (0.979), f1 0.9161 -> 0.9127. FP 59 -> 62.
   - `risk_fusion_stacked_lr` (production candidate): unchanged at 3
     decimal places (0.986 / 0.965 / 0.976); confusion matrix identical
     bit-for-bit. Fusion coefficients shifted slightly
     (`ae_score` 2.5519 -> 2.5206, `gcn_score` 2.4371 -> 2.4338,
     `stage_1_2_score` 6.8958 -> 6.8999, intercept -3.8434 -> -3.8456)
     but not enough to flip any row's classification at the fixed
     decision threshold in this corpus.
   - No shape or dtype changes; no new NaN/inf values introduced (spot
     checked via the successful `json.dump` round-trip and the
     regression test's explicit finite-value assertions).
   - Not forced back to old values -- the autoencoder is now honestly
     stricter without its "normal" baseline having quietly seen
     test-fold data, which is the expected, correct cost of this fix.

**12. Artifacts:**
   - `blue_team_output/risk_fusion_results.json` -- **AFFECTED,
     REGENERATED.** Command: `PYTHONPATH=src:. python3 risk_fusion.py`
     (run from repo root). Reflects the corrected fold-local
     standardization.
   - `blue_team_output_FROZEN/risk_fusion_results.json` -- **AFFECTED
     IN PRINCIPLE, PRESERVED AS-IS.** Not regenerated; this artifact is
     already from a different population/run than either the before-
     or after-fix run in this pass, and `streamlit_app/app.py` reads
     directly from this frozen path, so regenerating it is a decision
     deferred to whoever owns that dashboard's refresh cycle, not
     bundled into this fix.
   - `decision_policy_results.json` (root) -- **AFFECTED, REGENERATION
     DEFERRED.** `decision_policy.py` calls `risk_fusion.run_risk_fusion`
     live (imports `risk_fusion as rf`, `rf.run_risk_fusion(...)`), so
     it will pick up this fix automatically the next time it is run,
     but it was NOT re-run in this pass (out of the approved Phase 4A
     scope, which explicitly excludes `decision_policy.py`).
   - `frozen_reports/decision_policy_results.json` -- **NOT AFFECTED**,
     historical snapshot, untouched.
   - `web_prototype/api/*` -- reads `blue_team_output/risk_fusion_results.json`
     (the regenerated, non-frozen copy) via `web_prototype/api/reports.py`.
     **AFFECTED, ALREADY CURRENT** as of the regeneration above; no
     separate action needed.
   - No other artifacts were regenerated. Nothing was committed.

**13. Remaining identical patterns / limitations:** identical
preprocessing pattern identified in two additional files.
Remediation deferred from Phase 4A because the approved scope is
risk_fusion.py only. Specifically: `cascade_with_graph.py:400` and
`cascade_with_autoencoder.py:101-102` both still compute `mu`/`sigma`
over the full dataset before their own fold loops -- confirmed
untouched (`git diff` shows zero changes to either file in this pass).
Finding 2 (decision_policy.py threshold selection/evaluation
conflation) also remains fully undisposed, unchanged from the original
audit above. The row-level-vs-customer-level fold split limitation
(see "Not re-litigated here" section below) is also unaffected by this
fix and remains open.

## Finding 2 -- Decision thresholds are selected and evaluated on the same rows (decision_policy.py)

**Where:** `optimize_thresholds()` (grid search) and `policy_stats()` /
`diagnose_prevalence_bug()` (reporting), `decision_policy.py`.

`optimize_thresholds(y, proba, dollars, cost, ...)` grid-searches
`(t_review, t_block)` to minimize `expected_cost()` computed over the
entire validation population (`y`, `proba`, `dollars` -- the full,
5-fold-CV out-of-fold cascade/fusion score for every row). The winning
thresholds are then handed straight to `policy_stats()`, which reports
`allow_rate` / `block_rate` / `fraud_recall_blocked_plus_review` /
`expected_cost_at_assumed_prevalence` and the liability breakdown --
**on that same population**. There is no separate calibration split
held out from the threshold search before scoring the chosen policy.

**Why it's leakage, not just style:** `proba` itself is legitimately
out-of-fold with respect to the *base model* (Stage 1+2+3 or Risk
Fusion never saw a row's label during its own training) -- that part is
fine and correctly documented. But `t_review`/`t_block` are themselves
free parameters being fit to this dataset, and a 60x60 threshold grid
search (`n_candidates=60` by default, ~3,600 candidate pairs after the
`t_review <= t_block` filter) picking whichever pair happens to minimize
cost on a ~1,458-row validation set has real freedom to fit noise in
that specific sample -- particularly for the by-family and
liability-breakdown breakdowns, where per-family fraud counts (e.g. a
single-digit MULE_NETWORK count) are small enough that the "optimal"
threshold can be sensitive to a handful of rows. Reporting cost/recall
on the same rows the thresholds were chosen against means those numbers
are optimistic versus what the policy would actually achieve on unseen
traffic -- the standard train/eval conflation problem, just at the
threshold-selection layer instead of the model-training layer.

**Severity, honestly assessed:** moderate. This doesn't invalidate the
qualitative finding the file is built around (the block-everyone /
naive-prevalence bug and its fix are real and would show up regardless
of this issue). But every specific number in `decision_policy_results.json`
-- `expected_cost_at_assumed_prevalence`, the block/review/allow rates,
`fraud_recall_blocked_plus_review`, and every per-family
`dollars_allowed_through_this_institution_liable_for` figure -- should
be read as "best achievable on this sample," not as an unbiased estimate
of production performance.

**What the correct fix looks like** (not applied in this pass): split
the OOF-scored validation population itself into a threshold-selection
subset and a reporting subset (e.g. a further stratified split, or
reusing `folds` from the base cascade to do threshold selection on
folds 1-4 and report on fold 5, rotated), or at minimum add a bootstrap
confidence interval around the reported cost/recall numbers so their
sampling variance is visible instead of implied to be zero.

**Disposition:** documented, not fixed in this pass -- same reasoning
as Finding 1: correctly re-architecting the validation split changes
`get_validation_data_fused()`'s contract and would need its own
regression tests, which belongs in a dedicated follow-up rather than
bundled into this audit.

## Not re-litigated here (already documented upstream)

`risk_fusion.py`'s own module docstring already flags that every split
used by Stage 1+2 (and therefore GCN, Autoencoder, Risk Fusion, and
Decision Policy, all of which build on `compute_stage_1_2_cascade`'s
`folds`) is **row-level**, not customer/entity-level, per
`BLUE_TEAM_INTEGRATION_SPEC.md` Section 9. A customer with multiple
session-windowed rows can have some rows in train and others in test for
the same fold. This audit re-confirms that limitation is real and still
open, but doesn't repeat the full writeup -- see `risk_fusion.py`'s
"CARRIED-FORWARD LIMITATION" docstring section and
`reports/stage_31_fold_stability_fix.md`'s "What this fix does NOT
address" section for the existing coverage.

## What was checked and found clean

- `fit_fusion_oof()` (`risk_fusion.py`): meta-model fits only on
  `meta_X[train_idx]`/`y[train_idx]` and predicts only on `test_idx`,
  per fold. No label leakage into the meta-model.
- GCN/Autoencoder label usage in `compute_base_scores()`: both are
  trained using only `train_idx` labels (`train_mask`,
  `legit_train_idx`), scored only on `test_idx`. Labels themselves
  don't leak -- only the standardization statistics do (Finding 1).
- `get_validation_data()` / `get_validation_data_fused()`
  (`decision_policy.py`): the `dollars` array is built by
  positionally zipping against `all_records` with an explicit assertion
  (`assert len(dollars) == len(df) == len(proba) == len(y)`) rather than
  a join key, and `build_feature_table_and_graph` is documented as
  preserving row order -- correct and leakage-free construction, just
  worth flagging as a silent-misalignment risk if that ordering
  invariant is ever broken elsewhere (the assertion would catch a
  length mismatch but not a same-length reordering).
- `sample_weights()` / `expected_cost()` / `exposure_share()`
  (`decision_policy.py`): pure functions of `(y, target_fraud_rate)` /
  `(attack_family, cost)` respectively, no cross-row or cross-fold state.
- `stamp_artifact()` calls in both files: purely additive metadata, no
  interaction with model inputs or labels.
