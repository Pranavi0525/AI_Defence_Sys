# Leakage Audit -- risk_fusion.py / decision_policy.py

Status: AUDIT COMPLETE. Two findings below; neither has been code-fixed
in this pass (see "Disposition" per finding) -- this document exists to
make both visible and auditable rather than silently inherited.

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

**Disposition:** documented, not fixed in this pass. Fixing it safely
means touching the GCN/Autoencoder training loop and re-validating
`test_risk_fusion.py`, which is more than a documentation-only change
should carry. Recorded here as the concrete next step.

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
