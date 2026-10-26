# Stage 33: Role-Aware Sending/Receiving Liability Thresholds

Status: PRELIMINARY -- pending your review and official sign-off per
STAGE_STATUS.md's Mandatory Future Stage Update Procedure. This report
is a draft handed off for that review, not a self-certification.

## 1. Prerequisite Check (per STAGE_STATUS.md procedure)

- Stage 32 (MULE_NETWORK corpus freeze): Status = QUALIFIED for freeze
  (reports/stage_32_mule_corpus_freeze.md). Prerequisite satisfied.
- `decision_policy.py` prior to this stage had no role-aware liability
  logic (verified by inspection before starting).

## 2. Scope

Extends `decision_policy.py` so that a Review/Block decision is tagged
with WHO -- sending institution, receiving institution, or both --
bears the cost and should act, per attack family. Modeled on:

- **ACCOUNT_TAKEOVER**: unauthorised transaction. Under standard
  unauthorised-payment rules (PSD2 / UK Payment Services Regulations),
  the customer's own (sending) bank must reimburse in full.
  `liable_side = SENDING`, full exposure share.
- **AUTHORIZED_PUSH_PAYMENT**: the customer authorised the payment
  themselves under deception. The UK Payment Systems Regulator's APP
  reimbursement requirement (in force since Oct 2024) splits
  reimbursement 50/50 between the sending PSP and receiving PSP by
  default. `liable_side = SHARED_50_50`, `acting_side = BOTH`,
  sending-side exposure share = 0.5 (configurable via
  `CostModel.app_sending_liability_share`).
- **MULE_NETWORK**: the fraud IS the receiving account. Liability for
  having onboarded and failed to flag it sits with the RECEIVING
  institution. `liable_side = RECEIVING`, full exposure share (modeled
  conservatively -- the sending bank's own reimbursement exposure isn't
  waived just because the mule account sits elsewhere).

## 3. Implementation

New in `decision_policy.py`:
- `LIABILITY_SIDE` mapping (keyed on the corpus's actual
  `attack_family` strings -- `ACCOUNT_TAKEOVER`,
  `AUTHORIZED_PUSH_PAYMENT`, `MULE_NETWORK` -- not abbreviations; see
  bug log below).
- `liable_side()`, `acting_side()`, `exposure_share()`.
- `expected_cost()` now optionally accepts `attack_family` and scales
  each fraud trace's $ loss by its `exposure_share()` before costing
  it, so the threshold search itself reflects this institution's own
  liability, not the full loss.
- `policy_stats()` now returns a `liability_breakdown` dict keyed by
  attack family, with counts, liable/acting side, and $ this
  institution is liable for on allowed-through fraud.
- `optimize_thresholds()`, `diagnose_prevalence_bug()`, and `main()`
  updated to thread `attack_family` (`df["attack_family"].values`)
  through end to end.

## 4. Bugs found and fixed during this stage (not pre-existing to this
   stage's own code -- found while getting it running)

1. **`load_all_records()` two-value unpack.** `cascade_with_graph.load_all_records(cfg)`
   returns a single list; ring membership is obtained separately via
   `load_real_ring_membership(cfg)` (this is how `cascade_with_graph.py`'s
   own `main()` and `risk_fusion.py` already called it). Three call
   sites still had the old `all_records, ring_ids = cwg.load_all_records(cfg)`
   pattern and raised `ValueError: too many values to unpack`:
   - `decision_policy.py` (`get_validation_data`, `get_validation_data_fused`) -- fixed.
   - `explainability.py` (`build_reference_artifacts`) -- fixed.
   - `miss_collector.py` -- fixed.
   This bug predates this stage and is unrelated to the liability
   logic; it surfaced because this stage was the first time
   `get_validation_data_fused()` was actually exercised end-to-end.

2. **`attack_family` string mismatch.** First implementation used
   abbreviations (`"ATO"`, `"APP"`) that don't match the corpus's
   actual `ground_truth.attack_family` values (`"ACCOUNT_TAKEOVER"`,
   `"AUTHORIZED_PUSH_PAYMENT"`, confirmed directly against
   `reports/ato_corpus_raw.json` / `reports/app_corpus_raw.json`).
   Caused ATO and APP to silently fall through to `liable_side = N/A`
   and full (unshared) exposure. A second, independent hardcoded
   `"APP"` check inside the `liability_breakdown()` helper had the
   same bug on a different line. Fixed by correcting both string
   literals and rewriting `receiving_liability_share` to derive from
   `liable_side()` instead of checking the family name a second time.

## 5. Results (your run, Windows, full pipeline, post-fix)

```
t_review = 0.1153, t_block = 0.9622
allow/review/block: 72.9% / 5.1% / 22.0%
legit blocked: 1 (0.1%)
fraud recall (block+review): 98.4%
expected_cost_at_assumed_prevalence: 1199.77

liability_breakdown:
  ACCOUNT_TAKEOVER:       liable=SENDING       acting=SENDING  share=1.0  $liable=0.0
  AUTHORIZED_PUSH_PAYMENT: liable=SHARED_50_50  acting=BOTH     share=0.5  $liable=5015.72  (of $10,031.44 total)
  MULE_NETWORK:           liable=RECEIVING     acting=RECEIVING share=1.0  $liable=1748.23
```

The APP 50/50 split is verified arithmetically correct: sending-side
liability ($5,015.72) is exactly half of total dollars allowed through
($10,031.44).

## 6. Verification performed

- `ast.parse()` syntax check on `decision_policy.py`,
  `explainability.py`, `miss_collector.py` after every edit.
- Unit-level smoke test of `liable_side()` / `acting_side()` /
  `exposure_share()` / `expected_cost()` / `optimize_thresholds()` /
  `liability_breakdown()` against synthetic data with the correct
  literal family strings, confirming role-aware expected cost <=
  family-blind expected cost at the same thresholds, and correct
  50/50 dollar split.
- Full pipeline run on your machine (Windows, real xgboost/GCN/
  autoencoder/risk-fusion stack, 5-fold CV) -- see Section 5.

## 7. NOT done as part of this stage

- `explainability.py` and `miss_collector.py` were fixed for the
  unpacking bug but NOT otherwise re-run or re-validated end-to-end
  as part of this stage.
- No change made to `STAGE_STATUS.md`. That file's tracked
  "Core project" / "IEEE-CIS" stage series does not currently include
  entries for the Stage 13-32 Red Team/Blue Team series this work
  extends, so there's no existing template row to slot this into
  without a broader decision about how that series should be tracked.
  Flagging rather than guessing at a new schema.
- No change made to `FINAL_VALIDATION_REPORT.md`.
- `case_dossier_examples.md` / web_prototype cross-bank view (also
  called out as "after upgrade" deliverables) are NOT part of this
  stage -- this stage is decision_policy.py's role-aware thresholds
  only.

## 8. Suggested next steps (for your call)

- Review this report and, if accepted, add the official stage entry to
  STAGE_STATUS.md with a real commit hash per the Mandatory Future
  Stage Update Procedure.
- Decide whether `case_dossier_examples.md` (explainability.py
  investigator-facing dossier) is the next item to pick up.
- Consider whether the `AUTHORIZED_PUSH_PAYMENT` / `ACCOUNT_TAKEOVER`
  string-literal duplication (now living in `LIABILITY_SIDE`,
  `exposure_share()`, and wherever else attack families get matched
  by string elsewhere in the codebase) is worth centralizing into a
  single enum/constant, given it just caused two bugs in one stage.
