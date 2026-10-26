# Project Status Audit
**Date:** 2026-08-27

## 1. NORMAL WORLD TRANSACTION CATEGORY SCHEMA

**Finding: No analog to `ProductCD` exists in the synthetic Transaction schema.**
Inspection of `src/red_team/schemas/entities.py` confirms that the synthetic `Transaction` model lacks a direct categorical analog to IEEE-CIS's `ProductCD`. The closest fields describe channel or type, but not abstract product categories.

Quote from `Transaction` schema (`src/red_team/schemas/entities.py`):
```python
    transaction_type: Literal["purchase", "transfer", "withdrawal", "payment", "refund"] = Field(
        ..., description="Type of transaction."
    )
    channel: Literal["online", "pos", "atm", "mobile", "branch"] = Field(
        ..., description="Channel through which the transaction occurred."
    )
```

**Finding: `behavior.py` governs EVENT TYPE, not category.**
In `src/red_team/world/behavior.py`, the random choice dictates the underlying action structure (purchase vs. transfer), which maps strictly to `transaction_type`:
```python
        # Decide type (purchase vs transfer)
        tx_type = self.rng.choices(["purchase", "transfer"], weights=[0.8, 0.2])[0]
```
This is an EVENT TYPE choice. It does not implicitly stand in for a transaction category like `ProductCD`, as purchases require matching to a `merchant_id` while transfers match to a `beneficiary_id`.

**Applicability of Stage 4.5F Next-Stage Idea:**
The idea to "evaluate adopting the naive previous_ProductCD-persistence rule" **does not map onto the current schema at all**. To apply a persistence rule in the Normal World, it would have to be reframed to target a native domain field (e.g., predicting `transaction_type`, `merchant.mcc_code`, or `channel`). 

## 2. CALIBRATION SYSTEM EXECUTION STATUS

**Finding: Implemented but Unexecuted Code.**
Inspection of the `src/red_team/calibration/` directory reveals several existing modules (`behavioral.py`, `dependency.py`, `graph_metrics.py`, `marginal.py`, `structural.py`, `temporal.py`). However, a search through the project output logs, scratch directories, and `STAGE_STATUS.md` reveals no evidence that they have ever been run against actual Normal World output. 
They currently exist solely as implemented-but-unexecuted code. No actual calibration results exist.

## 3. ATO CORPUS REJECTION BREAKDOWN

**Finding: Breakdown does not exist.**
A recursive search for an "ATO corpus", traces, or rejection records (specifically tracking the "100-trace corpus (100 accepted / 31 rejected / 131 attempts)") returned zero results. There is no log or record locally documenting why any of the 31 rejected traces failed realism validation. This breakdown doesn't currently exist.

## 4. STAGE_STATUS.md FULL CONSISTENCY CHECK

A full review of `STAGE_STATUS.md` reveals the following inconsistencies:

1. **Stage 4.5C / 4.5D Status Contradiction:**
   - `IEEE-CIS Stage 4.5C` has `Status: PRELIMINARY`. 
   - `IEEE-CIS Stage 4.5D` lists `IEEE-CIS Stage 4.5C` as its prerequisite and is marked `Status: COMPLETED`.
   - *Violation:* The rules explicitly state `PRELIMINARY != COMPLETED` and mandate stopping if any prerequisite is incomplete.
2. **Stage 4.5E / 4.5E Addendum Next Stage Contradiction:**
   - `IEEE-CIS Stage 4.5E` has `Next stage allowed: NO`.
   - `IEEE-CIS Stage 4.5E Addendum` lists `IEEE-CIS Stage 4.5E` as its prerequisite and is marked `Status: COMPLETED (CONDITIONAL)`.
   - *Violation:* Stage 4.5E explicitly disallowed proceeding to the next stage.
3. **Stage 4.5E Addendum Missing Commit Hash:**
   - `IEEE-CIS Stage 4.5E Addendum` lists `Commit: (Pending Review / Uncommitted)`. The commit was executed (hash `ea1e7f5` or later), but the file was never updated to reflect it.
4. **Stage 4.5D Evidence Insufficiency:**
   - `IEEE-CIS Stage 4.5D` lists `Commit: N/A` and `Evidence: STAGE 4.5D ... report; historical reconstruction`.
   - *Violation:* The report does not exist on disk, meaning the evidence relies entirely on historical reconstruction. The rules state: "`HISTORICAL RECONSTRUCTION` alone is NOT sufficient."
