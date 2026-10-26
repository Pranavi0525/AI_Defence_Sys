# Red Team AI — Stage Status Registry

This file is the authoritative project stage-status record. Do not infer stage completion from code, tests, or model artifacts. A stage is COMPLETED only when its stage report/acceptance gate has explicitly declared completion and this file has been updated.

## Stage Dependency Map

### Core project

Stage 1
→ Stage 2
→ Stage 3
→ Stage 4

### IEEE-CIS experimental/learning track

IEEE-CIS Stage 4.5A
→ IEEE-CIS Stage 4.5B
→ IEEE-CIS Stage 4.5C
→ IEEE-CIS Stage 4.5D
→ IEEE-CIS Stage 4.5E
→ IEEE-CIS Stage 4.5F

### PaySim parallel/context track

PaySim Stage 4.5A

PaySim Stage 4.5A is a PARALLEL / CONTEXT track.

It informs the overall dataset/objective decision.

It is NOT a prerequisite for IEEE-CIS Stage 4.5E.

The dependency graph MUST NOT show:

IEEE-CIS 4.5D
→ PaySim 4.5A
→ IEEE-CIS 4.5E

That dependency is incorrect.

## Status Integrity Rules

Do NOT infer completion from:

- source code existing
- tests passing
- model artifacts existing
- commits existing
- a later stage having been implemented
- good ML metrics

A stage may be marked COMPLETED only when:

1. Its acceptance criteria have passed.
2. Its required tests have passed.
3. Its stage report explicitly declares completion.
4. STAGE_STATUS.md is updated as part of the official stage-completion commit.

PRELIMINARY != COMPLETED

NOT STARTED != COMPLETED

BLOCKED != COMPLETED

NEEDS RECONSTRUCTION != COMPLETED

Never retroactively upgrade a stage to justify downstream work.

## Evidence Rule

Every stage entry MUST contain a checkable evidence reference.

`HISTORICAL RECONSTRUCTION` alone is NOT sufficient.

Evidence should identify the strongest available source, such as:

- stage report title
- report section
- repository artifact path
- commit hash
- or a combination

Example:

Evidence:
`Stage 4.5D report — "STAGE 4.5D — IEEE-CIS PROXY SENSITIVITY"; historical reconstruction`

If the exact report cannot be retrieved:

Evidence:
`Historical reconstruction; exact source artifact not available`

Do NOT invent report titles, paths, or evidence.

## Mandatory Future Stage Update Procedure

Before beginning every future stage:

1. Read STAGE_STATUS.md.
2. Identify all required prerequisites.
3. Verify their statuses.
4. Report prerequisite statuses before implementation.
5. STOP if any required prerequisite is incomplete.

After completing a stage:

1. Run required tests.
2. Produce the official stage acceptance report.
3. Update STAGE_STATUS.md.
4. Record:
   - status
   - completion date
   - commit
   - scope
   - evidence
   - prerequisite(s)
   - dependency role
   - next stage
   - next stage allowed
5. Commit STAGE_STATUS.md in the same official stage-completion commit.
6. Push to origin/master.
7. STOP.

This procedure applies to ALL future stages.

---

### Stage 1
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: Pre-7931ead
Scope: Project Initialization & Architecture
Evidence: Historical reconstruction based on commit Pre-7931ead; exact stage report artifact not available
Prerequisite(s): None
Dependency role: DOWNSTREAM
Next stage: Stage 2
Next stage allowed: YES

---

### Stage 2
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: 7022ba2
Scope: Entity, Event, Observable, Ground-Truth, and Provenance Schema Definitions
Evidence: Historical reconstruction based on commit 7022ba2; exact stage report artifact not available
Prerequisite(s): Stage 1
Dependency role: DOWNSTREAM
Next stage: Stage 3
Next stage allowed: YES

---

### Stage 3
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: fdd11a6
Scope: Feature/Dataset Registry Population
Evidence: Historical reconstruction based on commit fdd11a6; exact stage report artifact not available
Prerequisite(s): Stage 2
Dependency role: DOWNSTREAM
Next stage: Stage 4
Next stage allowed: YES

---

### Stage 4
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: b489e4f
Scope: Dataset Access & Preprocessing / Statistics Integrity
Evidence: Historical reconstruction based on commit b489e4f; exact stage report artifact not available
Prerequisite(s): Stage 3
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5A / PaySim Stage 4.5A
Next stage allowed: YES

---

### IEEE-CIS Stage 4.5A
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: Pre-19112e0
Scope: IEEE-CIS Feasibility Analysis
Evidence: `STAGE 4.5A — IEEE-CIS DATASET INSPECTION / feasibility report`; historical reconstruction
Prerequisite(s): Stage 4
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5B
Next stage allowed: YES

---

### IEEE-CIS Stage 4.5B
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: 19112e0
Scope: Build supervised sequence dataset
Evidence: `STAGE 4.5B — BUILD SUPERVISED SEQUENCE DATASET` report; commit 19112e0; historical reconstruction
Prerequisite(s): IEEE-CIS Stage 4.5A
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5C
Next stage allowed: YES

---

### IEEE-CIS Stage 4.5C
Status: PRELIMINARY
Completed on: N/A
Commit: fd75111
Scope: Train First ML Baseline (Logistic Regression)
Evidence: `Stage 4.5C — Train First ML Baseline` report; commit fd75111; explicitly preliminary/unapproved
Resolution: permanent PRELIMINARY, superseded by 4.5F
Prerequisite(s): IEEE-CIS Stage 4.5B
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5D
Next stage allowed: YES

---

### IEEE-CIS Stage 4.5D
Status: COMPLETED
Completed on: 2026-08-28
Commit: 90ec87d
Scope: IEEE-CIS Proxy Sensitivity Analysis
Evidence: reports/stage_4_5D_evidence.md; data/reference/ml_sequence/sensitivity_results/
Prerequisite(s): IEEE-CIS Stage 4.5C (ARTIFACT ONLY)
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5E
Next stage allowed: YES

---

### PaySim Stage 4.5A
Status: COMPLETED
Completed on: HISTORICAL RECONSTRUCTION
Commit: c4fe2b4
Scope: PaySim Feasibility Analysis & Dataset Selection Gate
Evidence: `PROCESS AUDIT — PAYSim FEASIBILITY ANALYSIS` report; commit c4fe2b4; historical reconstruction
Prerequisite(s): Stage 4
Dependency role: PARALLEL / CONTEXT
Next stage: N/A — parallel/context track; informs overall dataset/objective decisions
Next stage allowed: NO

---

### IEEE-CIS Stage 4.5E
Status: COMPLETED
Completed on: 2026-08-27
Commit: 38ae50c
Scope: IEEE-CIS Behavioral Proxy Redesign / Construct-Validity Correction
Evidence: `data/reference/ml_sequence/proxy_specification.json`; NO_DEFENSIBLE_INDIVIDUAL_PROXY = YES; Fallback: COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR
Prerequisite(s): 
- IEEE-CIS Stage 4.5A
- IEEE-CIS Stage 4.5B
- IEEE-CIS Stage 4.5C (ARTIFACT ONLY)
- IEEE-CIS Stage 4.5D
Dependency role: DOWNSTREAM
Next stage: Conditional — IEEE-CIS Stage 4.5F only if the selected objective/proxy passes the acceptance decision
Next stage allowed: NO (directly) — gated by IEEE-CIS Stage 4.5E Addendum, which independently governs whether Stage 4.5F may begin. See Addendum entry for current status.

---

### IEEE-CIS Stage 4.5E Addendum
Status: COMPLETED (CONDITIONAL)
Completed on: 2026-08-27
Commit: ea1e7f5
Scope: Validate COMPOSITE_PAYMENT_CONTEXT_BEHAVIOR as a defensible fallback objective
Evidence: reports/stage_4_5E_addendum.md (Revision + Section 6 + AMBIGUOUS profiling)
Verdict: CONDITIONAL — proxy is defensible for the majority of entities,
  EXCLUDING entities matching the rule:
    device_match_rate >= 0.50 AND top3_known_device_coverage < 0.60
  This rule-based exclusion (not a size cutoff) removes 100 entities /
  30,023 rows (~5.3% of the 569,877-row legitimate corpus) confirmed as
  diffuse-gateway traffic via empirical drilldown.
  AMBIGUOUS entities (398 / 16,645 rows, ~2.9%) are RETAINED based on
  ASSUMPTION-tier structural reasoning (small-N device-usage profile
  resembling shared-but-coherent contexts), not the same empirical bar
  applied to CONFIRMED_DIFFUSE/CONFIRMED_STABLE. Flagged for revisit if
  Stage 4.5F model behavior suggests this population is problematic.
Required for Stage 4.5F: dataset construction MUST apply the exclusion
  rule above before building the supervised sequence dataset.
Prerequisite(s): IEEE-CIS Stage 4.5E
Dependency role: DOWNSTREAM
Next stage: IEEE-CIS Stage 4.5F
Next stage allowed: YES, conditional on exclusion rule being implemented

---

### IEEE-CIS Stage 4.5F
Status: COMPLETED
Completed on: 2026-08-27
Commit: 53e5cd1
Scope: Rebuild supervised dataset using the approved proxy/objective and retrain the corrected baseline
Evidence: reports/stage_4_5F_results.md; models/normal_behavior/logistic_regression_v2/
Result: Dataset rebuilt with Stage 4.5E Addendum exclusion rule applied
  (100 entities / 30,023 rows excluded; 539,854 rows retained). Manifest
  leakage from v1 caught and fixed (imputation stats regenerated from v2
  train split only). Corrected model: Macro F1 = 0.6790, Balanced
  Accuracy = 0.6372. Majority-class baseline for v2: 0.1810.
  NAIVE DOMAIN BASELINE ("predict previous_ProductCD"): Macro F1 = 0.7508
  — outperforms the trained model by ~0.07 F1 on identical test split.
Integration decision: DO NOT INTEGRATE. Per standing ML strategy (item 8:
  "ML performance alone must not justify a modeling decision"), the
  corrected model fails to beat a one-line domain heuristic on its own
  training/eval population, and its applicability to the Normal World's
  synthetic personas (a distinct population from IEEE-CIS proxy entities)
  was never validated. The Normal World should NOT adopt this ML
  pipeline for next-transaction-category prediction.
Prerequisite(s): IEEE-CIS Stage 4.5E Addendum
Dependency role: DOWNSTREAM
Next stage: TBD — the originally proposed persistence-rule transfer
  does not map onto the current schema (see
  reports/project_audit_2026-08-27.md, finding 1). Requires reframing
  around an actual Normal World field (e.g. transaction_type or
  merchant.mcc_code persistence) before a concrete next stage can be
  defined.
Next stage allowed: YES — scoped to the Normal World persistence-rule
  evaluation above. Blue Team / LLM planner stages remain NOT unblocked
  by this closeout and require their own separate readiness review.
