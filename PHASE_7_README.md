# Phase 7 -- case dossiers and an analyst feedback loop (v2)

Everything here is SYNTHETIC-data and additive: no v1 file changed; `web_prototype/api/v2_routes.py` gained one endpoint.
Result numbers live only in generated reports (`reports/v2/feedback_loop.md`, `reports/v2/dossiers.md`); none are typed in this file.

## 7a. Case dossiers (`v2/dossier.py`, `POST /api/v2/dossier`, `scripts/make_v2_dossiers.py`)
For every REVIEW / BLOCK an analyst gets one page with six labelled parts: decision and margins, model view (XGBoost + two novelty
percentiles), model drivers (XGBoost contributions in plain words), evidence (fixed readable checks on the raw trace), graph evidence
(earlier traces only), and acting party. Each part says whether it is a model output, a rule-of-thumb flag, or a policy mapping.

Design decisions you must be able to defend:
- **Family is unknown at scoring time.** The acting-party block stays empty until an analyst supplies a family. Guessing it would be overclaiming.
- **Liability:** the three v1 families reuse the v1 mapping (a test fails if it ever drifts from `decision_policy.py`); the four new families are
  my ASSUMPTIONS and are labelled `ASSUMED` in every dossier.
- **Evidence flags are not model outputs** and their cut-offs (1 hour, 90 days, 60 days, liveness 0.6) were not tuned on data.
- **Read-only:** building a dossier never writes to the entity store (tested).
- A MISSING feature value raising risk is flagged in the dossier as a likely simulator artifact.

Run: `python scripts/make_v2_dossiers.py` (cmd: `set PYTHONPATH=src` first). Picks are the first of each kind on a fresh world (seed 777), not cherry-picked.

## 7b. Analyst feedback loop (`v2/feedback.py`, `scripts/run_v2_feedback_loop.py`)
Round 0 model -> out-of-fold mistakes on the training part only (misses, false alarms) -> fresh look-alike cases from the generator ->
retrain -> compare on an untouched, customer-grouped test quarter. A **control arm** adds the same number of fresh fraud/legit traces from
the ordinary mix, so `targeted - control` measures learning from mistakes and not just extra data. Repeated over 10 worlds; paired differences.

Honest limits: the fresh cases come from the same generator as the test set (an in-distribution best case); real analysts find only some
mistakes, late and noisily (the `found_share` setting only partly models that); nothing here helps with unseen fraud families.

Run (cmd), each about 2 minutes:
```
set PYTHONPATH=src
python scripts\run_v2_feedback_loop.py --reps 10 --per-case 2 --found-share 1.0
python scripts\run_v2_feedback_loop.py --reps 10 --per-case 2 --found-share 0.5
python scripts\run_v2_feedback_loop.py --reps 10 --per-case 4 --found-share 1.0
python scripts\run_v2_feedback_loop.py --report
python scripts\v2_feedback_miss_audit.py
python scripts\make_v2_dossiers.py
```

## Tests
`tests/test_v2_dossier.py`, `tests/test_v2_feedback.py` (root suite) and `web_prototype/api/tests/test_v2_dossier_api.py` (API suite).
The feedback test checks the loop's pipeline reproduces the SERVED bundle's thresholds, so before/after describes the model you deploy.
Run root and API tests in SEPARATE pytest calls.

## Housekeeping
`tests/v2/` in the merged zip is a stale pre-Phase-6 copy of the `v2/` code (its `features.py` still has the old transductive graph and its
own extractor). Nothing imports it. Delete it: `rmdir /s /q tests\v2`.
