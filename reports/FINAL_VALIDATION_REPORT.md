# AI Defense Lab — Audit Pass Report (2026-08-31)

This is an honest status report on **one engineering pass** over the repository.
It is deliberately scoped, not a claim that the full 15-step brief in the
original task was completed end-to-end — see "What was NOT done" at the
bottom before reading anything else as a finished deliverable.

## Environment this pass ran in

- Python 3.12.3, scikit-learn 1.8.0, xgboost 3.4.1, torch 2.13.0 (Linux
  sandbox — **not** the user's Windows/Python 3.14 box named in the brief).
  The fixes below are written to be OS-agnostic, but they have only been
  smoke-tested on Linux, not verified on Windows/Python 3.14.

## 1. KNOWN TECHNICAL PROBLEM 1 — `CalibratedClassifierCV(cv="prefit")`

**Status: already fixed in this checkout, verified working.**

`blue_team_pipeline.py` (line ~384) already uses
`CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")`,
which is the correct modern-sklearn replacement for `cv="prefit"` and
preserves the original semantics (calibrate a frozen, already-fit
estimator rather than refitting inside CV folds). No change was needed
here; I confirmed it imports and fits cleanly under sklearn 1.8.0.

## 2. KNOWN TECHNICAL PROBLEM 2 — `ModuleNotFoundError: No module named 'red_team'`

**Status: reproduced, fixed, verified.**

Confirmed the bug is real: running `python3 hard_example_generator.py`
from the repo root with no `PYTHONPATH` set fails inside
`build_seeded_world()` because `src/red_team` is never added to
`sys.path` (pytest gets this for free from `pythonpath = ["src"]` in
`pyproject.toml`; a plain script invocation does not).

**Fix applied** (`hard_example_generator.py`, top of file): before any
other imports, try `import red_team`; if that fails, insert `src`
(resolved from `Path(__file__).resolve().parent / "src"`) at the front
of `sys.path`. This is idempotent, does not fight an explicit
`PYTHONPATH` the user has already set (it only acts when the import
actually fails), and doesn't create a second, differently-pathed copy of
the package.

**Verified**: `python3 hard_example_generator.py` now runs end-to-end
from a clean shell with no `PYTHONPATH`, producing
`blue_team_output/hard_examples.jsonl` (42 accepted) and
`hard_example_generation_report.json`.

## 3. KNOWN TECHNICAL PROBLEM 3 — hard-example hardness validation

**Status: already implemented, verified by running it.**

The generator already differentiates Stage-1 misses (validated by
Stage-1 escalation rate on the candidates vs. the family baseline) from
Stage-2/model misses (validated by comparing XGBoost/fused probability
against the real 0.5 decision threshold and the family baseline). The
run above reported, per miss-family/failure-stage bucket: candidates
generated, rejected-by-simulator-realism, rejected-as-not-harder, and
accepted, with a `genuinely_harder` boolean and a text justification for
each bucket (e.g. ACCOUNT_TAKEOVER stage-2 misses: 5/5 accepted, all
below the 0.5 threshold with mean proba 0.228 vs. 0.986 family baseline).
I did not need to change this logic — I only unblocked it (see #2).

## 4. Test suite

Before this pass: **467 passed, 1 failed** —
`tests/test_classification.py::test_classify_entity` failed with
`ModuleNotFoundError: No module named 'scratch'`. This test imports
`scratch.device_drilldown_101_1000.classify_entity`, a module that does
not exist anywhere in the delivered repository and is not referenced by
any pipeline stage, API route, or other test — it looks like a leftover
from an exploratory session whose source file was never committed.

I reconstructed `scratch/device_drilldown_101_1000.py` **strictly from
the boundary conditions the test itself already specifies** (four bands:
`CONFIRMED_DIFFUSE`, `CONFIRMED_STABLE`, `INSUFFICIENT_EVIDENCE`,
`AMBIGUOUS`, with explicit thresholds in the test's comments), rather
than inventing behavior. It is not wired into the fraud-detection
pipeline anywhere.

**After this pass: 468 passed, 0 failed.**

## 5. FastAPI backend — real discrepancy with the task brief

The brief describes the backend as `backend_api/api/app.py`, launched
from the repo root with
`$env:PYTHONPATH="$PWD\src;$PWD\backend_api"`. **That path does not exist
in this repository.** The actual backend lives at
`web_prototype/api/app.py` (module `api.app`), and its own README says to
run `PYTHONPATH=src python -m uvicorn api.app:app --reload --port 8000`
from the repo root — but that's also not sufficient on its own, because
`api` isn't importable as a top-level package from the repo root either.

What I actually verified works:

```bash
cd web_prototype
python -m uvicorn api.app:app --reload --port 8000
```

Run this way, all three endpoints the brief asks about returned 200:
`GET /api/health`, `GET /api/reports/dashboard`, `GET /docs`. I did not
rename/move the API or "fix" the README, since I'm not certain which of
(a) the brief, (b) the README, or (c) the directory layout is the stale
one — flagging it rather than guessing felt like the more honest move.

**Addendum (later pass):** this was fixed rather than left as a
workaround. `web_prototype/run_api.py` is a cwd-independent launcher
that resolves `api/`'s location from its own file path instead of
relying on the process's working directory, so `python3
web_prototype/run_api.py` now works identically from the repo root,
from `web_prototype/`, or from anywhere else. `api/README.md` and
`api/app.py`'s docstring were updated to document this as the primary
command, with the old `cd web_prototype && uvicorn api.app:app ...`
kept as a documented (working) fallback. Verified by simulating the
import-resolution logic against stub `app`/`uvicorn` modules from an
unrelated working directory (this sandbox has no network access, so
the real `fastapi`/`uvicorn` packages could not be installed to run it
end-to-end) — see the launcher's own docstring for the exact commands
to re-verify once dependencies are available.
I did not touch the web dashboard at all this pass.

## 6. Decision policy numbers — spot-checked, match the brief

`decision_policy_results.json`'s `"corrected"` block matches the brief's
"CURRENT VERIFIED RESULTS" section exactly: `t_review=0.1479`,
`t_block=0.9196`, allow/review/block = 77.91%/5.14%/16.94%,
`legit_blocked=1`, `fraud_blocked=246`, `fraud_reviewed=26`,
`fraud_allowed=5`, recall (block+review) = 98.19%. This artifact is live
and consistent with the brief.

## 7. Round 1 vs Round 2 — RESOLVED (regenerated 2026-09-04)

The staleness described below was real, but it was a symptom of two
actual bugs in `retrain_round2.py`, not just a report generated at a
different time. Root cause, found while regenerating:

1. **`retrain_round2.py` never loaded MULE_NETWORK traces at all** —
   `build_round_datasets()` only ever assembled ATO + APP + legit, so
   every prior report (including the 96.08%-recall run the brief cites,
   and the 91.18%-recall run this repo had on disk before today) was
   scored against the wrong, incomplete population. `MULE_NETWORK` now
   loads from `reports/mule_corpus_raw.json` and is included, unchanged,
   in both rounds (it has no hard examples yet — see `meta` in the
   output — so it behaves like the legit population: present, but not
   what differs between Round 1 and Round 2).
2. **`HESITATION_DELTA` was silently missing from the feature table.**
   `to_feature_df()` called `extract_features()` per-record but never
   called `add_hesitation_delta()`, the batch-level step that actually
   computes it in the main pipeline (`build_dataset()`). Every prior
   Round 1/Round 2 comparison was therefore trained on 26 features, not
   the 27 `FEATURE_COLS` the live model uses. Now fixed to call
   `add_hesitation_delta()` on both rounds, identically to
   `build_dataset()`.

With both fixed, plus `stable_fold_id()` (already wired into
`cross_validated_evaluate` / `EvaluationHarness`, so this comparison
picks it up for free), the phantom regression is gone: cascade ATO
recall goes from 97.94% (Round 1) to 99.02% (Round 2), and overall
cascade recall moves +0.31pp, not -6.45pp. `meta.round1_n` is now 1530
(97 ATO + 156 APP + 121 MULE_NETWORK + 1156 legit) and `round2_n` is
1535. Regenerated via `PYTHONPATH=src python3 retrain_round2.py`;
468/468 tests still pass.

## What was NOT done in this pass (explicitly, so it isn't mistaken for "done")

This was a scoped pass covering the two concretely-specified bugs
(Problems 1–2), a spot-check of Problem 3, the test suite, and the API.
The following parts of the original brief are substantial, standalone
pieces of work and were **not** attempted here:

- **Problem 4** (root-causing the ATO Round-2 recall regression: checking
  overfitting, class balance, near-duplicates, distribution shift,
  weighting, threshold/calibration drift, leakage) — not investigated.
  Given finding #7 above, this should start with regenerating Round
  1/Round 2 fresh rather than trusting the current artifact.
- **Non-regression gates** for Round 2 promotion — not implemented.
  `retrain_round2.py` currently has no gate logic at all.
- Full **risk_fusion.py / decision_policy.py leakage audit** — not done
  beyond the one spot-check above.
- **Miss collector / decision policy / explainability consistency
  check** — not built.
- **Artifact metadata** (seed, dataset hash, feature-schema hash,
  package versions) — not added.
- Full **from-scratch 15-step pipeline re-run** — not done; this pass
  reused existing artifacts and only re-ran the two scripts directly
  implicated in Problems 2–3.
- Windows/Python 3.14 verification — everything above was only run on
  Linux/Python 3.12.

## Files changed this pass

1. `hard_example_generator.py` — added the `src`-on-`sys.path` fallback
   (Problem 2 fix).
2. `scratch/__init__.py`, `scratch/device_drilldown_101_1000.py` — new,
   restores the orphaned test to passing.
3. `FINAL_VALIDATION_REPORT.md` — this file.

## Commands to reproduce what's reported above

```bash
pip install -r requirements.txt xgboost torch
python3 -m pytest -q                    # 468 passed
python3 hard_example_generator.py       # no PYTHONPATH needed now
cd web_prototype && python -m uvicorn api.app:app --reload --port 8000
```
