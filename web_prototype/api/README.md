# Backend API

Wires the existing Red Team / Blue Team pipeline in this repo to a web
frontend. Nothing is hardcoded — every `/api/reports/...` call reads the
real output files off disk at request time, and `/api/pipeline/run`
kicks off the real stage scripts as subprocesses.

## Setup

```powershell
pip install -r requirements.txt
```

(`fastapi` and `uvicorn` are already in `requirements.txt`.)

## Run

```bash
python3 web_prototype/run_api.py --reload
```

This works from **any** directory — the repo root, `web_prototype/`,
or elsewhere — because `run_api.py` resolves every path from its own
file location instead of relying on the current working directory.

Then open **http://localhost:8000/docs** — that's an interactive page
where you can try every endpoint by hand, useful for the demo itself if
you want to show the API directly.

### Superseded: the old documented command

`PYTHONPATH=src python -m uvicorn api.app:app --reload --port 8000`,
run from the repo root, **does not work** — `api` is a package under
`web_prototype/`, not a top-level package at the repo root, so
`import api.app` raises `ModuleNotFoundError`. It only ever worked by
accident, when the process's cwd already happened to be
`web_prototype/` (`cd web_prototype && uvicorn api.app:app ...`). This
was confirmed by reproduction, not assumed — see
`reports/FINAL_VALIDATION_REPORT.md`, section 5, which flagged the same
discrepancy in an earlier pass but deliberately left it unfixed pending
this launcher. `run_api.py` removes the cwd dependency; if you still
want to invoke uvicorn directly, `cd web_prototype && uvicorn
api.app:app --reload --port 8000` remains a valid workaround.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | liveness check |
| GET | `/api/reports/dashboard` | one call, everything the dashboard needs, all live |
| GET | `/api/reports/{key}` | one report by key (see `config.REPORTS` for the full list) |
| GET | `/api/reports/{key}/file` | serves a report as a raw file (used for the SHAP PNG) |
| GET | `/api/pipeline/stages` | the 10-stage catalog (id, label, script, est. runtime) |
| POST | `/api/pipeline/run` | body `{"stage_ids": [...]}` (omit for all 10, in order) — returns `job_id` immediately |
| GET | `/api/pipeline/status/{job_id}` | per-stage status/log tail, poll this while a run is in progress |
| GET | `/api/pipeline/jobs` | recent job history |

## Report keys

`stage1_2_results`, `stage3_graph_results`, `stage4_autoencoder_results`,
`risk_fusion_results`, `decision_policy_results`,
`decision_policy_sensitivity_results`, `case_reports`,
`global_feature_importance`, `global_shap_summary_png`, `misses`,
`adaptive_round2_report`, `adaptive_eval_holdout`,
`round1_vs_round2_report`, `hard_examples`,
`hard_example_generation_report`.

Every canonical path is documented with *why* it's canonical (vs. a
stale root-level copy) at the top of `config.py` — worth reading once.

## Demo-day guidance

- **Full pipeline run ≈ 10–12 minutes** (Stage 3 and Stage 4 each retrain
  fresh models per CV fold — that's the honest cost of cross-validated
  numbers, not a bug). Don't click "run full pipeline" live in front of
  judges and wait — pre-run it beforehand, then use the dashboard's live
  reads to show the *real, current* state, and demo the run button on a
  single fast stage (`decision_policy_sensitivity` finishes in ~2s) to
  prove it's connected, not staged.
- No LLM / API key is required anywhere in this loop — verified: nothing
  in the pipeline scripts imports `genai` or reads `GOOGLE_API_KEY`
  outside the test suite. The whole closed loop runs offline.
- If a stage fails mid-run, `/api/pipeline/status/{job_id}` shows exactly
  which one and the last 40 lines of its stdout/stderr — that's your
  debug trail, not a black box.

## Known limitation (say this if asked, don't hide it)

Job state lives in an in-memory dict — restart the server and history is
gone. Fine for a single-demo hackathon prototype; would need Redis/a DB
for anything persistent.
