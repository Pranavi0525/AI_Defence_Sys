# Phase 6: the full cascade is now served (`/api/v2/*`)

The v1 endpoint `POST /api/score` (Stage 1+2) is untouched. The new endpoints serve the whole v2 cascade:
XGBoost + Isolation Forest + autoencoder -> stacked probability -> cost-based ALLOW / REVIEW / BLOCK, with history-only graph
features, per-feature explanations, and drift monitoring.

## What each endpoint does
| Endpoint | Purpose |
|---|---|
| `POST /api/v2/score` | Scores one trace. Returns the decision, the stacked risk score, every stage's score, the thresholds, the top contributing features, and latency |
| `GET /api/v2/status` | Ready flag, model version, entity-store size, whether `commit` is enabled |
| `GET /api/v2/model-card` | Features, thresholds, cost assumptions, cross-validated evaluation summary, limitations |
| `GET /api/v2/drift` | PSI of recently scored traces vs the training distribution (needs 50+ traces; stable <0.10, watch 0.10-0.25, drift >0.25) |
| `GET /readyz` | Now also reports `v2_ready` (v1 readiness unchanged; a v2 load failure never blocks v1) |

## Request / response (see `web_prototype/examples/*.json`)
Request: `{"trace": {"trace_id", "customer_id", "events": [...], "observation_window": [start, end] (optional)}, "commit": false}`.
Events use the same JSON shape as the Red Team corpus, plus optional v2 events `ACCOUNT_OPENING` and `KYC_VERIFICATION` and
optional fields `account_age_days` (login), `merchant_id`/`merchant_age_days` (transaction).
Validation: known event types only, ISO timestamps, positive amounts, at most 200 events (otherwise HTTP 422).

Real outputs from the three example files (generated in my environment, ~3-5 ms each):
| Example | Decision | Risk | Notes |
|---|---|---|---|
| routine legit | ALLOW | 0.017 | |
| deepfake-KYC style | BLOCK | 0.994 | both novelty percentiles very high |
| AI-scripted APP | ALLOW | 0.736 | just under the review threshold (0.758): a realistic near-miss |

A feature shown with `value: null` is MISSING in that trace. The model learned that "no merchant purchase" pushes risk up, which is a
property of my simulator (my fraud families mostly use transfers). On real data that would need a rethink. Be ready to say so.

## How training and serving stay identical (the part to remember for interviews)
1. ONE feature extractor (`v2/serving/core_features.py`) is used by training and by the API. A test compares it with your v1 extractor on real traces.
2. Graph features are HISTORY-ONLY: each trace sees only traces before it, through the same `EntityStore` class in training (time-ordered replay) and serving.
   A test replays the training traces one by one through the scorer and reproduces the training graph features exactly (no train/serve skew).
   (Earlier batch graph features used the whole dataset, including future traces; moving to history-only removed the graph gain in World A.)
3. Isolation Forest, the quantile transformer and the autoencoder are exported as arrays and run in numpy (`numpy_models.py`), so the API does not depend on
   the scikit-learn version. A test checks they match scikit-learn to 1e-9. Only `numpy`, `scipy` and `xgboost` are needed to score.
4. The stacker and the thresholds are learned on out-of-fold scores only (same nested procedure as the evaluation). A test checks the served scores equal the training-side scores.

## Run it locally
```
python -m venv .venv-eval ; (activate) ; pip install -r requirements-eval.lock.txt ; pip install fastapi uvicorn httpx
$env:PYTHONPATH="src"                                   # PowerShell
python scripts/run_v2_all.py                            # optional: regenerate evaluation (stages: see --help)
python scripts/build_v2_bundle.py                       # trains World B model -> models/v2/ (already included in this package)
python web_prototype/run_api.py --host 127.0.0.1 --port 8000
```
PowerShell call:
```
$body = Get-Content web_prototype\examples\v2_sample_deepfake_kyc.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v2/score -Method Post -ContentType "application/json" -Body $body
```
Env vars: `V2_MODEL_DIR` (default `models/v2`), `V2_ALLOW_COMMIT` (default false: `commit=true` returns 403 so a public demo cannot pollute the live entity store).

## Deploying on Render
No new dependencies (numpy, scipy, xgboost are already installed by `scripts/install.sh`). Commit `models/v2/` (about 2 MB) with the code.
The free plan sleeps when idle, so open `/api/v2/status` before a demo. The v2 model stays in memory; the entity store resets on restart (it reloads from `models/v2/entity_store.json`).

## Edits to existing files (`web_prototype/api/app.py`; already applied in this package)
1. add the repo root to `sys.path`; 2. `import v2_routes`; 3. `v2_routes.registry.load()` in the lifespan; 4. `app.include_router(v2_routes.router)`; 5. `"v2_ready"` in `/readyz`.

## Tests
`tests/test_v2_serving.py` (7): feature equality with v1, history-only graph, store JSON round-trip, drift, numpy==sklearn, served==training scores, no train/serve skew.
`web_prototype/api/tests/test_v2_api.py` (12): status, model card, scoring, validation errors, commit guard, drift.
Run the root tests and the API tests in SEPARATE pytest calls (both folders are named `tests`):
```
pytest tests/test_eval_v2_splits.py tests/test_v2_suite.py tests/test_v2_serving.py -v     # expect 23 passed
pytest web_prototype/api/tests -v                                                          # expect 24 passed (12 are the new v2 tests)
```

## Limits of what is served
- Trained on synthetic World B; the score is not a real-world fraud probability, and the API says so in every response.
- No authentication or rate limiting; the entity store and drift buffer are in memory and per-process (one worker only).
- Stage 1 rules are not part of v2 (the cascade is the learned models plus the policy); the v1 endpoint still serves the v1 rules.
- Explanations come from XGBoost's native contributions (log-odds), not from the novelty models.
- Thresholds depend on assumed prevalence and cost constants, which are stored in the model card.
