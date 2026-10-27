# Phase 10 — Credit Card Fraud Detection module

A second, **independent** product inside the AI Defence System. The first is *Digital Payment Fraud Detection* (unchanged).
Both are reachable from one dashboard through a top-level switch.

> **Everything here is SYNTHETIC.** The served model is trained on the simulator in `v2/card/sim.py`. Its score is a *ranking
> score*, **not** a real-world fraud probability. No real transaction dataset has been used to train or validate it. It is a
> prototype and must not be used for real authorisation decisions.

## Architecture

```
v2/card/sim.py        synthetic simulator (existing)          ─┐
v2/card/features.py   27 causal features (existing, unchanged) │  Phase 10 evaluation framework (existing, unchanged)
v2/card/rules.py, evaluate.py                                  ─┘
v2/card/train.py      NEW  trains + exports the served bundle   (uses sim, features, evaluate.split / thr_at_fpr)
v2/card/serving.py    NEW  CardScorer: validate -> features -> booster -> threshold   (one pipeline for single + batch)
models/card/          NEW  xgb.json + policy.json  (own directory; no payment artifact is touched)
web_prototype/api/card_routes.py   NEW  /api/card/*  (own registry, own router)
web_prototype/dashboard/card.js, card.css   NEW  card UI + module switch
```

**Single and batch use the same code path.** `CardScorer.score_one(tx, history)` is `score_batch(history + [tx])` and returns the
last row, so the two cannot disagree (covered by tests).

**Independence.** `v2/card/*` imports nothing from the payment cascade (`v2.serving`, `v2.fusion`, `red_team`, …) and nothing in the
payment code imports the card package; the only link is a guarded mount in `web_prototype/api/app.py`. A missing/corrupt card
bundle, or even an import error in the card module, only disables `/api/card/*`. This is enforced by `tests/test_card_separation.py`
and `web_prototype/api/tests/test_card_api.py::TestIsolation`.

**No fallback model.** If `models/card/xgb.json` or `policy.json` is missing, unreadable, fails its SHA-256 check, or lists a feature
order different from `v2.card.features.FEATURES`, loading fails with an explicit message and the API answers `503` for card routes.

### Input fields (derived from the feature code, not invented)
`card, ts, amount, mcc, channel (POS|ECOM|ATM), lat, lon, merchant` and optional `declined` (0/1; only *prior* declines are used as
features — the row's own flag is not). Features for a transaction use only that card's **earlier** rows in the same request, so
supply the card's history. Invalid or missing values reject the whole request with row/field messages; nothing is imputed or dropped.

### Score meaning
`model_score` is the raw XGBoost output. `legit_percentile` is the share of *legitimate validation* transactions scoring lower.
`FRAUD` means `model_score >= threshold`, where the threshold is the score exceeded by ~1% of legitimate validation transactions
(the same operating-point rule as the Phase 10 evaluation).

## API (`/api/card`)
| Route | Purpose |
|---|---|
| `GET /status` | readiness, model version, threshold |
| `GET /model-card` | evaluation, limitations, input schema, feature list |
| `POST /predict` | `{transaction, history?}` → classification, score, percentile, warnings |
| `POST /predict-batch` | `{transactions:[…]}` **or** `{csv_text:"…"}` → per-row results + summary |

CSV is sent as text in a JSON body, so no `python-multipart` dependency was added. Payment routes are unchanged.

## Commands
```bash
# (re)train the served model — reproducible (seed 0); refuses to overwrite without --force
python scripts/train_card_model.py                      # writes models/card/{xgb.json,policy.json}

# run the API + dashboard locally
python web_prototype/run_api.py --port 8000
cd web_prototype/dashboard && python -m http.server 8080
# open http://localhost:8080/?module=card   (paste http://localhost:8000 into the API URL box if it is not preset)

# tests (run the two test trees as separate invocations: both are Python packages named `tests`, so one pytest run cannot mix them)
pytest tests/test_card_phase10.py tests/test_card_serving.py tests/test_card_separation.py   # card: unit + separation
pytest web_prototype/api/tests/test_card_api.py                                              # card: API + isolation from payment
pytest                                                                                       # tests/ (full, incl. all card tests above)
pytest web_prototype/api/tests                                                               # API suite incl. payment regression
pytest test_risk_fusion.py test_risk_fusion_leakage.py src/red_team/test_phase5a_evaluation_harness.py src/red_team/test_phase5b_robustness_evaluation.py
```

## Results (held-out **synthetic** test block, seed 0)
Model `card-xgb-synthetic-seed0-744050d8`; 34,239 test rows, 367 fraud rows in 70 simulated fraud events; event recall 100%.
These are measured on simulator data the model never saw, **not** on real transactions.

| Target FPR (validation legit) | Threshold | Recall | Precision | F1 | FP | FN |
|---|---|---|---|---|---|---|
| 5% | 0.00013 | 99.7% | 21.5% | 0.353 | 1338 | 1 |
| 2% | 0.00064 | 98.9% | 40.9% | 0.578 | 525 | 4 |
| **1% (served)** | **0.00194** | **98.4%** | **58.4%** | **0.733** | **257** | **6** |
| 0.5% | 0.00519 | 97.0% | 70.2% | 0.815 | 151 | 11 |
| 0.1% | 0.05463 | 91.6% | 92.6% | 0.921 | 27 | 31 |

Confusion matrix at the served threshold: TP 361 · FP 257 · FN 6 · TN 33,615. Precision here is at the simulator's test prevalence
(≈1.07%); `precision_at_assumed_0p6pct` (43.9%) uses an *assumed* 0.6% prevalence. Neither is a measured production precision.

The existing Phase 10 reports (`reports/v2/phase10_card*.md|json`) are unchanged. See their limitations: very high
leave-one-pattern-out scores may reflect signals shared across simulated patterns and do not demonstrate novel real-world fraud detection.

## Remaining limitations
- Synthetic data only; no real-data validation. Sparkov (`scripts/run_card_sparkov.py`) remains a separate, optional evaluation path
  (it needs the CSVs, which are not in the repo, and uses extra fields the served schema does not have).
- The card features need per-card history; the API is stateless, so the caller supplies it (no card-history store exists).
- The dashboard's example transactions are hand-written inputs; their results come from the model.


## Update: large-amount fix (branch `fix/card-fraud-large-amounts`)
Root cause and measurements: `reports/card/large_amount_investigation.md`; before/after: `reports/card/bundle_comparison.json`;
amount sweep: `reports/card/amount_sweep.md`. The tables above describe the ORIGINAL bundle; the served bundle is now trained on the
extended simulator (`policy.json` has the current metrics, PR-AUC, per-pattern and per-slice results).

**Amount boundaries and policies (INR).**
1. *Synthetic simulator cap* (`DEFAULT_SIM_MAX_AMOUNT` = 1,000,000.0 INR): The ceiling used during synthetic data generation (`v2/card/sim.py`).
2. *Actual training-data support* (`policy.amount_support.train_max_amount` = 465,290.34 INR): The maximum amount observed in training data. Tree models cannot extrapolate beyond learned splits; amounts above training support but within the serving limit are scored with out-of-training-range warnings and `score_reliable=False`.
3. *Serving input-validation limit* (`DEFAULT_MAX_TXN_AMOUNT` = 1,000,000.0 INR, env `CARD_MAX_TXN_AMOUNT`): The system input-validation boundary for incoming transaction-scoring requests. Requests strictly exceeding it are rejected with HTTP 422 before reaching the model. The serving limit is separate from the simulator cap and is not lowered to training support (e.g. ₹4 lakh).
4. *Issuer/business transaction policy*: Issuer-specific transaction rules (e.g., daily card caps, velocity limits). NOTE: Issuer-specific policy engines are NOT currently implemented in this module.

**History cleaning.** For `/predict`, history rows of another card, later than the transaction, exact duplicates, or identical to the
transaction are dropped; the response reports `history_rows_supplied/used` and `history_dropped`.
```powershell
python scripts/card_amount_sweep.py --old <dir with the old bundle>
pytest tests/test_card_amounts.py        # run together with the other tests/ card files
pytest web_prototype/api/tests/test_card_amount_api.py
```
