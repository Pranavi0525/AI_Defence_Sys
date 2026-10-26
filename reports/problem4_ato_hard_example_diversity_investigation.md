# Problem 4 Follow-up: Near-Duplication / Diversity Check on the 5 ATO Hard Examples

**Scope:** This report investigates whether the 5 `ACCOUNT_TAKEOVER` (ATO) hard examples in
`blue_team_output/hard_examples.jsonl` are near-duplicates, per the concentration risk flagged
in the Round 1 → Round 2 regression diagnosis (2 source misses → 5 kept examples). No retraining
or model changes were made. This is a read-only data investigation, reproduced directly from the
repo's own `extract_features` / `FEATURE_COLS` used to train the Stage-2 XGBoost model.

## 1. Provenance: which examples come from which source miss

Recovered from each hard example's `generation_metadata.source_miss_trace_id`:

| hard example trace_id | source miss | seed | xgb_proba at generation time |
|---|---|---|---|
| `atk-aa6b9b1c` | `atk-693a3017` | 9000 | 0.153 |
| `atk-771e11d8` | `atk-693a3017` | 9000 | 0.111 |
| `atk-d35248f4` | `atk-1179705d` | 9001 | 0.323 |
| `atk-c08b09a8` | `atk-1179705d` | 9001 | 0.170 |
| `atk-f908a1b3` | `atk-1179705d` | 9001 | 0.384 |

Confirms the generation report's breakdown exactly: 2 examples from `atk-693a3017`, 3 from
`atk-1179705d` — all 5 ATO hard examples trace back to just these 2 original misses.

## 2. Full feature-vector comparison (the actual XGBoost input space)

Running every example through the real `extract_features()` (the same function used for
training) and comparing across all 26 `FEATURE_COLS`:

**25 of 26 features are bit-identical across all 5 examples**, regardless of which source miss
they came from:

```
count_transaction=0, count_session_login=1, count_device_registration=1,
count_beneficiary_addition=0, total_events=2, transactions_per_hour=0.0,
transactions_per_session=0.0, time_login_to_first_transaction=-1.0,
login_attempt_count_max=1, auth_failure_present=0, new_device_present=1,
time_device_registration_to_transaction=-1.0, beneficiary_added_before_transaction=0,
time_from_beneficiary_add_to_transaction=-1.0, failed_transaction_count=0,
failed_then_completed=0, amount_mean=amount_max=amount_min=amount_std=amount_cv=0.0,
amount_change_after_failure=0, amount_trend=0.0, distinct_channels=0
```

The **only** feature that varies is `window_seconds` (= `mean_time_between_transactions` =
`min_time_between_transactions`, since there are only 2 events): the gap between session login
and device registration.

| trace_id | window_seconds (login → device reg) |
|---|---|
| `atk-aa6b9b1c` | 1,020 |
| `atk-771e11d8` | 1,080 |
| `atk-d35248f4` | 139,380 |
| `atk-c08b09a8` | 95,520 |
| `atk-f908a1b3` | 117,780 |

**Finding: in the feature space the model actually sees, these 5 "hard examples" are not 5
independent attack variations — they are 5 samples along a single scalar axis (one timing
gap), with every other dimension pinned to the same constant vector.** The `xgb_proba` spread
in the table above (0.11–0.38) is consistent with the model having essentially learned a
threshold on this one number.

## 3. Is this specific to the hard-example generator, or inherited from the simulator?

Checked against the full 97-trace baseline ATO corpus (`reports/ato_corpus_raw.json`), not just
the 5 hard examples. Both original source misses (`atk-693a3017`, `atk-1179705d`) are themselves
part of a recognizable subgroup: traces consisting of **only** `SESSION_LOGIN` +
`DEVICE_REGISTRATION`, with no `TRANSACTION` event ever appearing in the observation window
(despite the `extract_funds` hidden objective — the transaction presumably falls outside the
window the detector sees).

- 21/97 (22%) of the baseline ATO corpus has `count_transaction == 0`.
- 13/97 have exactly 2 total events (login + device registration only).
- **12 of those 13** already collapse to the identical constant feature vector described above —
  this is a pre-existing structural template in the Red Team simulator's ATO output, not
  something the hard-example generator introduced. Both source misses were simply two more
  instances of it, at the more unusual (larger) end of the `window_seconds` range (420s and
  114,840s respectively), which is presumably why they evaded Stage 2 in the first place.

So the low diversity isn't a defect in `hard_example_generator.py` — it's a real, pre-existing
characteristic of this attack template: **for this entire minimal-trace ATO subclass, the model
is functionally a 1-D classifier on a single timing feature**, and the hard-example generator
faithfully reproduced that reality via honest rejection sampling (per its own documented
methodology — no candidates were rejected by the simulator for either source miss, meaning the
pool of 100 generated candidates per miss was itself low-diversity at the feature level).

## 4. Answering the specific question: genuinely different attack variations?

**No.** As raw event logs they differ (different customer IDs, device UUIDs, timestamps), so
they are not byte-identical duplicates. But as inputs to the Stage-2 model — which is what
matters for training/evaluation — they are **near-duplicates concentrated on one feature axis**,
drawn from only 2 underlying misses. This is a genuine data-diversity limitation, consistent
with the concentration risk already flagged, and now confirmed and quantified rather than
just suspected.

## 5. Recommendation

Per instruction, no examples were removed and no retraining was performed. Document as a known
limitation:

- The 5 ATO hard examples give the CV process 5 data points but effectively only 2 independent
  "attack scenarios," each expressed through a single scalar feature.
- This does not invalidate the Problem 4 regression diagnosis (0/97 original ATO misses under
  controlled same-fold evaluation still holds) — it explains *why* the hard examples behave so
  uniformly under CV (near-identical inputs land in the same fold and score similarly), and
  tempers any claim that Round 2 was evaluated against "5 diverse new attack patterns."
- If future work expands the ATO hard-example set, the simulator's lack of a working difficulty-
  steering mechanism (already noted in `hard_example_generator.py`'s own docstring) means new
  source misses — not more resamples from these same 2 — are the only way to add real diversity
  to this subclass.
