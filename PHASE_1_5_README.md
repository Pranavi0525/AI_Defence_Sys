# v2 package: Phases 1-5 (+ lightweight Phase 3 and 4)

## Files (paths relative to repo root; copy as-is; v1 files are not modified)
| File | What it is |
|---|---|
| `v2/gen.py` | Generators: hard-negative legit + 4 GenAI-era fraud families (synthetic identity, fake merchant, deepfake-KYC telemetry, AI-scripted APP) |
| `v2/features.py` | Feature extraction (fixed failure definition), graph features, pooled dataset builder |
| `v2/evalkit.py` | XGBoost factory, novelty detectors, metrics, calibration, customer-cluster bootstrap |
| `v2/phases.py` | Phase 1 (in-fold Stage 1, calibration, bootstrap), Phase 2 (LOFO + transfer), Phase 4 (graph ablation) |
| `v2/fusion.py` | Phase 5: nested stacking + cost-aware thresholds + unseen-family policy test |
| `v2/audit.py` | Artifact audits |
| `scripts/run_v2_all.py` | Runs everything, writes `reports/v2/v2_report.md` and `v2_results.json` |
| `tests/test_v2_suite.py` | 11 tests (generators, no id-format leak, failure definition, graph degree, weights, thresholds, cost) |
| `eval_v2/`, `scripts/v2_leakage_check.py`, `tests/test_eval_v2_splits.py` | Phase 0/1.1 from the earlier package |
| `reports/v2/*` | Results generated in my environment: REGENERATE on your machine |

## Run
```
python -m venv .venv-eval ; (activate) ; pip install -r requirements-eval.lock.txt
PowerShell:  $env:PYTHONPATH="src"
pytest tests/test_eval_v2_splits.py tests/test_v2_suite.py -v        # expect 23 passed (with tests/test_v2_serving.py)
python scripts/run_v2_all.py                                          # ~6-10 min; or one stage at a time:
python scripts/run_v2_all.py --stage data ; --stage p1 ; --stage p2p4 ; --stage p5a ; --stage p5b ; --stage p5c ; --stage report
```

## Two worlds (important)
- **World A**: v1 Red Team attacks vs Normal World legit (1,556 traces, 374 fraud).
- **World B**: my v2 generators: hard-negative legit vs 4 GenAI-era families (1,260 traces, 360 fraud).
They are evaluated SEPARATELY. Pooling them let the model learn "which simulator made this trace" (new-schema fields are
empty for every v1 trace), which inflated v1-family recall (MULE 0.85 -> 0.96 for no fraud-related reason).

## What changed vs my earlier statements (corrections)
1. Normal World legit traces DO contain beneficiary additions (12% of traces) and device registrations (5%). The v1
   docstring saying they never occur is out of date. So "beneficiary-add is a perfect separator" is no longer true.
2. The v1 failure feature counted status `pending` as failed. Every legit transaction is `pending`, so for legit traces
   `failed_transaction_count` equalled the transaction count (correlation 1.0). Fixed in v2. Effect on F1 is small (0.981 -> 0.978).
3. The "0 leakage findings" check covers ground-truth fields. The RAW observable fields still contain perfect proxies
   (legit `fingerprint`/`channel`/`transaction_status` values never occur in attacks; see the audit table in the report).
   v1 features do not read those fields, so v1 scores are not driven by them, but any new feature built on them would be.
4. Customer overlap between train/test folds is real (88%) but its effect on F1 is within noise (0.980 vs 0.981).

## Update: making World B genuinely hard (next-parts step 1)
My first World B was trivially separable (F1 ~0.999). I audited my OWN generators and found five giveaways, then fixed each
by giving the legit side an equivalent hard negative or by letting a share of attackers adapt ("stealth"):
1. `kyc_present`: only 6% of legit had KYC, so "has a KYC event" meant fraud. -> legit onboarding is now 20% of legit.
2. Scam payee reuse (AUC 1.00): scam payees were shared far more than any legit payee. -> bigger scam-payee pool; stealthy attackers use a fresh payee per victim.
3. `g_component_size`: merchants glued all shoppers into one giant component, so customers who never shop looked "isolated". -> merchant hubs excluded from connectivity (standard hub handling).
4. "No merchant = fraud": all legit routine activity was merchant purchases. -> ~30% of routine legit transactions are person-to-person transfers (v1 legit is ~25%). Added a legit "p2p spree" hard negative.
5. Synthetic-identity account opening appeared in the same trace as the bust-out. -> stealthy (aged) identities open the account earlier, outside the trace.
New knob: `stealth` = share of attackers that adapt. `scripts/v2_stealth_sweep.py` reports F1 vs stealth.
Default evaluation world: stealth 0.5, onboarding 20%, young legit accounts 12% (`WORLD_B_HARD` in v2/features.py).

## Current headline numbers (all synthetic; history-only graph features; regenerate on your machine; see v2_report.md)
- World A (v1 attacks): grouped-CV F1 0.975 +/- 0.003; recall 0.973 [0.957, 0.987]; precision at 0.6% prevalence 0.40 [0.27, 0.58].
- World B (hard): F1 0.824 +/- 0.012; recall 0.78 [0.74, 0.82]; precision at 0.6% prevalence 0.103 [0.077, 0.141].
  Per-family recall: SYNTHETIC_IDENTITY 0.86, FAKE_MERCHANT 0.49, DEEPFAKE_KYC 0.89, AI_SCRIPTED_APP 0.83.
- Stealth sweep: F1 0.986 (no stealth) -> 0.824 (50%) -> 0.774 (100%).
- Graph features (HISTORY-ONLY): no gain in World A (0.975 vs 0.975); small gain in World B (0.810 -> 0.824; FP 41 -> 35, FN 87 -> 83).
  An earlier batch version used future traces and looked better in World A: that was leakage, now removed.
- Calibration: raw XGBoost has the best Brier score in both worlds (A 0.0119 vs 0.0145/0.0144; B 0.0685 vs 0.0865/0.0824 for sigmoid/isotonic).
- Policy (World B, in-distribution): expected cost per 10k transactions: allow-everything 30,000; XGBoost-only 12,186; stacked 11,901; naive max 24,268.
- Novelty-aware stacking (leave-family-out inner folds): NOT worth it. Mean unseen-family cost World B 27,955 vs 21,627 (XGBoost-only); World A 4,470 vs 4,637 (negligible).
- Unseen families (fixed threshold 0.5, ~5% legit flagged by the novelty detectors): World B supervised recall 0-17%; novelty detectors best on
  DEEPFAKE_KYC (41%) and AI_SCRIPTED_APP (28-38%); FAKE_MERCHANT ~0%. World A: MULE_NETWORK supervised 59%, novelty detectors 2-14%.
- Where it errs: legit "new payee then pay" flagged 28% of the time, legit person-to-person sprees 8%.

## Honest limits of this package
- World B generators are lightweight and assumption-driven (stealth level, onboarding share, distributions): NOT validated by the Red Team
  validators and NOT estimated from real data. The stealth sweep shows how conclusions move with the assumption; none is a real-world claim.
- Hard negatives and stealth were designed after seeing which features separated my families. That is deliberate adversarial hardening, but it
  means World B was shaped by this model's behaviour; real validation needs data I did not simulate (Phase 8).
- Canonical signals remain strong by design: "new payee then pay" (single-feature AUC ~0.96) is a real scam signal, so AI_SCRIPTED_APP stays easier than FAKE_MERCHANT.
- v1 and v2 legit are almost perfectly distinguishable (AUC ~1.0), so the A->B transfer test is confounded by domain shift.
- Cost constants (loss 500, review 12, friction 25, review catch 90%) are assumptions.
- Graph features are history-only (time-ordered replay), the same logic the API uses online.
- Not done: Phase 7 (v2 case dossiers/analyst loop), 8 (real-data check), 9 (final interview packaging); porting generators into the Red Team; LLM planner. Phase 6 is in PHASE_6_README.md.
