# Stage 32 -- MULE_NETWORK Corpus Freeze

Status: QUALIFIED for freeze
Generated: reports/mule_corpus_raw.json
Duration: 0.5s

## Generation parameters
- world_seed: 100, n_customers: 60
- master_seed: 44
- target_traces: 120
- attempted: 42

## Results
- accepted traces: 121
- rejected attempts: 0
- distinct rings (network_id) recovered from flat storage: 42
- traces per ring (accepted / distinct rings): 2.88
  (NOTE: this ratio is expected to exceed 1 -- each accepted ring
  contributes 2-4 individual mule traces, unlike ATO/APP where one
  attempt == one trace. Not a correctness bug, see corpus.py docstring.)
- difficulty distribution: {'easy': 34, 'medium': 27, 'hard': 24, 'advanced': 36}
- invariant + leakage check: CLEAN

## Integration note
Every trace here is stored as an ordinary AttackRecord
({"observable_trace", "ground_truth"}), identical in shape to
ato_corpus_raw.json / app_corpus_raw.json. Blue Team's load_attack_corpus()
requires zero special-case code to consume this file -- rings remain
recoverable downstream via ground_truth.planner_metadata.plan_json.network_id
for anyone who wants ring-level (not just trace-level) analysis.
