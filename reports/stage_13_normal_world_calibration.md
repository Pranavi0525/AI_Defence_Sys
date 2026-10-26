NORMAL WORLD CALIBRATION
------------------------
Customers: 100
Events: 10000
Simulation period: 2025-01-01 01:48:57 to 2025-02-13 22:15:37
Seed: 42

Marginal:
- Amount Distribution: Mean=1738.39, Std=1418.04 (Status: NOT_AVAILABLE - No reference data)
- Transaction Types: {'purchase': 0.766641045349731, 'transfer': 0.233358954650269} (Status: NOT_AVAILABLE - No reference data)

Dependency:
- Amount vs TxType (Status: NOT_AVAILABLE)

Temporal:
- Inter-event timing: Mean 50988.72s (Status: NOT_AVAILABLE)

Behavioral:
- Transactions/customer: 65.71
- Behavioral persistence: 67.62% (Stateless coin-flip behavior detected)

Graph:
- Devices/customer: 1.23
- Beneficiaries/customer: 1.60
- Avg Graph Degree: 9.46 (Status: NOT_AVAILABLE)

Structural:
- Schema Validity: PASS (100% compliant)

PASS:
- Structural Validation

FAIL:
- None (all distance metrics NOT_AVAILABLE)

NOT_AVAILABLE:
- Marginal distance
- Dependency distance
- Temporal distance
- Behavioral distance
- Graph distance

Major weaknesses:
1. Transaction-type distribution is purely stateless and hardcoded (approx 80% purchase, 20% transfer).
2. Customer behavioral persistence is basically random (matching the global coin-flip probability rather than showing individual habits).
3. Amount distribution lacks complex multimodal behavior (driven by simple persona parameters).
4. Beneficiary and Device relationships are extremely flat/simple (almost strictly 1-to-1 or randomly assigned uniformly).
5. Inter-event timing is stateless (drawn from exponential distribution, no burst/sleep cycles).

Tests:
- Deterministic generation: PASS
- Legitimate-only generation: PASS
- Calibration execution: PASS
- NOT_AVAILABLE handling: PASS
- No fabricated reference values: PASS
- Provenance preservation: PASS

Recommendation:
NEEDS BEHAVIORAL IMPROVEMENT
