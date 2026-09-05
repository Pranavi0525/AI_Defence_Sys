"""Run with: python mule_smoke_test.py
(run this from the repo root, with PYTHONPATH set to .\\src, same as pytest)
"""
from red_team.world.world import NormalWorld
from red_team.attacks.mule_network_orchestrator import MuleNetworkOrchestrator
from red_team.attacks.corpus import generate_mule_network_corpus

w = NormalWorld(seed=11)
w.generate_population(n_customers=30)
cust_ids = list(w.state.customers.keys())[:4]

orch = MuleNetworkOrchestrator(w.state, seed=99)
ring = orch.generate_ring(cust_ids, correlation_type="shared_beneficiary")
print("single ring:", ring.network_id, "->", len(ring.traces), "mule traces")

result = generate_mule_network_corpus(w.state, target_traces=20, master_seed=101)
print("batch corpus: accepted =", len(result.accepted_traces), "rejected =", len(result.rejected_attempts))
