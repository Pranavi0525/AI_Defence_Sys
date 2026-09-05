from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_mule_network_corpus

w = NormalWorld(seed=33)
w.generate_population(n_customers=30)
result = generate_mule_network_corpus(w.state, target_traces=20, master_seed=101)
print(len(result.accepted_traces), len(result.rejected_attempts))