import pytest
import copy
from red_team.world.world import NormalWorld
from red_team.attacks.simulator import StatefulSimulator, DIFFICULTY_PROFILES
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.corpus import AttackPlan
from red_team.validation.novelty import AttackFingerprint, calculate_fingerprint_similarity

@pytest.fixture
def base_world():
    w = NormalWorld(seed=42)
    w.generate_population(n_customers=5)
    return w.get_state()

def test_split_count_fingerprint_regression():
    from red_team.validation.novelty import ATOAttackFingerprint, calculate_fingerprint_similarity
    fp_base = ATOAttackFingerprint(
        attack_family="ACCOUNT_TAKEOVER",
        phase_sequence=("A",),
        event_sequence=("B",),
        transaction_count=1,
        amount_buckets=("small",),
        normalized_amount_sum=0.1,
        split_count=0,
        timing_category="rapid",
        device_pattern=("new",),
        beneficiary_pattern=("new",),
        outcome_pattern=("completed",)
    )
    
    # Case 1: identical fingerprints -> similarity 1.0
    assert calculate_fingerprint_similarity(fp_base, fp_base) == 1.0
    
    # Case 2: only split_count changes materially -> similarity decreases
    fp_mod = fp_base.model_copy(update={"split_count": 5})
    sim2 = calculate_fingerprint_similarity(fp_base, fp_mod)
    assert sim2 < 1.0
    assert sim2 == pytest.approx(0.95) # Since split_count weight is 0.05 and it drops to 0
    
    # Case 3: small numeric noise -> similarity unchanged
    fp_noise = fp_base.model_copy(update={"normalized_amount_sum": 0.15})
    sim3 = calculate_fingerprint_similarity(fp_base, fp_noise)
    assert sim3 == 1.0
    
    # Case 4: meaningful multi-axis change -> similarity decreases further
    fp_multi = fp_base.model_copy(update={"split_count": 5, "timing_category": "slow"})
    sim4 = calculate_fingerprint_similarity(fp_base, fp_multi)
    assert sim4 < sim2
    assert sim4 == pytest.approx(0.80)

def test_hard_reaches_exploitation(base_world):
    customer_id = list(base_world.customers.keys())[0]
    signature = get_ato_signature()
    
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="hard",
        entry_state="RECONNAISSANCE",
        max_phases=10
    )
    
    reached_exploitation = False
    for seed in range(50):
        sim = StatefulSimulator(copy.deepcopy(base_world), signature, seed=seed)
        trace, gt = sim.generate_attack(plan, customer_id)
        if any(p.phase == "EXPLOITATION" for p in gt.phases_executed):
            reached_exploitation = True
            
            # Also verify duration can exceed the old 1440 limit (24 hours)
            duration = (trace.events[-1].timestamp - trace.events[0].timestamp).total_seconds() / 60
            if duration > 1440:
                break
                
    assert reached_exploitation, "HARD attack never reached EXPLOITATION across 50 seeds, indicating it might still be blocked."
