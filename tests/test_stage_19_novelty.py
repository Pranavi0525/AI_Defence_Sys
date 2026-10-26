import pytest
import copy
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from red_team.world.world import NormalWorld
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.attacks.ato_signature import get_ato_signature
from red_team.validation.novelty import NoveltyIndex, extract_fingerprint, calculate_fingerprint_similarity
from red_team.attacks.corpus import generate_attack_corpus

@pytest.fixture
def base_world():
    w = NormalWorld(seed=42)
    w.generate_population(n_customers=2)
    return w.get_state()

@pytest.fixture
def signature():
    return get_ato_signature()

def generate_trace(world, signature, seed, difficulty="medium"):
    sim = StatefulSimulator(copy.deepcopy(world), signature, seed=seed)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty=difficulty)
    c_id = list(world.customers.keys())[0]
    trace, gt = sim.generate_attack(plan, c_id)
    return trace, gt

def test_trivial_noise_is_not_novel(base_world, signature):
    # Same seed yields same attack
    t1, gt1 = generate_trace(base_world, signature, seed=123)
    t2, gt2 = generate_trace(base_world, signature, seed=123)
    
    # Mutate UUIDs
    t2.trace_id = "atk-something-else"
    for e in t2.events:
        e.event_id = "evt-new-id"
    
    # Mutate Timestamps by shifting exact same amount (1 hour)
    for e in t2.events:
        e.timestamp = e.timestamp + timedelta(hours=1)
        
    # Mutate amount by tiny fraction (within same bucket)
    for e in t2.events:
        if getattr(e, "amount", None):
            e.amount = str(float(e.amount) + 0.05)

    fp1 = extract_fingerprint(t1, gt1, base_world)
    fp2 = extract_fingerprint(t2, gt2, base_world)
    
    sim = calculate_fingerprint_similarity(fp1, fp2)
    assert sim > 0.99  # Exact match on all categorical vectors and within tolerance on amount

def test_meaningful_change_is_novel(base_world, signature):
    # Different seeds produce different attacks (usually)
    t1, gt1 = generate_trace(base_world, signature, seed=10)
    t2, gt2 = generate_trace(base_world, signature, seed=42)
    
    fp1 = extract_fingerprint(t1, gt1, base_world)
    fp2 = extract_fingerprint(t2, gt2, base_world)
    
    sim = calculate_fingerprint_similarity(fp1, fp2)
    assert sim < 1.0  # Should be less than perfect match
    
def test_bounded_index():
    index = NoveltyIndex(max_size=3)
    from red_team.validation.novelty import ATOAttackFingerprint
    
    def dummy_fp(i):
        return ATOAttackFingerprint(
            attack_family="ACCOUNT_TAKEOVER",
            phase_sequence=("A", str(i)),
            event_sequence=("B",),
            transaction_count=i,
            amount_buckets=("small",),
            normalized_amount_sum=0.1 * i,
            split_count=1,
            timing_category="rapid",
            device_pattern=("new",),
            beneficiary_pattern=("new",),
            outcome_pattern=("completed",)
        )
        
    index.add(dummy_fp(1), "easy")
    index.add(dummy_fp(2), "easy")
    index.add(dummy_fp(3), "easy")
    assert len(index.fingerprints["ACCOUNT_TAKEOVER"]["easy"]) == 3
    index.add(dummy_fp(4), "easy")
    assert len(index.fingerprints["ACCOUNT_TAKEOVER"]["easy"]) == 3

def test_novelty_pipeline_integration(base_world):
    # Just ensuring it runs without crashing and correctly filters duplicates
    res = generate_attack_corpus(base_world, target_count=5, master_seed=42, max_attempts=50, use_novelty=True, novelty_threshold=0.99)
    assert len(res.accepted_traces) > 0
    # They should all be considered novel
    for rec in res.accepted_traces:
        assert getattr(rec, "novelty").is_novel == True

def test_cross_difficulty_behavior():
    index = NoveltyIndex(similarity_threshold=0.85)
    from red_team.validation.novelty import ATOAttackFingerprint
    
    fp_base = ATOAttackFingerprint(
        attack_family="ACCOUNT_TAKEOVER",
        phase_sequence=("A",),
        event_sequence=("B",),
        transaction_count=1,
        amount_buckets=("small",),
        normalized_amount_sum=0.1,
        split_count=1,
        timing_category="rapid",
        device_pattern=("new",),
        beneficiary_pattern=("new",),
        outcome_pattern=("completed",)
    )
    
    # Add to EASY bucket
    index.add(fp_base, "easy")
    
    # same structure, different difficulty -> evaluated in different bucket it IS novel
    res = index.evaluate(fp_base, "hard")
    assert res.is_novel == True
    
    # same difficulty, meaningful difference -> novelty decreases similarity appropriately
    fp_diff = fp_base.model_copy(update={"amount_buckets": ("large",), "timing_category": "bursty"})
    res2 = index.evaluate(fp_diff, "easy")
    assert res2.is_novel == True
    assert res2.similarity_to_closest < 0.85
    
    # same difficulty, same structure
    res3 = index.evaluate(fp_base, "easy")
    assert res3.is_novel == False
