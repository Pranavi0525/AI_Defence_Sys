import pytest
import copy
from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

@pytest.fixture
def base_world():
    w = NormalWorld(seed=42)
    w.generate_population(n_customers=5)
    return w.get_state()

def test_exact_quota_completion(base_world):
    quotas = {"easy": 2, "medium": 2, "hard": 2, "advanced": 2}
    # No novelty to ensure fast completion
    res = generate_attack_corpus(
        copy.deepcopy(base_world),
        use_novelty=False,
        difficulty_quotas=quotas,
        max_attempts_multiplier=10
    )
    
    assert res.generation_statistics.accepted_by_difficulty == quotas
    for d, st in res.generation_statistics.status_by_difficulty.items():
        assert st == "COMPLETE"
    assert res.generation_statistics.shortfall_by_difficulty == {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}

def test_blocked_bucket_does_not_steal(base_world):
    quotas = {"easy": 2, "medium": 2, "hard": 20, "advanced": 2}
    # Turn on novelty with a very tight max attempts. Hard will likely fail to get 20 novel traces within 20*1 = 20 attempts.
    # We set multiplier to 2 so it exhausts quickly.
    res = generate_attack_corpus(
        copy.deepcopy(base_world),
        use_novelty=True,
        novelty_threshold=0.9,
        difficulty_quotas=quotas,
        max_attempts_multiplier=2
    )
    
    stats = res.generation_statistics
    # It must not generate more than 2 advanced attacks even if hard is short
    assert stats.accepted_by_difficulty["advanced"] == 2
    assert stats.accepted_by_difficulty["easy"] <= 2
    
    # If hard is short, it should be BLOCKED
    if stats.shortfall_by_difficulty["hard"] > 0:
        assert stats.status_by_difficulty["hard"] == "BLOCKED"
        assert stats.attempted_by_difficulty["hard"] >= quotas["hard"] * 2

def test_attempt_budgets_enforced(base_world):
    quotas = {"easy": 5}
    # Hardcode a low multiplier so it stops immediately
    res = generate_attack_corpus(
        copy.deepcopy(base_world),
        use_novelty=True,
        difficulty_quotas=quotas,
        max_attempts_multiplier=1
    )
    
    stats = res.generation_statistics
    assert stats.attempted_by_difficulty["easy"] <= 5
    if stats.shortfall_by_difficulty["easy"] > 0:
        assert stats.status_by_difficulty["easy"] == "BLOCKED"

def test_seed_reproducibility(base_world):
    quotas = {"easy": 5, "advanced": 5}
    res1 = generate_attack_corpus(
        copy.deepcopy(base_world), master_seed=123, use_novelty=True, difficulty_quotas=quotas, max_attempts_multiplier=5
    )
    res2 = generate_attack_corpus(
        copy.deepcopy(base_world), master_seed=123, use_novelty=True, difficulty_quotas=quotas, max_attempts_multiplier=5
    )
    
    assert res1.generation_statistics.accepted_by_difficulty == res2.generation_statistics.accepted_by_difficulty
    assert res1.generation_statistics.attempted_by_difficulty == res2.generation_statistics.attempted_by_difficulty
