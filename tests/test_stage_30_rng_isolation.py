import pytest
from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

def test_rng_isolation_same_master_seed():
    """
    Regression test to ensure that identical master seeds do not produce identical
    attack IDs or identical phase timestamps across different attack families,
    thanks to the attack_family salt in the StatefulSimulator seed generation.
    """
    # 1. Setup a small identical world state
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    world.generate_legitimate_events(num_events=20)
    ws = world.get_state()

    # 2. Generate ATO corpus with master_seed=42
    ato_res = generate_attack_corpus(
        world_state=ws,
        target_count=5,
        master_seed=42,
        max_attempts=50,
        attack_family="ACCOUNT_TAKEOVER",
        difficulty_quotas={"easy": 5}
    )

    # 3. Generate APP corpus with the exact same master_seed=42
    app_res = generate_attack_corpus(
        world_state=ws,
        target_count=5,
        master_seed=42,
        max_attempts=50,
        attack_family="AUTHORIZED_PUSH_PAYMENT",
        difficulty_quotas={"easy": 5}
    )

    # 4. Gather attack IDs
    ato_ids = {t.ground_truth.attack_id for t in ato_res.accepted_traces}
    app_ids = {t.ground_truth.attack_id for t in app_res.accepted_traces}

    # Gather failed attack IDs if any
    for r in getattr(ato_res, "rejected_attempts", []):
        if isinstance(r, dict) and r.get("trace"):
            ato_ids.add(r["trace"].trace_id)
            
    for r in getattr(app_res, "rejected_attempts", []):
        if isinstance(r, dict) and r.get("trace"):
            app_ids.add(r["trace"].trace_id)

    # 5. Assert no attack_id collisions
    collisions = ato_ids.intersection(app_ids)
    assert not collisions, f"Found colliding attack IDs across families: {collisions}"

    # 6. Gather timestamps for the first phase of each trace (which would collide if seeds matched)
    def get_initial_timestamps(res):
        ts = set()
        for t in res.accepted_traces:
            phases = t.ground_truth.phases_executed
            if phases:
                ts.add((phases[0].entered_at, phases[0].exited_at))
        return ts

    ato_ts = get_initial_timestamps(ato_res)
    app_ts = get_initial_timestamps(app_res)

    # 7. Assert no timestamp collisions
    ts_collisions = ato_ts.intersection(app_ts)
    assert not ts_collisions, f"Found colliding initial phase timestamps: {ts_collisions}"
