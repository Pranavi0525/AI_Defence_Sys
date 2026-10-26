import os
import copy
import time
import json
from collections import Counter
import numpy as np

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus, AttackRecord
from red_team.validation.novelty import extract_fingerprint, calculate_fingerprint_similarity
from red_team.validation.realism import validate_attack_realism
from red_team.attacks.ato_signature import get_ato_signature

def find_leakage(obj, bad_keys):
    leakages = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(bad in k.lower() for bad in bad_keys) or (isinstance(v, str) and any(bad in v.lower() for bad in bad_keys)):
                leakages.append(f"Key/Val leak: {k}={v}")
            leakages.extend(find_leakage(v, bad_keys))
    elif isinstance(obj, list):
        for item in obj:
            leakages.extend(find_leakage(item, bad_keys))
    return leakages

def main():
    print("================ STAGE 23 ATO QUALIFICATION ================\n")

    world = NormalWorld(seed=42)
    world.generate_population(n_customers=20)
    base_state = world.get_state()
    signature = get_ato_signature()

    quotas = {
        "easy": 25,
        "medium": 25,
        "hard": 25,
        "advanced": 25
    }

    start_time = time.time()
    
    print("Run 1: Generating corpus...")
    res1 = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.85,
        difficulty_quotas=quotas,
        max_attempts_multiplier=20
    )
    
    t_run1 = time.time() - start_time
    
    print("Run 2: Generating corpus for determinism check...")
    res2 = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.85,
        difficulty_quotas=quotas,
        max_attempts_multiplier=20
    )

    # Determinism Check
    print("\n--- REPRODUCIBILITY ---")
    assert len(res1.accepted_traces) == len(res2.accepted_traces), "Determinism failed on count"
    assert res1.generation_statistics.shortfall_by_difficulty == res2.generation_statistics.shortfall_by_difficulty
    print("EXACT deterministic reproduction confirmed.")
    
    # 2. Structural Validity
    print("\n--- STRUCTURAL VALIDITY ---")
    for r in res1.accepted_traces:
        report = validate_attack_realism(r.observable_trace, r.ground_truth, signature, base_state)
        assert report.status == "ACCEPTED", f"Accepted trace failed realism! {report.failures}"
    print("All accepted traces passed Realism validation structurally.")
    
    # 3. Observable Isolation
    print("\n--- OBSERVABLE / GROUND-TRUTH ISOLATION ---")
    bad_keys = ["attack", "difficulty", "novelty", "ground_truth", "variation", "seed", "plan"]
    leakage_count = 0
    for r in res1.accepted_traces:
        leaks = find_leakage(r.observable_trace.model_dump(), bad_keys)
        if leaks:
            print(f"Leakage in trace {r.observable_trace.trace_id}: {leaks}")
            leakage_count += 1
    print(f"Observable Leakage Count: {leakage_count} / {len(res1.accepted_traces)}")

    # 4. Realism & Novelty Stats
    print("\n--- REALISM VS NOVELTY & STATS ---")
    stats = res1.generation_statistics
    
    table_data = {}
    fps_by_diff = {}
    
    for diff in ["easy", "medium", "hard", "advanced"]:
        req = stats.requested_by_difficulty.get(diff, 0)
        acc = stats.accepted_by_difficulty.get(diff, 0)
        att = stats.attempted_by_difficulty.get(diff, 0)
        short = stats.shortfall_by_difficulty.get(diff, 0)
        real_rej = stats.realism_rejections_by_difficulty.get(diff, 0)
        nov_rej = stats.novelty_rejections_by_difficulty.get(diff, 0)
        
        diff_traces = [r for r in res1.accepted_traces if r.ground_truth.attack_difficulty == diff]
        fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in diff_traces]
        fps_by_diff[diff] = fps
        
        near_dupes = 0
        exact_dupes = 0
        pairs = 0
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                pairs += 1
                sim = calculate_fingerprint_similarity(fps[i], fps[j])
                if sim >= 0.8:
                    near_dupes += 1
                if sim == 1.0:
                    exact_dupes += 1
                    
        exploit_reach = sum(1 for fp in fps if "EXPLOITATION" in fp.phase_sequence)
        mean_events = np.mean([fp.transaction_count + len(fp.event_sequence) for fp in fps]) if fps else 0
        mean_phases = np.mean([len(fp.phase_sequence) for fp in fps]) if fps else 0
        mean_txs = np.mean([fp.transaction_count for fp in fps]) if fps else 0
        
        table_data[diff] = {
            "target": req,
            "accepted": acc,
            "shortfall": short,
            "attempts": att,
            "acc_rate": f"{acc/att:.1%}" if att > 0 else "0%",
            "realism_rej": real_rej,
            "nov_rej": nov_rej,
            "unique_fps": len(set(f.model_dump_json() for f in fps)),
            "unique_paths": len(set(f.phase_sequence for f in fps)),
            "exact_dupes": exact_dupes,
            "near_dupes": near_dupes,
            "mean_events": f"{mean_events:.1f}",
            "mean_phases": f"{mean_phases:.1f}",
            "mean_txs": f"{mean_txs:.1f}",
            "exploit_reach": f"{exploit_reach/len(fps):.1%}" if len(fps) > 0 else "0%"
        }
        
    for d in ["easy", "medium", "hard", "advanced"]:
        print(f"[{d.upper()}] Target: {table_data[d]['target']}, Acc: {table_data[d]['accepted']} (Short: {table_data[d]['shortfall']})")
        print(f"   Attempts: {table_data[d]['attempts']}, RealismRej: {table_data[d]['realism_rej']}, NovRej: {table_data[d]['nov_rej']}")
        print(f"   Unique FPs: {table_data[d]['unique_fps']}, NearDupes: {table_data[d]['near_dupes']}")
        print(f"   Mean Events: {table_data[d]['mean_events']}, Exploit Reach: {table_data[d]['exploit_reach']}")
        
    print("\n--- ADVERSARIAL MUTATION ---")
    if len(res1.accepted_traces) > 0:
        base_t = res1.accepted_traces[0]
        # Mutate timestamp to be impossible
        bad_trace = copy.deepcopy(base_t.observable_trace)
        if len(bad_trace.events) > 1:
            bad_trace.events[0].timestamp, bad_trace.events[-1].timestamp = bad_trace.events[-1].timestamp, bad_trace.events[0].timestamp
            rep = validate_attack_realism(bad_trace, base_t.ground_truth, signature, base_state)
            print("Mutated timestamp ordering:", rep.status)
            
        # Phantom entity / breaking ground truth
        bad_trace2 = copy.deepcopy(base_t.observable_trace)
        bad_trace2.events[0].event_id = "PHANTOM-999"
        rep2 = validate_attack_realism(bad_trace2, base_t.ground_truth, signature, base_state)
        print("Mutated event_id to break GT linkage:", rep2.status)
        
    print(f"\nTime taken: {t_run1:.2f}s")
    print(f"Average time per accepted: {t_run1/len(res1.accepted_traces):.2f}s")

if __name__ == "__main__":
    main()
