import os
import sys
from collections import Counter
import logging

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Normal World (Seed 42)...")
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    
    print("Generating Stage 12 Corpus...")
    result = generate_attack_corpus(
        world.get_state(), 
        target_count=100, 
        master_seed=42, 
        max_attempts=1000
    )
    
    stats = result.generation_statistics
    
    print("\n--- REJECTION STATISTICS ---")
    print(f"Attempts: {stats.attempted}")
    print(f"Accepted: {stats.accepted}")
    print(f"Rejected: {stats.rejected}")
    print(f"Acceptance Rate: {stats.acceptance_rate:.2%}")
    
    primary_reasons = Counter()
    category_reasons = Counter()
    
    for r in result.rejected_attempts:
        if r["failure_category"] == "validation_rejection":
            report = r.get("validation_metadata")
            if report and report.primary_failure:
                code = report.primary_failure.reason_code
                cat = report.primary_failure.category
                primary_reasons[code] += 1
                category_reasons[cat] += 1
    
    print("\n--- PRIMARY REJECTIONS BY CODE ---")
    for k, v in primary_reasons.most_common():
        print(f"  {k}: {v}")
        
    print("\n--- TRANSACTION STATS ---")
    approved = 0
    declined = 0
    
    for rec in result.accepted_traces:
        for e in rec.observable_trace.events:
            if e.event_type == "TRANSACTION":
                if e.transaction_status == "completed":
                    approved += 1
                elif e.transaction_status == "failed":
                    declined += 1
                    
    print(f"  Approved: {approved}")
    print(f"  Declined: {declined}")

if __name__ == "__main__":
    main()
