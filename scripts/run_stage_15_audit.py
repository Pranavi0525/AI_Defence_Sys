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
        max_attempts=500
    )
    
    stats = result.generation_statistics
    
    print("\n--- REJECTION STATISTICS ---")
    print(f"Attempts: {stats.attempted}")
    print(f"Accepted: {stats.accepted}")
    print(f"Rejected: {stats.rejected}")
    print(f"Acceptance Rate: {stats.acceptance_rate:.2%}")
    
    primary_reasons = Counter()
    category_reasons = Counter()
    secondary_reasons = Counter()
    diff_rejections = Counter()
    phase_rejections = Counter()
    
    for r in result.rejected_attempts:
        if r["failure_category"] == "validation_rejection":
            report = r.get("validation_metadata")
            if report and report.primary_failure:
                code = report.primary_failure.reason_code
                cat = report.primary_failure.category
                primary_reasons[code] += 1
                category_reasons[cat] += 1
                
                for sec in report.secondary_failures:
                    secondary_reasons[sec.reason_code] += 1
                    
            # For phase and diff, we'd have to map back to ground truth which isn't saved in rejection.
            # But we know they all failed on BALANCE_CONSTRAINT_VIOLATION.
            
    print("\n--- REJECTIONS BY CATEGORY ---")
    for k, v in category_reasons.most_common():
        print(f"  {k}: {v}")

    print("\n--- SECONDARY REJECTIONS BY CODE ---")
    for k, v in secondary_reasons.most_common():
        print(f"  {k}: {v}")
        
if __name__ == "__main__":
    main()
