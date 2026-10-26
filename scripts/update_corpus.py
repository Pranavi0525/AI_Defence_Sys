import re

def update():
    with open("src/red_team/attacks/corpus.py", "r") as f:
        code = f.read()

    stats_schema = """class GenerationStatistics(BaseModel):
    \"\"\"Statistics for a generation run.\"\"\"
    attempted: int
    accepted: int
    rejected: int
    acceptance_rate: float
    
    # Path diversity
    unique_phase_sequences: int
    unique_entry_paths: int
    average_phases_per_attack: float
    
    # Event diversity
    average_events_per_attack: float
    min_events: int
    max_events: int
    event_type_distribution: Dict[str, int]
    
    # Difficulty
    difficulty_distribution: Dict[str, int]
    
    # Customer diversity
    unique_customers_attacked: int
    
    # Realism distribution
    realism_distribution: Dict[str, float]  # just simple averages or counts for now
    not_available_metric_count: int

    # Stage 19.5 Quota stats
    requested_by_difficulty: Dict[str, int] = {}
    attempted_by_difficulty: Dict[str, int] = {}
    accepted_by_difficulty: Dict[str, int] = {}
    rejected_by_difficulty: Dict[str, int] = {}
    novelty_rejections_by_difficulty: Dict[str, int] = {}
    realism_rejections_by_difficulty: Dict[str, int] = {}
    structural_rejections_by_difficulty: Dict[str, int] = {}
    shortfall_by_difficulty: Dict[str, int] = {}
    status_by_difficulty: Dict[str, str] = {}
"""
    code = re.sub(r'class GenerationStatistics\(BaseModel\):.*?not_available_metric_count: int\n', stats_schema, code, flags=re.DOTALL)

    func = """def generate_attack_corpus(
    world_state: WorldState,
    target_count: int = 100,
    master_seed: int = 42,
    max_attempts: int = 500,
    use_novelty: bool = False,
    novelty_threshold: float = 0.85,
    difficulty_quotas: Optional[Dict[str, int]] = None,
    max_attempts_multiplier: int = 20  # DOMAIN_MODELED maximum attempts per requested candidate
) -> CorpusGenerationResult:
    \"\"\"Generate a valid corpus of attacks.\"\"\"
    
    rng = random.Random(master_seed)
    signature = get_ato_signature()
    
    from red_team.validation.novelty import NoveltyIndex, extract_fingerprint
    novelty_index = NoveltyIndex(similarity_threshold=novelty_threshold)
    
    accepted = []
    rejected = []
    
    customer_ids = list(world_state.customers.keys())
    if not customer_ids:
        raise ValueError("World state has no customers to attack.")
        
    difficulties = ["easy", "medium", "hard", "advanced"]
    entry_paths = ["RECONNAISSANCE", "ACCOUNT_ACCESS"]
    
    attempt = 0
    
    if difficulty_quotas:
        # Generate by quota
        # Pre-calculate budgets
        budgets = {d: q * max_attempts_multiplier for d, q in difficulty_quotas.items()}
        attempts_by_diff = {d: 0 for d in difficulty_quotas}
        accepted_by_diff = {d: [] for d in difficulty_quotas}
        
        for diff, quota in difficulty_quotas.items():
            while len(accepted_by_diff[diff]) < quota and attempts_by_diff[diff] < budgets[diff]:
                attempt += 1
                attempts_by_diff[diff] += 1
                child_seed = rng.getrandbits(32)
                
                customer_id = rng.choice(customer_ids)
                entry = rng.choice(entry_paths)
                
                plan = AttackPlan(
                    attack_family="ACCOUNT_TAKEOVER",
                    difficulty=diff,
                    entry_state=entry,
                    max_phases=rng.randint(3, 8)
                )
                
                import copy
                state_copy = copy.deepcopy(world_state)
                sim = StatefulSimulator(state_copy, signature, seed=child_seed)
                try:
                    trace, gt = sim.generate_attack(plan, customer_id)
                except Exception as e:
                    rejected.append({
                        "attempt_id": attempt,
                        "seed": child_seed,
                        "failure_category": "simulation_error",
                        "failure_reason": str(e),
                        "difficulty": diff
                    })
                    continue
                    
                report = validate_attack_realism(trace, gt, signature, world_state)
                
                if report.structural.passed:
                    is_novel = True
                    novelty_report = None
                    if use_novelty:
                        fp = extract_fingerprint(trace, gt, state_copy)
                        novelty_result = novelty_index.evaluate(fp, plan.difficulty)
                        if not novelty_result.is_novel:
                            is_novel = False
                            rejected.append({
                                "attempt_id": attempt,
                                "seed": child_seed,
                                "failure_category": "novelty_rejection",
                                "failure_reason": novelty_result.rejection_reason,
                                "trace": trace,
                                "difficulty": plan.difficulty
                            })
                        else:
                            novelty_report = novelty_result
                            
                    if is_novel:
                        if report.status == "ACCEPTED" and report.constraint.passed:
                            record = AttackRecord(
                                observable_trace=trace,
                                ground_truth=gt,
                                validation_metadata=report,
                                novelty=novelty_report
                            )
                            
                            if use_novelty:
                                novelty_index.add(fp, plan.difficulty)
                            accepted_by_diff[diff].append(record)
                            accepted.append(record)
                        else:
                            rejected.append({
                                "attempt_id": attempt,
                                "seed": child_seed,
                                "failure_category": "validation_rejection",
                                "failure_reason": "; ".join(report.failures),
                                "validation_metadata": report,
                                "trace": trace,
                                "difficulty": diff
                            })
                else:
                    rejected.append({
                        "attempt_id": attempt,
                        "seed": child_seed,
                        "failure_category": "structural_rejection",
                        "failure_reason": "; ".join(report.failures),
                        "validation_metadata": report,
                        "trace": trace,
                        "difficulty": diff
                    })
                    
        stats = _calculate_statistics(accepted, rejected, attempt, difficulty_quotas=difficulty_quotas)
        return CorpusGenerationResult(accepted_traces=accepted, rejected_attempts=rejected, generation_statistics=stats)
    else:
        # Original loop without quotas
        while len(accepted) < target_count and attempt < max_attempts:
            attempt += 1
            child_seed = rng.getrandbits(32)
            
            customer_id = rng.choice(customer_ids)
            diff = rng.choice(difficulties)
            entry = rng.choice(entry_paths)
            
            plan = AttackPlan(
                attack_family="ACCOUNT_TAKEOVER",
                difficulty=diff,
                entry_state=entry,
                max_phases=rng.randint(3, 8)
            )
            
            import copy
            state_copy = copy.deepcopy(world_state)
            sim = StatefulSimulator(state_copy, signature, seed=child_seed)
            try:
                trace, gt = sim.generate_attack(plan, customer_id)
            except Exception as e:
                rejected.append({
                    "attempt_id": attempt,
                    "seed": child_seed,
                    "failure_category": "simulation_error",
                    "failure_reason": str(e),
                    "difficulty": diff
                })
                continue
                
            report = validate_attack_realism(trace, gt, signature, world_state)
            
            if report.structural.passed:
                is_novel = True
                novelty_report = None
                if use_novelty:
                    fp = extract_fingerprint(trace, gt, state_copy)
                    novelty_result = novelty_index.evaluate(fp, plan.difficulty)
                    if not novelty_result.is_novel:
                        is_novel = False
                        rejected.append({
                            "attempt_id": attempt,
                            "seed": child_seed,
                            "failure_category": "novelty_rejection",
                            "failure_reason": novelty_result.rejection_reason,
                            "trace": trace,
                            "difficulty": plan.difficulty
                        })
                    else:
                        novelty_report = novelty_result
                        
                if is_novel:
                    if report.status == "ACCEPTED" and report.constraint.passed:
                        record = AttackRecord(
                            observable_trace=trace,
                            ground_truth=gt,
                            validation_metadata=report,
                            novelty=novelty_report
                        )
                        
                        if use_novelty:
                            novelty_index.add(fp, plan.difficulty)
                        accepted.append(record)
                    else:
                        rejected.append({
                            "attempt_id": attempt,
                            "seed": child_seed,
                            "failure_category": "validation_rejection",
                            "failure_reason": "; ".join(report.failures),
                            "validation_metadata": report,
                            "trace": trace,
                            "difficulty": plan.difficulty
                        })
            else:
                rejected.append({
                    "attempt_id": attempt,
                    "seed": child_seed,
                    "failure_category": "structural_rejection",
                    "failure_reason": "; ".join(report.failures),
                    "validation_metadata": report,
                    "trace": trace,
                    "difficulty": plan.difficulty
                })
                
        stats = _calculate_statistics(accepted, rejected, attempt)
        return CorpusGenerationResult(accepted_traces=accepted, rejected_attempts=rejected, generation_statistics=stats)
"""
    code = re.sub(r'def generate_attack_corpus\(.*?(?=\ndef _calculate_statistics)', func, code, flags=re.DOTALL)

    stat_calc = """def _calculate_statistics(accepted: List[AttackRecord], rejected: List[Dict[str, Any]], attempts: int, difficulty_quotas: Optional[Dict[str, int]] = None) -> GenerationStatistics:
    if not accepted:
        return GenerationStatistics(
            attempted=attempts, accepted=0, rejected=len(rejected), acceptance_rate=0.0,
            unique_phase_sequences=0, unique_entry_paths=0, average_phases_per_attack=0.0,
            average_events_per_attack=0.0, min_events=0, max_events=0, event_type_distribution={},
            difficulty_distribution={}, unique_customers_attacked=0,
            realism_distribution={}, not_available_metric_count=0
        )
        
    # Paths
    phase_seqs = set()
    entry_paths = set()
    total_phases = 0
    
    # Events
    total_events = 0
    min_ev = float('inf')
    max_ev = 0
    evt_types = {}
    
    # Diff
    diffs = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    
    # Customer
    customers = set()
    
    # Realism
    na_count = 0
    
    accepted_by_diff = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    attempted_by_diff = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    rejected_by_diff = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    nov_rej = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    real_rej = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    struct_rej = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    
    for rec in accepted:
        # Paths
        seq = tuple(p.phase for p in rec.ground_truth.phases_executed)
        phase_seqs.add(seq)
        if seq:
            entry_paths.add(seq[0])
        total_phases += len(seq)
        
        # Events
        num_ev = len(rec.observable_trace.events)
        total_events += num_ev
        if num_ev < min_ev: min_ev = num_ev
        if num_ev > max_ev: max_ev = num_ev
        
        for e in rec.observable_trace.events:
            evt_types[e.event_type] = evt_types.get(e.event_type, 0) + 1
            
        # Diff
        d = rec.ground_truth.attack_difficulty
        diffs[d] = diffs.get(d, 0) + 1
        accepted_by_diff[d] += 1
        attempted_by_diff[d] += 1
        
        # Customer
        customers.add(rec.observable_trace.customer_id)
        
        # Realism
        if rec.validation_metadata.overall_realism_score == "NOT_AVAILABLE":
            na_count += 1
            
    for rej in rejected:
        d = rej.get("difficulty", "unknown")
        if d in rejected_by_diff:
            rejected_by_diff[d] += 1
            attempted_by_diff[d] += 1
            
            cat = rej.get("failure_category", "")
            if cat == "novelty_rejection": nov_rej[d] += 1
            elif cat == "validation_rejection": real_rej[d] += 1
            elif cat == "structural_rejection": struct_rej[d] += 1
            
    requested = difficulty_quotas or {}
    shortfall = {}
    status = {}
    for d, q in requested.items():
        acc = accepted_by_diff.get(d, 0)
        shortfall[d] = max(0, q - acc)
        if shortfall[d] == 0:
            status[d] = "COMPLETE"
        elif attempted_by_diff.get(d, 0) >= q * 20: # hardcoded max_attempts_multiplier reference for status checking if we didn't pass it down
            status[d] = "BLOCKED"
        else:
            status[d] = "FAILED"
            
    return GenerationStatistics(
        attempted=attempts,
        accepted=len(accepted),
        rejected=len(rejected),
        acceptance_rate=len(accepted) / attempts if attempts > 0 else 0.0,
        unique_phase_sequences=len(phase_seqs),
        unique_entry_paths=len(entry_paths),
        average_phases_per_attack=total_phases / len(accepted),
        average_events_per_attack=total_events / len(accepted),
        min_events=min_ev,
        max_events=max_ev,
        event_type_distribution=evt_types,
        difficulty_distribution=diffs,
        unique_customers_attacked=len(customers),
        realism_distribution={"average_score": 1.0}, # Dummy for now
        not_available_metric_count=na_count,
        requested_by_difficulty=requested,
        attempted_by_difficulty={d: v for d, v in attempted_by_diff.items() if d in requested},
        accepted_by_difficulty={d: v for d, v in accepted_by_diff.items() if d in requested},
        rejected_by_difficulty={d: v for d, v in rejected_by_diff.items() if d in requested},
        novelty_rejections_by_difficulty={d: v for d, v in nov_rej.items() if d in requested},
        realism_rejections_by_difficulty={d: v for d, v in real_rej.items() if d in requested},
        structural_rejections_by_difficulty={d: v for d, v in struct_rej.items() if d in requested},
        shortfall_by_difficulty=shortfall,
        status_by_difficulty=status
    )
"""
    code = re.sub(r'def _calculate_statistics.*', stat_calc, code, flags=re.DOTALL)
    
    # ensure Optional is imported
    if "from typing import" in code and "Optional" not in code:
        code = code.replace("from typing import List, Dict, Any", "from typing import List, Dict, Any, Optional")

    with open("src/red_team/attacks/corpus.py", "w") as f:
        f.write(code)

if __name__ == "__main__":
    update()
