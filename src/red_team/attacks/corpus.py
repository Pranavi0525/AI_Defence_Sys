"""Corpus Generation Module.

Handles batch generation of synthetic attack traces using the StatefulSimulator
and validates them using the RealismValidator.
"""

import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from red_team.world.state import WorldState
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.schemas.observable import ObservableAttackTrace
from red_team.schemas.ground_truth import AttackGroundTruth
from red_team.validation.realism import RealismReport, validate_attack_realism


from red_team.validation.novelty import NoveltyResult

class AttackRecord(BaseModel):
    """Internal storage format coupling trace, ground truth, and validation."""
    observable_trace: ObservableAttackTrace
    ground_truth: AttackGroundTruth
    validation_metadata: RealismReport
    novelty: Optional[NoveltyResult] = None


class GenerationStatistics(BaseModel):
    """Statistics for a generation run."""
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


class CorpusGenerationResult(BaseModel):
    """Result of a batch generation run."""
    accepted_traces: List[AttackRecord]
    rejected_attempts: List[Dict[str, Any]]
    generation_statistics: GenerationStatistics


def generate_attack_corpus(
    world_state: WorldState,
    target_count: int = 100,
    master_seed: int = 42,
    max_attempts: int = 500,
    use_novelty: bool = False,
    novelty_threshold: float = 0.85,
    difficulty_quotas: Optional[Dict[str, int]] = None,
    max_attempts_multiplier: int = 20,  # DOMAIN_MODELED maximum attempts per requested candidate
    attack_family: str = "ACCOUNT_TAKEOVER"
) -> CorpusGenerationResult:
    """Generate a valid corpus of attacks."""
    
    from red_team.schemas.id_generator import seed_ids
    seed_ids(master_seed)  # any new entity/event created below becomes
                             # reproducible given master_seed too, not
                             # just the attack behavior choices
    rng = random.Random(master_seed)
    
    if attack_family == "AUTHORIZED_PUSH_PAYMENT":
        from red_team.attacks.app_signature import get_app_signature
        signature = get_app_signature()
    elif attack_family == "ACCOUNT_TAKEOVER":
        signature = get_ato_signature()
    else:
        # Previously this was a bare `else: signature = get_ato_signature()`,
        # which silently generated ATO-signature traces (and wrote
        # ground_truth.attack_family="ACCOUNT_TAKEOVER") for ANY
        # unrecognized family, including "MULE_NETWORK" -- a real
        # ground-truth mislabeling bug caught when hard_example_generator.py
        # tried to mine harder MULE_NETWORK examples and got 12 traces
        # silently relabeled as ACCOUNT_TAKEOVER. MULE_NETWORK has its own
        # dedicated generator (generate_mule_network_corpus, ring-based,
        # no single-signature/difficulty-quota concept) and must never
        # reach this function. Fail loudly instead of silently mislabeling.
        raise ValueError(
            f"generate_attack_corpus() does not support attack_family="
            f"{attack_family!r}. Recognized: 'ACCOUNT_TAKEOVER', "
            f"'AUTHORIZED_PUSH_PAYMENT'. For MULE_NETWORK, call "
            f"generate_mule_network_corpus() instead."
        )
        
    from red_team.validation.novelty import NoveltyIndex, extract_fingerprint
    novelty_index = NoveltyIndex(similarity_threshold=novelty_threshold)
    
    accepted = []
    rejected = []
    
    customer_ids = list(world_state.customers.keys())
    if not customer_ids:
        raise ValueError("World state has no customers to attack.")
        
    difficulties = ["easy", "medium", "hard", "advanced"]
    entry_paths = signature.entry_states
    
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
                    attack_family=attack_family,
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
                    
        stats = _calculate_statistics(accepted, rejected, attempt, difficulty_quotas=difficulty_quotas, max_multiplier=max_attempts_multiplier)
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
                attack_family=attack_family,
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


def generate_mule_network_corpus(
    world_state: WorldState,
    target_traces: int = 100,
    mules_per_ring_range: tuple = (2, 4),
    master_seed: int = 42,
    max_attempts: int = 500,
    correlation_types: tuple = ("shared_beneficiary", "shared_bank_corridor"),
) -> CorpusGenerationResult:
    """Generate a valid MULE_NETWORK corpus.

    Each attempt builds one ring (several mule customers correlated
    through a shared collector beneficiary or bank_id corridor -- see
    MuleNetworkOrchestrator), against its OWN deepcopy of world_state so
    rings don't bleed state into each other, then validates each mule's
    trace individually against the ORIGINAL, untouched world_state --
    the same pattern generate_attack_corpus already uses for ATO/APP, so
    a mule's balance is checked against its true pre-attack starting
    point rather than a value some other ring already mutated.

    Individual mule traces are stored as ordinary AttackRecords, exactly
    like ATO/APP output, so nothing downstream needs special-case code
    for MULE_NETWORK. A ring stays recoverable after the fact via
    ground_truth.planner_metadata.plan_json['network_id'] -- every trace
    that shares a network_id came from the same ring.
    """
    import copy
    from red_team.attacks.mule_network_orchestrator import MuleNetworkOrchestrator
    from red_team.attacks.mule_network_signature import get_mule_network_signature
    from red_team.schemas.id_generator import seed_ids

    seed_ids(master_seed)  # collector-beneficiary IDs (network_id) become
                             # reproducible given master_seed too
    rng = random.Random(master_seed)
    signature = get_mule_network_signature()

    customer_ids = list(world_state.customers.keys())
    if len(customer_ids) < 2:
        raise ValueError("World state needs at least 2 customers to build a mule network.")

    accepted: List[AttackRecord] = []
    rejected: List[Dict[str, Any]] = []
    attempt = 0

    while len(accepted) < target_traces and attempt < max_attempts:
        attempt += 1
        child_seed = rng.getrandbits(32)
        ring_rng = random.Random(child_seed)

        ring_size = min(ring_rng.randint(*mules_per_ring_range), len(customer_ids))
        mule_customer_ids = ring_rng.sample(customer_ids, ring_size)
        difficulty = ring_rng.choice(["easy", "medium", "hard", "advanced"])
        correlation_type = ring_rng.choice(correlation_types)

        state_copy = copy.deepcopy(world_state)
        try:
            orch = MuleNetworkOrchestrator(state_copy, seed=child_seed)
            ring = orch.generate_ring(
                mule_customer_ids, difficulty=difficulty, correlation_type=correlation_type
            )
        except Exception as e:
            rejected.append({
                "attempt_id": attempt,
                "seed": child_seed,
                "failure_category": "simulation_error",
                "failure_reason": str(e),
                "difficulty": difficulty,
            })
            continue

        for trace, gt in zip(ring.traces, ring.ground_truths):
            report = validate_attack_realism(trace, gt, signature, world_state)
            if report.status == "ACCEPTED" and report.constraint.passed:
                accepted.append(AttackRecord(
                    observable_trace=trace,
                    ground_truth=gt,
                    validation_metadata=report,
                    novelty=None,
                ))
            else:
                rejected.append({
                    "attempt_id": attempt,
                    "seed": child_seed,
                    "failure_category": "validation_rejection",
                    "failure_reason": "; ".join(report.failures),
                    "validation_metadata": report,
                    "trace": trace,
                    "difficulty": difficulty,
                })

    stats = _calculate_statistics(accepted, rejected, attempt)
    return CorpusGenerationResult(accepted_traces=accepted, rejected_attempts=rejected, generation_statistics=stats)

def _calculate_statistics(accepted: List[AttackRecord], rejected: List[Dict[str, Any]], attempts: int, difficulty_quotas: Optional[Dict[str, int]] = None, max_multiplier: int = 20) -> GenerationStatistics:
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
        elif attempted_by_diff.get(d, 0) >= q * max_multiplier:
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
