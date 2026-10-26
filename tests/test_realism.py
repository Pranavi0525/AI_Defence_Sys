"""Tests for Stage 11 — Attack Realism Validator."""

import pytest
import datetime
from decimal import Decimal

from red_team.world.world import NormalWorld
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.validation.realism import validate_attack_realism
from red_team.schemas.observable import ObservableTransactionEvent, ObservableAttackTrace
from red_team.schemas.ground_truth import AttackPhaseRecord


def test_valid_trace():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    import copy
    state_copy = copy.deepcopy(world.get_state())
    sim = StatefulSimulator(state_copy, get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    
    assert report.status == "ACCEPTED"
    assert report.structural.passed
    assert report.constraint.passed
    assert "statistical" in report.unavailable_metrics


def test_invalid_timestamp_rejection():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Tamper with timestamps to go backwards
    if len(trace.events) >= 2:
        trace.events[1].timestamp = trace.events[0].timestamp - datetime.timedelta(days=1)
        
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.structural.passed
    assert any("chronologically ordered" in f for f in report.failures)


def test_invalid_account_reference():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Tamper with an account ID in a transaction event if it exists
    for e in trace.events:
        if e.event_type == "TRANSACTION":
            e.account_id = "nonexistent-account"
            
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    assert any("Invalid entity reference" in f for f in report.failures)


def test_invalid_balance():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Tamper with amount to be huge
    for e in trace.events:
        if e.event_type == "TRANSACTION":
            e.amount = Decimal("999999999.00")
            
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    assert any("Impossible balance transition" in f for f in report.failures)


def test_invalid_entity_or_ground_truth_rejection():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Tamper with event IDs to break linkage
    trace.events[0].event_id = "fake-id"
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.structural.passed
    assert any("Ground truth IDs do not match" in f for f in report.failures)


def test_beneficiary_ordering():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    # Create fake trace where transfer occurs before beneficiary addition
    from red_team.schemas.observable import ObservableBeneficiaryEvent, ObservableRelationshipEvent
    
    trace = ObservableAttackTrace(
        trace_id="atk-1",
        customer_id=customer_id,
        events=[
            ObservableRelationshipEvent(
                event_type="RELATIONSHIP_CHANGE",
                event_id="e1",
                timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
                customer_id=customer_id,
                source_entity_type="customer",
                source_entity_id=customer_id,
                target_entity_type="beneficiary",
                target_entity_id="b1",
                relationship_type="transacts_with",
                action="created"
            ),
            ObservableBeneficiaryEvent(
                event_type="BENEFICIARY_ADDITION",
                event_id="e2",
                timestamp=datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc),
                customer_id=customer_id,
                beneficiary_id="b1",
                relationship_type="other",
                is_verified=False,
                action="add"
            )
        ],
        observation_window=(
            datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
        )
    )
    
    from red_team.schemas.ground_truth import AttackGroundTruth, GenerationMetadata, PlannerMetadata, EvaluationMetadata
    gt = AttackGroundTruth(
        attack_id="atk-1",
        attack_family="ACCOUNT_TAKEOVER",
        attack_difficulty="medium",
        hidden_objective="drain",
        phases_executed=[],
        linked_event_ids=["e1", "e2"],
        generation_metadata=GenerationMetadata(
            random_seed=1, generator_version="1.0", signature_version="1.0",
            provenance_registry_version="1.0", configuration_hash="a", generated_at=datetime.datetime.now(datetime.timezone.utc)
        ),
        planner_metadata=PlannerMetadata(planner_type="mock", plan_json={})
    )
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    assert any("beneficiary" in f for f in report.failures)


def test_invalid_attack_transition():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Tamper with phase order to something impossible: END -> RECONNAISSANCE
    gt.phases_executed.insert(0, AttackPhaseRecord(
        phase="END", entered_at=datetime.datetime.now(), was_optional=False
    ))
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    assert any("Transition not allowed" in f for f in report.failures)


def test_reference_unavailable():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.statistical.score == "NOT_AVAILABLE"
    assert "statistical" in report.unavailable_metrics


def test_reproducibility():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim1 = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace1, gt1 = sim1.generate_attack(plan, customer_id)
    report1 = validate_attack_realism(trace1, gt1, get_ato_signature(), world.get_state())
    
    world2 = NormalWorld(seed=1)
    world2.generate_population(n_customers=2)
    customer_id2 = list(world2.get_state().customers.keys())[0]
    sim2 = StatefulSimulator(world2.get_state(), get_ato_signature(), seed=42)
    trace2, gt2 = sim2.generate_attack(plan, customer_id2)
    report2 = validate_attack_realism(trace2, gt2, get_ato_signature(), world2.get_state())
    
    assert report1.model_dump() == report2.model_dump()


def test_artificially_bad_trace():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    from red_team.schemas.observable import ObservableTransactionEvent, ObservableRelationshipEvent
    trace = ObservableAttackTrace(
        trace_id="atk-1",
        customer_id=customer_id,
        events=[
            ObservableRelationshipEvent(
                event_type="RELATIONSHIP_CHANGE",
                event_id="e0",
                timestamp=datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc), # Future! (Impossible timing)
                customer_id=customer_id,
                source_entity_type="customer",
                source_entity_id=customer_id,
                target_entity_type="beneficiary",
                target_entity_id="b1",
                relationship_type="transacts_with",
                action="created"
            ),
            ObservableTransactionEvent(
                event_type="TRANSACTION",
                event_id="e1",
                timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc), # Past! (unordered)
                customer_id=customer_id,
                account_id="nonexistent-account", # Invalid entity
                amount=Decimal("100.00"),
                currency="USD",
                transaction_type="transfer",
                channel="online",
                transaction_status="completed"
            ),
            ObservableTransactionEvent(
                event_type="TRANSACTION",
                event_id="e2",
                timestamp=datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc),
                customer_id=customer_id,
                account_id=list(world.get_state().accounts.keys())[0], # Valid account
                amount=Decimal("999999999.00"), # Overspend
                currency="USD",
                transaction_type="transfer",
                channel="online",
                transaction_status="completed"
            )
        ],
        observation_window=(
            datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc)
        )
    )
    
    from red_team.schemas.ground_truth import AttackGroundTruth, GenerationMetadata, PlannerMetadata, EvaluationMetadata
    gt = AttackGroundTruth(
        attack_id="atk-1",
        attack_family="ACCOUNT_TAKEOVER",
        attack_difficulty="medium",
        hidden_objective="drain",
        phases_executed=[
            AttackPhaseRecord(
                phase="END", entered_at=datetime.datetime.now(datetime.timezone.utc), was_optional=False
            ),
            AttackPhaseRecord(
                phase="RECONNAISSANCE", entered_at=datetime.datetime.now(datetime.timezone.utc), was_optional=False
            )
        ],
        linked_event_ids=["e0", "e1", "e2"],
        generation_metadata=GenerationMetadata(
            random_seed=1, generator_version="1.0", signature_version="1.0",
            provenance_registry_version="1.0", configuration_hash="a", generated_at=datetime.datetime.now(datetime.timezone.utc)
        ),
        planner_metadata=PlannerMetadata(planner_type="mock", plan_json={})
    )
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    
    assert report.status == "REJECTED"
    assert not report.structural.passed
    assert not report.constraint.passed
    
    failures = str(report.failures)
    assert "chronologically ordered" in failures
    assert "Invalid entity reference" in failures
    assert "Impossible balance transition" in failures
    assert "Transition not allowed" in failures
