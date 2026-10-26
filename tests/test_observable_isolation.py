"""Stage 2.3 — Observable / Ground-Truth isolation tests.

Proves structurally that:
    1. ObservableAttackTrace serialization contains ZERO ground-truth fields
    2. AttackGroundTruth metadata cannot leak into observable models
    3. extract_observable() strips all internal metadata
    4. Observable events use discriminated unions and reject mismatches
    5. Forbidden fields are rejected at construction (extra='forbid')
    6. No shared base class between Observable and GroundTruth
    7. No reverse transformation path exists
    8. Serialization round-trips preserve observable data exactly
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from red_team.schemas.entities import (
    Beneficiary,
    Device,
    Relationship,
    Session,
    Transaction,
)
from red_team.schemas.events import (
    AccountContextEventPayload,
    BeneficiaryEventPayload,
    DeviceEventPayload,
    Event,
    EventEnvelope,
    EventType,
    RelationshipEventPayload,
    SessionEventPayload,
    TransactionEventPayload,
)
from red_team.schemas.ground_truth import (
    AttackGroundTruth,
    AttackPhaseRecord,
    EvaluationMetadata,
    GenerationMetadata,
    PlannerMetadata,
)
from red_team.schemas.observable import (
    ObservableAccountContextEvent,
    ObservableAttackTrace,
    ObservableBeneficiaryEvent,
    ObservableDeviceEvent,
    ObservableRelationshipEvent,
    ObservableSessionEvent,
    ObservableTransactionEvent,
    extract_observable,
)


# =========================================================================
# Forbidden fields — MUST NEVER appear in observable output
# =========================================================================

FORBIDDEN_FIELDS = frozenset({
    "attack_id", "attack_family", "attack_phase", "is_fraud", "label",
    "hidden_objective", "attacker_intent", "genai_used", "attack_type",
    "ground_truth", "planner_metadata", "generation_metadata",
    "evaluation_metadata", "random_seed", "configuration_hash",
    "signature_version", "provenance_registry_version", "model_version",
    "prompt_hash", "realism_score", "novelty_score",
})


# =========================================================================
# Constants & Factories
# =========================================================================

NOW = datetime(2025, 6, 15, 12, 0, 0)
EARLIER = NOW - timedelta(hours=2)
LATER = NOW + timedelta(hours=1)
YESTERDAY = NOW - timedelta(days=1)


def _make_internal_transaction_event(**env_overrides) -> Event:
    tx = Transaction(
        account_id="acct-001", merchant_id="merch-001",
        amount=Decimal("42.50"), currency="USD",
        transaction_type="purchase", status="completed",
        channel="online", timestamp=NOW,
    )
    envelope_kw = dict(
        timestamp=NOW, event_type=EventType.TRANSACTION,
        customer_id="cust-001", account_id="acct-001",
    )
    envelope_kw.update(env_overrides)
    return Event(
        envelope=EventEnvelope(**envelope_kw),
        payload=TransactionEventPayload(
            transaction=tx,
            pre_balance=Decimal("1000.00"),
            post_balance=Decimal("957.50"),
        ),
    )


def _make_internal_session_event(
    event_type: EventType = EventType.SESSION_LOGIN, **env_overrides,
) -> Event:
    sess = Session(
        customer_id="cust-001", device_id="dev-001",
        ip_address="10.0.0.1", geo_country="US", geo_city="NYC",
        start_time=NOW, auth_method="password", auth_success=True,
    )
    envelope_kw = dict(
        timestamp=NOW, event_type=event_type, customer_id="cust-001",
    )
    envelope_kw.update(env_overrides)
    return Event(
        envelope=EventEnvelope(**envelope_kw),
        payload=SessionEventPayload(session=sess, login_attempt_count=1),
    )


def _make_internal_device_event(
    event_type: EventType = EventType.DEVICE_REGISTRATION,
) -> Event:
    dev = Device(
        device_type="mobile", os="iOS 17", fingerprint="fp-abc",
        first_seen=YESTERDAY, last_seen=NOW,
    )
    return Event(
        envelope=EventEnvelope(
            timestamp=NOW, event_type=event_type, customer_id="cust-001",
        ),
        payload=DeviceEventPayload(device=dev, action="register"),
    )


def _make_internal_beneficiary_event(
    event_type: EventType = EventType.BENEFICIARY_ADDITION,
) -> Event:
    ben = Beneficiary(
        name="Bob", account_reference="IBAN-123",
        created_date=YESTERDAY, relationship_type="personal", is_verified=True,
    )
    return Event(
        envelope=EventEnvelope(
            timestamp=NOW, event_type=event_type, customer_id="cust-001",
        ),
        payload=BeneficiaryEventPayload(beneficiary=ben, action="add"),
    )


def _make_internal_account_context_event() -> Event:
    return Event(
        envelope=EventEnvelope(
            timestamp=NOW, event_type=EventType.ACCOUNT_CONTEXT_CHANGE,
            customer_id="cust-001", account_id="acct-001",
        ),
        payload=AccountContextEventPayload(
            change_type="contact_info", field_changed="email",
        ),
    )


def _make_internal_relationship_event() -> Event:
    rel = Relationship(
        source_entity_type="customer", source_entity_id="cust-001",
        target_entity_type="beneficiary", target_entity_id="ben-001",
        relationship_type="knows", established_date=YESTERDAY,
    )
    return Event(
        envelope=EventEnvelope(
            timestamp=NOW, event_type=EventType.RELATIONSHIP_CHANGE,
            customer_id="cust-001",
        ),
        payload=RelationshipEventPayload(relationship=rel, action="establish"),
    )


def _make_ground_truth(linked_ids: list[str] | None = None) -> AttackGroundTruth:
    return AttackGroundTruth(
        attack_id="atk-001",
        attack_family="account_takeover",
        attack_difficulty="medium",
        hidden_objective="Drain account via new beneficiary",
        phases_executed=[
            AttackPhaseRecord(
                phase="account_access", entered_at=EARLIER,
                exited_at=NOW, transition_to="exploitation", was_optional=False,
            ),
            AttackPhaseRecord(
                phase="exploitation", entered_at=NOW,
                exited_at=LATER, transition_to=None, was_optional=False,
            ),
        ],
        linked_event_ids=linked_ids or ["evt-001", "evt-002"],
        generation_metadata=GenerationMetadata(
            random_seed=42, generator_version="0.1.0",
            signature_version="ato-1.0",
            provenance_registry_version="reg-1.0",
            configuration_hash="sha256-abc", generated_at=NOW,
        ),
        planner_metadata=PlannerMetadata(
            planner_type="mock",
            plan_json={"phases": ["access", "exploit"]},
        ),
        evaluation_metadata=EvaluationMetadata(structural_valid=True),
    )


def _make_all_event_types() -> list[Event]:
    """Create one internal event of each type, all for cust-001."""
    return [
        _make_internal_transaction_event(timestamp=NOW),
        _make_internal_session_event(EventType.SESSION_LOGIN, timestamp=NOW + timedelta(minutes=1)),
        _make_internal_session_event(EventType.SESSION_LOGOUT, timestamp=NOW + timedelta(minutes=2)),
        _make_internal_device_event(EventType.DEVICE_REGISTRATION),
        _make_internal_device_event(EventType.DEVICE_CHANGE),
        _make_internal_beneficiary_event(EventType.BENEFICIARY_ADDITION),
        _make_internal_beneficiary_event(EventType.BENEFICIARY_REMOVAL),
        _make_internal_account_context_event(),
        _make_internal_relationship_event(),
    ]


# =========================================================================
# Helpers
# =========================================================================

def _collect_keys_recursive(obj) -> set[str]:
    """Recursively collect all dict keys from a nested structure."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _collect_keys_recursive(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys |= _collect_keys_recursive(item)
    return keys


# =========================================================================
# Observable event type construction
# =========================================================================

class TestObservableTransactionEvent:
    def test_valid_construction(self):
        e = ObservableTransactionEvent(
            event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
            customer_id="cust-001", account_id="acct-001",
            amount=Decimal("50.00"), currency="USD",
            transaction_type="purchase", channel="online",
            transaction_status="completed",
        )
        assert e.event_type == "TRANSACTION"
        assert e.amount == Decimal("50.00")

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001",
                # missing account_id, amount, currency, etc.
            )

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001", account_id="acct-001",
                amount=Decimal("50.00"), currency="USD",
                transaction_type="purchase", channel="online",
                transaction_status="completed",
                attack_id="SHOULD_FAIL",
            )


class TestObservableSessionEvent:
    def test_valid_login(self):
        e = ObservableSessionEvent(
            event_type="SESSION_LOGIN", event_id="evt-1", timestamp=NOW,
            customer_id="cust-001", session_id="sess-001",
            device_id="dev-001", auth_method="password",
            auth_success=True, login_attempt_count=1,
        )
        assert e.event_type == "SESSION_LOGIN"

    def test_valid_logout(self):
        e = ObservableSessionEvent(
            event_type="SESSION_LOGOUT", event_id="evt-1", timestamp=NOW,
            customer_id="cust-001", session_id="sess-001",
            device_id="dev-001", auth_method="password",
            auth_success=True, login_attempt_count=0,
        )
        assert e.event_type == "SESSION_LOGOUT"

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableSessionEvent(
                event_type="SESSION_LOGIN", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001", session_id="sess-001",
                device_id="dev-001", auth_method="password",
                auth_success=True, login_attempt_count=1,
                is_fraud=True,
            )


class TestObservableDeviceEvent:
    def test_valid_registration(self):
        e = ObservableDeviceEvent(
            event_type="DEVICE_REGISTRATION", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            device_id="dev-001", device_type="mobile",
            fingerprint="fp-123", action="register",
        )
        assert e.event_type == "DEVICE_REGISTRATION"

    def test_valid_change(self):
        e = ObservableDeviceEvent(
            event_type="DEVICE_CHANGE", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            device_id="dev-002", device_type="desktop",
            fingerprint="fp-456", action="change_primary",
            previous_device_id="dev-001",
        )
        assert e.previous_device_id == "dev-001"

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableDeviceEvent(
                event_type="DEVICE_REGISTRATION", event_id="evt-1",
                timestamp=NOW, customer_id="cust-001",
                device_id="dev-001", device_type="mobile",
                fingerprint="fp-123", action="register",
                attacker_intent="credential_stuffing",
            )


class TestObservableBeneficiaryEvent:
    def test_valid_addition(self):
        e = ObservableBeneficiaryEvent(
            event_type="BENEFICIARY_ADDITION", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            beneficiary_id="ben-001", relationship_type="personal",
            is_verified=True, action="add",
        )
        assert e.event_type == "BENEFICIARY_ADDITION"

    def test_valid_removal(self):
        e = ObservableBeneficiaryEvent(
            event_type="BENEFICIARY_REMOVAL", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            beneficiary_id="ben-001", relationship_type="business",
            is_verified=False, action="remove",
        )
        assert e.action == "remove"

    def test_no_customer_ownership_in_beneficiary(self):
        """Beneficiary remains independent — no customer_id ownership field."""
        assert "owner_customer_id" not in ObservableBeneficiaryEvent.model_fields

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableBeneficiaryEvent(
                event_type="BENEFICIARY_ADDITION", event_id="evt-1",
                timestamp=NOW, customer_id="cust-001",
                beneficiary_id="ben-001", relationship_type="personal",
                is_verified=True, action="add",
                ground_truth={"attack": "ato"},
            )


class TestObservableAccountContextEvent:
    def test_valid_construction(self):
        e = ObservableAccountContextEvent(
            event_type="ACCOUNT_CONTEXT_CHANGE", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            change_type="security_settings", field_changed="mfa_enabled",
        )
        assert e.change_type == "security_settings"

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableAccountContextEvent(
                event_type="ACCOUNT_CONTEXT_CHANGE", event_id="evt-1",
                timestamp=NOW, customer_id="cust-001",
                change_type="contact_info", field_changed="email",
                attack_phase="account_modification",
            )


class TestObservableRelationshipEvent:
    def test_valid_construction(self):
        e = ObservableRelationshipEvent(
            event_type="RELATIONSHIP_CHANGE", event_id="evt-1",
            timestamp=NOW, customer_id="cust-001",
            source_entity_type="customer", source_entity_id="cust-001",
            target_entity_type="beneficiary", target_entity_id="ben-001",
            relationship_type="knows", action="establish",
        )
        assert e.action == "establish"

    def test_forbidden_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ObservableRelationshipEvent(
                event_type="RELATIONSHIP_CHANGE", event_id="evt-1",
                timestamp=NOW, customer_id="cust-001",
                source_entity_type="customer", source_entity_id="cust-001",
                target_entity_type="beneficiary", target_entity_id="ben-001",
                relationship_type="knows", action="establish",
                label="fraud",
            )


# =========================================================================
# ObservableAttackTrace construction & validation
# =========================================================================

class TestObservableAttackTrace:
    def test_valid_construction(self):
        events = [
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001", account_id="acct-001",
                amount=Decimal("50.00"), currency="USD",
                transaction_type="purchase", channel="online",
                transaction_status="completed",
            ),
        ]
        trace = ObservableAttackTrace(
            trace_id="trace-001", customer_id="cust-001",
            events=events, observation_window=(NOW, NOW),
        )
        assert trace.trace_id == "trace-001"
        assert len(trace.events) == 1

    def test_empty_events_rejected(self):
        with pytest.raises(ValidationError):
            ObservableAttackTrace(
                trace_id="trace-001", customer_id="cust-001",
                events=[], observation_window=(NOW, NOW),
            )

    def test_invalid_observation_window_rejected(self):
        events = [
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001", account_id="acct-001",
                amount=Decimal("50.00"), currency="USD",
                transaction_type="purchase", channel="online",
                transaction_status="completed",
            ),
        ]
        with pytest.raises(ValidationError, match="observation_window"):
            ObservableAttackTrace(
                trace_id="trace-001", customer_id="cust-001",
                events=events, observation_window=(LATER, EARLIER),
            )

    def test_forbidden_field_on_trace_rejected(self):
        events = [
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="evt-1", timestamp=NOW,
                customer_id="cust-001", account_id="acct-001",
                amount=Decimal("50.00"), currency="USD",
                transaction_type="purchase", channel="online",
                transaction_status="completed",
            ),
        ]
        with pytest.raises(ValidationError, match="extra"):
            ObservableAttackTrace(
                trace_id="trace-001", customer_id="cust-001",
                events=events, observation_window=(NOW, NOW),
                attack_id="SHOULD_FAIL",
            )

    def test_discriminated_union_resolves_all_types(self):
        """Trace accepts all 6 observable event types via discriminated union."""
        events = [
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="e1", timestamp=NOW,
                customer_id="c1", account_id="a1", amount=Decimal("10"),
                currency="USD", transaction_type="purchase",
                channel="online", transaction_status="completed",
            ),
            ObservableSessionEvent(
                event_type="SESSION_LOGIN", event_id="e2", timestamp=NOW,
                customer_id="c1", session_id="s1", device_id="d1",
                auth_method="mfa", auth_success=True, login_attempt_count=1,
            ),
            ObservableDeviceEvent(
                event_type="DEVICE_REGISTRATION", event_id="e3",
                timestamp=NOW, customer_id="c1", device_id="d1",
                device_type="mobile", fingerprint="fp1", action="register",
            ),
            ObservableBeneficiaryEvent(
                event_type="BENEFICIARY_ADDITION", event_id="e4",
                timestamp=NOW, customer_id="c1", beneficiary_id="b1",
                relationship_type="personal", is_verified=True, action="add",
            ),
            ObservableAccountContextEvent(
                event_type="ACCOUNT_CONTEXT_CHANGE", event_id="e5",
                timestamp=NOW, customer_id="c1",
                change_type="address", field_changed="city",
            ),
            ObservableRelationshipEvent(
                event_type="RELATIONSHIP_CHANGE", event_id="e6",
                timestamp=NOW, customer_id="c1",
                source_entity_type="customer", source_entity_id="c1",
                target_entity_type="beneficiary", target_entity_id="b1",
                relationship_type="knows", action="establish",
            ),
        ]
        trace = ObservableAttackTrace(
            trace_id="t1", customer_id="c1",
            events=events, observation_window=(NOW, NOW),
        )
        assert len(trace.events) == 6
        assert isinstance(trace.events[0], ObservableTransactionEvent)
        assert isinstance(trace.events[1], ObservableSessionEvent)
        assert isinstance(trace.events[2], ObservableDeviceEvent)
        assert isinstance(trace.events[3], ObservableBeneficiaryEvent)
        assert isinstance(trace.events[4], ObservableAccountContextEvent)
        assert isinstance(trace.events[5], ObservableRelationshipEvent)


# =========================================================================
# Ground-truth construction
# =========================================================================

class TestAttackGroundTruth:
    def test_valid_construction(self):
        gt = _make_ground_truth()
        assert gt.attack_id == "atk-001"
        assert gt.attack_family == "account_takeover"
        assert gt.attack_difficulty == "medium"
        assert len(gt.phases_executed) == 2
        assert gt.generation_metadata.random_seed == 42

    def test_evaluation_metadata_optional(self):
        gt = _make_ground_truth()
        gt2 = gt.model_copy(update={"evaluation_metadata": None})
        assert gt2.evaluation_metadata is None

    def test_planner_metadata_present(self):
        gt = _make_ground_truth()
        assert gt.planner_metadata.planner_type == "mock"
        assert "phases" in gt.planner_metadata.plan_json

    def test_linked_event_ids_are_references_only(self):
        gt = _make_ground_truth(linked_ids=["evt-1", "evt-2", "evt-3"])
        assert gt.linked_event_ids == ["evt-1", "evt-2", "evt-3"]
        # IDs only — no observable event payloads embedded
        for eid in gt.linked_event_ids:
            assert isinstance(eid, str)

    def test_serialization_round_trip(self):
        gt = _make_ground_truth()
        data = gt.model_dump()
        gt2 = AttackGroundTruth.model_validate(data)
        assert gt2.attack_id == gt.attack_id
        assert gt2.generation_metadata.random_seed == 42


# =========================================================================
# No shared base class
# =========================================================================

class TestNoSharedBaseClass:
    def test_observable_and_ground_truth_have_no_common_base(self):
        """ObservableAttackTrace and AttackGroundTruth share no custom base."""
        obs_bases = set(ObservableAttackTrace.__mro__)
        gt_bases = set(AttackGroundTruth.__mro__)
        common = obs_bases & gt_bases
        # Only object and BaseModel (Pydantic) should be shared
        for cls in common:
            assert cls.__module__ in (
                "builtins", "pydantic.main", "pydantic._internal._model_construction",
                "abc",
            ) or cls is object, (
                f"Unexpected shared base class: {cls.__name__} from {cls.__module__}"
            )

    def test_observable_does_not_import_ground_truth(self):
        """observable.py must not import anything from ground_truth.py."""
        import red_team.schemas.observable as obs_module
        source = inspect.getsource(obs_module)
        assert "ground_truth" not in source, (
            "observable.py must not import from ground_truth"
        )


# =========================================================================
# extract_observable — one-way transformation
# =========================================================================

class TestExtractObservable:
    def test_extract_transaction_event(self):
        events = [_make_internal_transaction_event()]
        trace = extract_observable(events, trace_id="t-001")
        assert trace.trace_id == "t-001"
        assert trace.customer_id == "cust-001"
        assert len(trace.events) == 1
        obs = trace.events[0]
        assert isinstance(obs, ObservableTransactionEvent)
        assert obs.amount == Decimal("42.50")
        assert obs.channel == "online"
        assert obs.transaction_status == "completed"

    def test_extract_session_login(self):
        events = [_make_internal_session_event(EventType.SESSION_LOGIN)]
        trace = extract_observable(events, trace_id="t-002")
        obs = trace.events[0]
        assert isinstance(obs, ObservableSessionEvent)
        assert obs.event_type == "SESSION_LOGIN"
        assert obs.auth_method == "password"
        assert obs.login_attempt_count == 1

    def test_extract_session_logout(self):
        events = [_make_internal_session_event(EventType.SESSION_LOGOUT)]
        trace = extract_observable(events, trace_id="t-003")
        obs = trace.events[0]
        assert isinstance(obs, ObservableSessionEvent)
        assert obs.event_type == "SESSION_LOGOUT"

    def test_extract_device_event(self):
        events = [_make_internal_device_event()]
        trace = extract_observable(events, trace_id="t-004")
        obs = trace.events[0]
        assert isinstance(obs, ObservableDeviceEvent)
        assert obs.device_type == "mobile"
        assert obs.fingerprint == "fp-abc"

    def test_extract_beneficiary_event(self):
        events = [_make_internal_beneficiary_event()]
        trace = extract_observable(events, trace_id="t-005")
        obs = trace.events[0]
        assert isinstance(obs, ObservableBeneficiaryEvent)
        assert obs.relationship_type == "personal"
        assert obs.is_verified is True

    def test_extract_account_context_event(self):
        events = [_make_internal_account_context_event()]
        trace = extract_observable(events, trace_id="t-006")
        obs = trace.events[0]
        assert isinstance(obs, ObservableAccountContextEvent)
        assert obs.change_type == "contact_info"

    def test_extract_relationship_event(self):
        events = [_make_internal_relationship_event()]
        trace = extract_observable(events, trace_id="t-007")
        obs = trace.events[0]
        assert isinstance(obs, ObservableRelationshipEvent)
        assert obs.relationship_type == "knows"
        assert obs.action == "establish"

    def test_extract_all_event_types(self):
        """Extract a mixed trace with all 9 event types."""
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-all")
        assert len(trace.events) == 9
        types_seen = {type(e).__name__ for e in trace.events}
        assert "ObservableTransactionEvent" in types_seen
        assert "ObservableSessionEvent" in types_seen
        assert "ObservableDeviceEvent" in types_seen
        assert "ObservableBeneficiaryEvent" in types_seen
        assert "ObservableAccountContextEvent" in types_seen
        assert "ObservableRelationshipEvent" in types_seen

    def test_extract_empty_events_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            extract_observable([], trace_id="t-fail")

    def test_extract_mixed_customers_rejected(self):
        ev1 = _make_internal_transaction_event(customer_id="cust-001")
        ev2 = _make_internal_transaction_event(customer_id="cust-002")
        with pytest.raises(ValueError, match="same customer"):
            extract_observable([ev1, ev2], trace_id="t-fail")

    def test_observation_window_computed(self):
        ev1 = _make_internal_transaction_event(timestamp=EARLIER)
        ev2 = _make_internal_transaction_event(timestamp=LATER)
        trace = extract_observable([ev1, ev2], trace_id="t-win")
        assert trace.observation_window == (EARLIER, LATER)


# =========================================================================
# Serialization isolation — forbidden fields NEVER in output
# =========================================================================

class TestSerializationIsolation:
    def test_observable_trace_model_dump_no_forbidden_fields(self):
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-ser")
        data = trace.model_dump()
        all_keys = _collect_keys_recursive(data)
        violations = all_keys & FORBIDDEN_FIELDS
        assert violations == set(), (
            f"Forbidden fields found in model_dump(): {violations}"
        )

    def test_observable_trace_model_dump_json_no_forbidden_fields(self):
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-json")
        json_str = trace.model_dump_json()
        for field in FORBIDDEN_FIELDS:
            assert f'"{field}"' not in json_str, (
                f"Forbidden field '{field}' found in model_dump_json()"
            )

    def test_ground_truth_fields_absent_from_observable(self):
        """Construct ground truth, then verify its fields don't leak."""
        gt = _make_ground_truth()
        gt_data = gt.model_dump()
        gt_keys = _collect_keys_recursive(gt_data)

        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-leak")
        obs_data = trace.model_dump()
        obs_keys = _collect_keys_recursive(obs_data)

        # Ground-truth-specific keys must not appear in observable
        gt_only = gt_keys - obs_keys
        # These are expected ground-truth keys that should never be in observable
        expected_gt_only = {
            "attack_id", "attack_family", "attack_difficulty",
            "hidden_objective", "phases_executed", "linked_event_ids",
            "generation_metadata", "planner_metadata", "evaluation_metadata",
            "random_seed", "generator_version", "signature_version",
            "provenance_registry_version", "configuration_hash",
            "generated_at", "planner_type", "plan_json", "model_version",
            "prompt_hash", "structural_valid", "rejection_reason",
            "phase", "entered_at", "exited_at", "transition_to",
            "was_optional",
        }
        for key in expected_gt_only:
            assert key not in obs_keys, (
                f"Ground-truth key '{key}' found in observable output"
            )

    def test_each_observable_event_serialized_cleanly(self):
        """Every individual observable event's serialization is clean."""
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-each")
        for obs_event in trace.events:
            data = obs_event.model_dump()
            keys = _collect_keys_recursive(data)
            violations = keys & FORBIDDEN_FIELDS
            assert violations == set(), (
                f"{type(obs_event).__name__} contains forbidden: {violations}"
            )


# =========================================================================
# No reverse path
# =========================================================================

class TestNoReversePath:
    def test_no_observable_to_ground_truth_function(self):
        """No public function converts ObservableAttackTrace → AttackGroundTruth."""
        import red_team.schemas.observable as obs_mod
        import red_team.schemas.ground_truth as gt_mod

        for module in (obs_mod, gt_mod):
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # extract_observable goes Events → Observable (allowed)
                if name == "extract_observable":
                    continue
                # No function should take ObservableAttackTrace and return GroundTruth
                sig = inspect.signature(obj)
                return_annotation = sig.return_annotation
                assert return_annotation is not AttackGroundTruth, (
                    f"Function {name} returns AttackGroundTruth from observable"
                )

    def test_extract_observable_does_not_return_ground_truth(self):
        events = [_make_internal_transaction_event()]
        result = extract_observable(events, trace_id="t-type")
        assert isinstance(result, ObservableAttackTrace)
        assert not isinstance(result, AttackGroundTruth)


# =========================================================================
# Forbidden field injection tests
# =========================================================================

class TestForbiddenFieldInjection:
    """Attempt to inject ground-truth fields into observable models."""

    @pytest.mark.parametrize("field,value", [
        ("attack_id", "atk-001"),
        ("attack_family", "ato"),
        ("attack_phase", "exploitation"),
        ("is_fraud", True),
        ("label", "fraud"),
        ("hidden_objective", "drain"),
        ("attacker_intent", "steal"),
        ("genai_used", True),
        ("attack_type", "ato"),
        ("ground_truth", {}),
        ("planner_metadata", {}),
        ("generation_metadata", {}),
        ("evaluation_metadata", {}),
        ("random_seed", 42),
        ("realism_score", 0.9),
        ("novelty_score", 0.8),
    ])
    def test_forbidden_field_rejected_on_transaction(self, field, value):
        with pytest.raises(ValidationError):
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="e1", timestamp=NOW,
                customer_id="c1", account_id="a1", amount=Decimal("10"),
                currency="USD", transaction_type="purchase",
                channel="online", transaction_status="completed",
                **{field: value},
            )

    @pytest.mark.parametrize("field,value", [
        ("attack_id", "atk-001"),
        ("is_fraud", True),
        ("hidden_objective", "drain"),
        ("ground_truth", {}),
        ("realism_score", 0.9),
    ])
    def test_forbidden_field_rejected_on_trace(self, field, value):
        events = [
            ObservableTransactionEvent(
                event_type="TRANSACTION", event_id="e1", timestamp=NOW,
                customer_id="c1", account_id="a1", amount=Decimal("10"),
                currency="USD", transaction_type="purchase",
                channel="online", transaction_status="completed",
            ),
        ]
        with pytest.raises(ValidationError):
            ObservableAttackTrace(
                trace_id="t1", customer_id="c1",
                events=events, observation_window=(NOW, NOW),
                **{field: value},
            )


# =========================================================================
# Serialization round-trip
# =========================================================================

class TestSerializationRoundTrip:
    def test_observable_trace_round_trip(self):
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-rt")
        data = trace.model_dump()
        trace2 = ObservableAttackTrace.model_validate(data)
        assert trace2.trace_id == trace.trace_id
        assert trace2.customer_id == trace.customer_id
        assert len(trace2.events) == len(trace.events)
        assert trace2.observation_window == trace.observation_window

    def test_observable_trace_json_round_trip(self):
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-jrt")
        json_data = trace.model_dump(mode="json")
        trace2 = ObservableAttackTrace.model_validate(json_data)
        assert trace2.trace_id == trace.trace_id
        assert len(trace2.events) == len(trace.events)

    def test_individual_event_round_trips(self):
        events = _make_all_event_types()
        trace = extract_observable(events, trace_id="t-iert")
        for obs_event in trace.events:
            data = obs_event.model_dump()
            cls = type(obs_event)
            restored = cls.model_validate(data)
            assert restored.event_id == obs_event.event_id
            assert restored.timestamp == obs_event.timestamp

    def test_ground_truth_round_trip(self):
        gt = _make_ground_truth()
        data = gt.model_dump()
        gt2 = AttackGroundTruth.model_validate(data)
        assert gt2.attack_id == gt.attack_id
        assert gt2.generation_metadata.random_seed == gt.generation_metadata.random_seed
