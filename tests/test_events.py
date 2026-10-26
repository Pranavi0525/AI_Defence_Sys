"""Stage 2.2 — Event schema tests.

Covers:
    - EventEnvelope: valid creation, missing/invalid fields, optional ID validation
    - Each payload type: valid construction, missing required fields, invalid enums
    - Discriminated union: every valid event_type/payload combo, every invalid combo
    - Serialization round-trip: Event → model_dump → model_validate
    - Reference coherence: envelope ↔ payload cross-reference matching
    - Negative tests: malformed combinations are rejected
    - No ground-truth fields in any event schema
"""

from __future__ import annotations

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
    EVENTTYPE_TO_PAYLOAD_TYPE,
    RelationshipEventPayload,
    SessionEventPayload,
    TransactionEventPayload,
)


# =========================================================================
# Constants & Factories
# =========================================================================

NOW = datetime(2025, 6, 15, 12, 0, 0)
YESTERDAY = NOW - timedelta(days=1)


def _make_envelope(event_type: EventType = EventType.TRANSACTION, **overrides) -> EventEnvelope:
    defaults = dict(
        timestamp=NOW,
        event_type=event_type,
        customer_id="cust-001",
    )
    defaults.update(overrides)
    return EventEnvelope(**defaults)


def _make_transaction(**overrides) -> Transaction:
    defaults = dict(
        account_id="acct-001",
        merchant_id="merch-001",
        amount=Decimal("42.50"),
        currency="USD",
        transaction_type="purchase",
        status="completed",
        channel="online",
        timestamp=NOW,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


def _make_transaction_payload(**overrides) -> TransactionEventPayload:
    defaults = dict(
        transaction=_make_transaction(),
        pre_balance=Decimal("1000.00"),
        post_balance=Decimal("957.50"),
    )
    defaults.update(overrides)
    return TransactionEventPayload(**defaults)


def _make_session(**overrides) -> Session:
    defaults = dict(
        customer_id="cust-001",
        device_id="dev-001",
        ip_address="192.168.1.1",
        start_time=NOW,
        end_time=NOW + timedelta(minutes=30),
        auth_method="password",
        auth_success=True,
    )
    defaults.update(overrides)
    return Session(**defaults)


def _make_session_payload(**overrides) -> SessionEventPayload:
    defaults = dict(
        session=_make_session(),
        login_attempt_count=1,
    )
    defaults.update(overrides)
    return SessionEventPayload(**defaults)


def _make_device(**overrides) -> Device:
    defaults = dict(
        device_type="mobile",
        fingerprint="fp-abc123",
        first_seen=YESTERDAY,
        last_seen=NOW,
    )
    defaults.update(overrides)
    return Device(**defaults)


def _make_device_payload(**overrides) -> DeviceEventPayload:
    defaults = dict(
        device=_make_device(),
        action="register",
    )
    defaults.update(overrides)
    return DeviceEventPayload(**defaults)


def _make_beneficiary(**overrides) -> Beneficiary:
    defaults = dict(
        name="Bob Jones",
        account_reference="IBAN-DE89370400440532013000",
        created_date=YESTERDAY,
        relationship_type="personal",
    )
    defaults.update(overrides)
    return Beneficiary(**defaults)


def _make_beneficiary_payload(**overrides) -> BeneficiaryEventPayload:
    defaults = dict(
        beneficiary=_make_beneficiary(),
        action="add",
    )
    defaults.update(overrides)
    return BeneficiaryEventPayload(**defaults)


def _make_account_context_payload(**overrides) -> AccountContextEventPayload:
    defaults = dict(
        change_type="contact_info",
        field_changed="email",
    )
    defaults.update(overrides)
    return AccountContextEventPayload(**defaults)


def _make_relationship(**overrides) -> Relationship:
    defaults = dict(
        source_entity_type="customer",
        source_entity_id="cust-001",
        target_entity_type="beneficiary",
        target_entity_id="ben-001",
        relationship_type="knows",
        established_date=YESTERDAY,
    )
    defaults.update(overrides)
    return Relationship(**defaults)


def _make_relationship_payload(**overrides) -> RelationshipEventPayload:
    defaults = dict(
        relationship=_make_relationship(),
        action="establish",
    )
    defaults.update(overrides)
    return RelationshipEventPayload(**defaults)


def _make_event(event_type: EventType, payload, **envelope_overrides) -> Event:
    envelope = _make_envelope(event_type=event_type, **envelope_overrides)
    return Event(envelope=envelope, payload=payload)


# =========================================================================
# EventEnvelope tests
# =========================================================================

class TestEventEnvelope:
    def test_valid_creation(self):
        e = _make_envelope()
        assert e.event_type == EventType.TRANSACTION
        assert e.customer_id == "cust-001"
        assert e.event_id  # auto-generated UUID
        assert e.account_id is None
        assert e.session_id is None

    def test_auto_uuid_unique(self):
        e1 = _make_envelope()
        e2 = _make_envelope()
        assert e1.event_id != e2.event_id

    def test_explicit_event_id(self):
        e = _make_envelope(event_id="custom-id-123")
        assert e.event_id == "custom-id-123"

    def test_empty_event_id_rejected(self):
        with pytest.raises(ValidationError, match="event_id"):
            _make_envelope(event_id="")

    def test_missing_timestamp_rejected(self):
        with pytest.raises(ValidationError, match="timestamp"):
            EventEnvelope(event_type=EventType.TRANSACTION, customer_id="cust-001")

    def test_missing_event_type_rejected(self):
        with pytest.raises(ValidationError, match="event_type"):
            EventEnvelope(timestamp=NOW, customer_id="cust-001")

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError, match="event_type"):
            _make_envelope(event_type="INVALID_TYPE")

    def test_empty_customer_id_rejected(self):
        with pytest.raises(ValidationError, match="customer_id"):
            _make_envelope(customer_id="")

    def test_empty_account_id_rejected(self):
        with pytest.raises(ValidationError, match="account_id"):
            _make_envelope(account_id="")

    def test_whitespace_account_id_rejected(self):
        with pytest.raises(ValidationError, match="account_id"):
            _make_envelope(account_id="   ")

    def test_empty_session_id_rejected(self):
        with pytest.raises(ValidationError, match="session_id"):
            _make_envelope(session_id="")

    def test_whitespace_session_id_rejected(self):
        with pytest.raises(ValidationError, match="session_id"):
            _make_envelope(session_id="  ")

    def test_valid_optional_ids(self):
        e = _make_envelope(account_id="acct-001", session_id="sess-001")
        assert e.account_id == "acct-001"
        assert e.session_id == "sess-001"

    def test_none_optional_ids_allowed(self):
        e = _make_envelope(account_id=None, session_id=None)
        assert e.account_id is None
        assert e.session_id is None

    def test_all_event_types_valid(self):
        for et in EventType:
            e = _make_envelope(event_type=et)
            assert e.event_type == et


# =========================================================================
# TransactionEventPayload tests
# =========================================================================

class TestTransactionEventPayload:
    def test_valid_creation(self):
        p = _make_transaction_payload()
        assert p.payload_type == "transaction"
        assert p.transaction.amount == Decimal("42.50")
        assert p.pre_balance == Decimal("1000.00")
        assert p.post_balance == Decimal("957.50")

    def test_missing_transaction_rejected(self):
        with pytest.raises(ValidationError, match="transaction"):
            TransactionEventPayload(
                pre_balance=Decimal("100.00"),
                post_balance=Decimal("50.00"),
            )

    def test_missing_pre_balance_rejected(self):
        with pytest.raises(ValidationError, match="pre_balance"):
            TransactionEventPayload(
                transaction=_make_transaction(),
                post_balance=Decimal("50.00"),
            )

    def test_missing_post_balance_rejected(self):
        with pytest.raises(ValidationError, match="post_balance"):
            TransactionEventPayload(
                transaction=_make_transaction(),
                pre_balance=Decimal("100.00"),
            )

    def test_infinite_pre_balance_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            _make_transaction_payload(pre_balance=Decimal("Infinity"))

    def test_nan_post_balance_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            _make_transaction_payload(post_balance=Decimal("NaN"))

    def test_negative_balances_allowed(self):
        """Negative balances are valid — credit accounts can have them."""
        p = _make_transaction_payload(
            pre_balance=Decimal("-100.00"),
            post_balance=Decimal("-142.50"),
        )
        assert p.pre_balance == Decimal("-100.00")

    def test_zero_balances_allowed(self):
        p = _make_transaction_payload(
            pre_balance=Decimal("0.00"),
            post_balance=Decimal("0.00"),
        )
        assert p.pre_balance == Decimal("0.00")

    def test_serialization_round_trip(self):
        p = _make_transaction_payload()
        data = p.model_dump()
        p2 = TransactionEventPayload(**data)
        assert p2.payload_type == "transaction"
        assert p2.transaction.amount == p.transaction.amount


# =========================================================================
# SessionEventPayload tests
# =========================================================================

class TestSessionEventPayload:
    def test_valid_creation(self):
        p = _make_session_payload()
        assert p.payload_type == "session"
        assert p.login_attempt_count == 1

    def test_missing_session_rejected(self):
        with pytest.raises(ValidationError, match="session"):
            SessionEventPayload(login_attempt_count=1)

    def test_negative_login_attempt_count_rejected(self):
        with pytest.raises(ValidationError, match="login_attempt_count"):
            _make_session_payload(login_attempt_count=-1)

    def test_zero_login_attempt_count_valid(self):
        p = _make_session_payload(login_attempt_count=0)
        assert p.login_attempt_count == 0

    def test_high_login_attempt_count_valid(self):
        p = _make_session_payload(login_attempt_count=100)
        assert p.login_attempt_count == 100

    def test_serialization_round_trip(self):
        p = _make_session_payload()
        data = p.model_dump()
        p2 = SessionEventPayload(**data)
        assert p2.payload_type == "session"
        assert p2.session.customer_id == p.session.customer_id


# =========================================================================
# DeviceEventPayload tests
# =========================================================================

class TestDeviceEventPayload:
    def test_valid_register(self):
        p = _make_device_payload(action="register")
        assert p.action == "register"
        assert p.payload_type == "device"

    def test_valid_change_primary_with_previous(self):
        p = _make_device_payload(action="change_primary", previous_device_id="dev-old")
        assert p.previous_device_id == "dev-old"

    def test_valid_change_primary_without_previous(self):
        """change_primary without previous_device_id is structurally valid."""
        p = _make_device_payload(action="change_primary", previous_device_id=None)
        assert p.previous_device_id is None

    def test_valid_deactivate(self):
        p = _make_device_payload(action="deactivate")
        assert p.action == "deactivate"

    def test_missing_device_rejected(self):
        with pytest.raises(ValidationError, match="device"):
            DeviceEventPayload(action="register")

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError, match="action"):
            _make_device_payload(action="destroy")

    def test_serialization_round_trip(self):
        p = _make_device_payload()
        data = p.model_dump()
        p2 = DeviceEventPayload(**data)
        assert p2.action == p.action


# =========================================================================
# BeneficiaryEventPayload tests
# =========================================================================

class TestBeneficiaryEventPayload:
    def test_valid_add(self):
        p = _make_beneficiary_payload(action="add")
        assert p.action == "add"
        assert p.payload_type == "beneficiary"

    def test_valid_remove(self):
        p = _make_beneficiary_payload(action="remove")
        assert p.action == "remove"

    def test_valid_modify(self):
        p = _make_beneficiary_payload(action="modify")
        assert p.action == "modify"

    def test_missing_beneficiary_rejected(self):
        with pytest.raises(ValidationError, match="beneficiary"):
            BeneficiaryEventPayload(action="add")

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError, match="action"):
            _make_beneficiary_payload(action="delete")

    def test_beneficiary_has_no_customer_id(self):
        """Confirm Beneficiary independence is preserved in event payloads."""
        p = _make_beneficiary_payload()
        assert "customer_id" not in Beneficiary.model_fields

    def test_serialization_round_trip(self):
        p = _make_beneficiary_payload()
        data = p.model_dump()
        p2 = BeneficiaryEventPayload(**data)
        assert p2.action == p.action


# =========================================================================
# AccountContextEventPayload tests
# =========================================================================

class TestAccountContextEventPayload:
    def test_valid_creation(self):
        p = _make_account_context_payload()
        assert p.change_type == "contact_info"
        assert p.field_changed == "email"
        assert p.payload_type == "account_context"

    def test_all_change_types_valid(self):
        for ct in ["contact_info", "security_settings", "address", "limits", "status"]:
            p = _make_account_context_payload(change_type=ct)
            assert p.change_type == ct

    def test_invalid_change_type_rejected(self):
        with pytest.raises(ValidationError, match="change_type"):
            _make_account_context_payload(change_type="attack_type")

    def test_empty_field_changed_rejected(self):
        with pytest.raises(ValidationError, match="field_changed"):
            _make_account_context_payload(field_changed="")

    def test_missing_change_type_rejected(self):
        with pytest.raises(ValidationError, match="change_type"):
            AccountContextEventPayload(field_changed="email")

    def test_missing_field_changed_rejected(self):
        with pytest.raises(ValidationError, match="field_changed"):
            AccountContextEventPayload(change_type="contact_info")

    def test_serialization_round_trip(self):
        p = _make_account_context_payload()
        data = p.model_dump()
        p2 = AccountContextEventPayload(**data)
        assert p2.change_type == p.change_type


# =========================================================================
# RelationshipEventPayload tests
# =========================================================================

class TestRelationshipEventPayload:
    def test_valid_establish(self):
        p = _make_relationship_payload(action="establish")
        assert p.action == "establish"
        assert p.payload_type == "relationship"

    def test_valid_strengthen(self):
        p = _make_relationship_payload(action="strengthen")
        assert p.action == "strengthen"

    def test_valid_weaken(self):
        p = _make_relationship_payload(action="weaken")
        assert p.action == "weaken"

    def test_valid_terminate(self):
        p = _make_relationship_payload(action="terminate")
        assert p.action == "terminate"

    def test_missing_relationship_rejected(self):
        with pytest.raises(ValidationError, match="relationship"):
            RelationshipEventPayload(action="establish")

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError, match="action"):
            _make_relationship_payload(action="destroy")

    def test_serialization_round_trip(self):
        p = _make_relationship_payload()
        data = p.model_dump()
        p2 = RelationshipEventPayload(**data)
        assert p2.action == p.action


# =========================================================================
# Event — Valid event_type/payload combinations
# =========================================================================

class TestEventValidCombinations:
    """Every valid EventType must produce a valid Event with its matching payload."""

    def test_transaction_event(self):
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id="acct-001",
        )
        assert ev.envelope.event_type == EventType.TRANSACTION
        assert ev.payload.payload_type == "transaction"

    def test_session_login_event(self):
        ev = _make_event(EventType.SESSION_LOGIN, _make_session_payload())
        assert ev.envelope.event_type == EventType.SESSION_LOGIN
        assert ev.payload.payload_type == "session"

    def test_session_logout_event(self):
        ev = _make_event(EventType.SESSION_LOGOUT, _make_session_payload())
        assert ev.envelope.event_type == EventType.SESSION_LOGOUT
        assert ev.payload.payload_type == "session"

    def test_device_registration_event(self):
        ev = _make_event(EventType.DEVICE_REGISTRATION, _make_device_payload())
        assert ev.envelope.event_type == EventType.DEVICE_REGISTRATION

    def test_device_change_event(self):
        ev = _make_event(
            EventType.DEVICE_CHANGE,
            _make_device_payload(action="change_primary", previous_device_id="dev-old"),
        )
        assert ev.envelope.event_type == EventType.DEVICE_CHANGE

    def test_beneficiary_addition_event(self):
        ev = _make_event(EventType.BENEFICIARY_ADDITION, _make_beneficiary_payload(action="add"))
        assert ev.envelope.event_type == EventType.BENEFICIARY_ADDITION

    def test_beneficiary_removal_event(self):
        ev = _make_event(EventType.BENEFICIARY_REMOVAL, _make_beneficiary_payload(action="remove"))
        assert ev.envelope.event_type == EventType.BENEFICIARY_REMOVAL

    def test_account_context_change_event(self):
        ev = _make_event(EventType.ACCOUNT_CONTEXT_CHANGE, _make_account_context_payload())
        assert ev.envelope.event_type == EventType.ACCOUNT_CONTEXT_CHANGE

    def test_relationship_change_event(self):
        ev = _make_event(EventType.RELATIONSHIP_CHANGE, _make_relationship_payload())
        assert ev.envelope.event_type == EventType.RELATIONSHIP_CHANGE


# =========================================================================
# Event — Invalid event_type/payload combinations (discriminator mismatch)
# =========================================================================

class TestEventInvalidCombinations:
    """Mismatched event_type/payload_type must be rejected."""

    def test_transaction_type_with_session_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.TRANSACTION, _make_session_payload())

    def test_session_login_type_with_transaction_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(
                EventType.SESSION_LOGIN,
                _make_transaction_payload(),
                account_id="acct-001",
            )

    def test_device_registration_with_beneficiary_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.DEVICE_REGISTRATION, _make_beneficiary_payload())

    def test_beneficiary_addition_with_relationship_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.BENEFICIARY_ADDITION, _make_relationship_payload())

    def test_account_context_with_device_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.ACCOUNT_CONTEXT_CHANGE, _make_device_payload())

    def test_relationship_change_with_account_context_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.RELATIONSHIP_CHANGE, _make_account_context_payload())

    def test_session_logout_with_device_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.SESSION_LOGOUT, _make_device_payload())

    def test_device_change_with_transaction_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(
                EventType.DEVICE_CHANGE,
                _make_transaction_payload(),
                account_id="acct-001",
            )

    def test_beneficiary_removal_with_session_payload(self):
        with pytest.raises(ValidationError, match="payload_type"):
            _make_event(EventType.BENEFICIARY_REMOVAL, _make_session_payload())


# =========================================================================
# Event — Serialization round-trip
# =========================================================================

class TestEventSerialization:
    def test_transaction_event_round_trip(self):
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id="acct-001",
        )
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.envelope.event_type == EventType.TRANSACTION
        assert ev2.payload.payload_type == "transaction"
        assert ev2.payload.transaction.amount == Decimal("42.50")

    def test_session_event_round_trip(self):
        ev = _make_event(EventType.SESSION_LOGIN, _make_session_payload())
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.envelope.event_type == EventType.SESSION_LOGIN
        assert ev2.payload.session.customer_id == "cust-001"

    def test_device_event_round_trip(self):
        ev = _make_event(EventType.DEVICE_REGISTRATION, _make_device_payload())
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.payload.device.fingerprint == "fp-abc123"

    def test_beneficiary_event_round_trip(self):
        ev = _make_event(EventType.BENEFICIARY_ADDITION, _make_beneficiary_payload())
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.payload.beneficiary.name == "Bob Jones"

    def test_account_context_event_round_trip(self):
        ev = _make_event(EventType.ACCOUNT_CONTEXT_CHANGE, _make_account_context_payload())
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.payload.change_type == "contact_info"

    def test_relationship_event_round_trip(self):
        ev = _make_event(EventType.RELATIONSHIP_CHANGE, _make_relationship_payload())
        data = ev.model_dump()
        ev2 = Event.model_validate(data)
        assert ev2.payload.action == "establish"

    def test_round_trip_from_json(self):
        """model_dump(mode='json') → model_validate should also work."""
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id="acct-001",
        )
        json_data = ev.model_dump(mode="json")
        ev2 = Event.model_validate(json_data)
        assert ev2.envelope.event_type == EventType.TRANSACTION
        assert ev2.payload.payload_type == "transaction"


# =========================================================================
# Event — Reference coherence validation
# =========================================================================

class TestEventReferenceCoherence:
    def test_transaction_account_id_matches(self):
        """Envelope account_id must match transaction.account_id when both present."""
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id="acct-001",  # matches transaction default
        )
        assert ev.payload.transaction.account_id == "acct-001"

    def test_transaction_account_id_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="account_id"):
            _make_event(
                EventType.TRANSACTION,
                _make_transaction_payload(),
                account_id="acct-WRONG",
            )

    def test_transaction_no_envelope_account_id_allowed(self):
        """If envelope doesn't specify account_id, no coherence check needed."""
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id=None,
        )
        assert ev.envelope.account_id is None

    def test_session_customer_id_matches(self):
        """Session customer_id must match envelope customer_id."""
        ev = _make_event(
            EventType.SESSION_LOGIN,
            _make_session_payload(),
            customer_id="cust-001",  # matches session default
        )
        assert ev.payload.session.customer_id == "cust-001"

    def test_session_customer_id_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="customer_id"):
            _make_event(
                EventType.SESSION_LOGIN,
                _make_session_payload(),
                customer_id="cust-WRONG",
            )

    def test_session_id_matches(self):
        """If envelope specifies session_id, it must match session.session_id."""
        session = _make_session()
        payload = _make_session_payload(session=session)
        ev = _make_event(
            EventType.SESSION_LOGIN,
            payload,
            session_id=session.session_id,
        )
        assert ev.envelope.session_id == session.session_id

    def test_session_id_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="session_id"):
            _make_event(
                EventType.SESSION_LOGIN,
                _make_session_payload(),
                session_id="sess-WRONG",
            )

    def test_session_no_envelope_session_id_allowed(self):
        """If envelope doesn't specify session_id, no coherence check needed."""
        ev = _make_event(
            EventType.SESSION_LOGIN,
            _make_session_payload(),
            session_id=None,
        )
        assert ev.envelope.session_id is None


# =========================================================================
# Event — Structural validation (empty/incomplete events)
# =========================================================================

class TestEventStructuralValidation:
    def test_transaction_without_transaction_data_rejected(self):
        """TRANSACTION event with missing transaction in payload must fail."""
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.TRANSACTION),
                payload={"payload_type": "transaction"},  # missing transaction, pre/post
            )

    def test_session_login_without_session_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.SESSION_LOGIN),
                payload={"payload_type": "session", "login_attempt_count": 1},
            )

    def test_device_registration_without_device_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.DEVICE_REGISTRATION),
                payload={"payload_type": "device", "action": "register"},
            )

    def test_beneficiary_addition_without_beneficiary_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.BENEFICIARY_ADDITION),
                payload={"payload_type": "beneficiary", "action": "add"},
            )

    def test_account_context_without_change_type_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.ACCOUNT_CONTEXT_CHANGE),
                payload={"payload_type": "account_context", "field_changed": "email"},
            )

    def test_relationship_change_without_relationship_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                envelope=_make_envelope(EventType.RELATIONSHIP_CHANGE),
                payload={"payload_type": "relationship", "action": "establish"},
            )


# =========================================================================
# EventType-to-payload_type mapping completeness
# =========================================================================

class TestEventTypeMapping:
    def test_all_event_types_have_mapping(self):
        """Every EventType must be in EVENTTYPE_TO_PAYLOAD_TYPE."""
        for et in EventType:
            assert et in EVENTTYPE_TO_PAYLOAD_TYPE, (
                f"EventType.{et.name} missing from EVENTTYPE_TO_PAYLOAD_TYPE"
            )

    def test_mapping_values_are_valid_payload_types(self):
        """Every mapped value must match a payload model's payload_type."""
        valid_payload_types = {
            "transaction", "session", "device",
            "beneficiary", "account_context", "relationship",
        }
        for et, pt in EVENTTYPE_TO_PAYLOAD_TYPE.items():
            assert pt in valid_payload_types, (
                f"EventType.{et.name} maps to invalid payload_type '{pt}'"
            )


# =========================================================================
# No ground-truth fields in event schemas
# =========================================================================

FORBIDDEN_FIELDS = frozenset({
    "attack_id", "attack_family", "attack_phase", "is_fraud",
    "ground_truth", "attacker_intent", "genai_used", "attack_type",
    "label", "hidden_objective", "planner_metadata",
    "generation_metadata", "evaluation_metadata",
})


class TestNoGroundTruthFields:
    def test_envelope_has_no_ground_truth_fields(self):
        for field_name in EventEnvelope.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"EventEnvelope contains forbidden ground-truth field: {field_name}"
            )

    def test_transaction_payload_has_no_ground_truth_fields(self):
        for field_name in TransactionEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"TransactionEventPayload contains forbidden field: {field_name}"
            )

    def test_session_payload_has_no_ground_truth_fields(self):
        for field_name in SessionEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"SessionEventPayload contains forbidden field: {field_name}"
            )

    def test_device_payload_has_no_ground_truth_fields(self):
        for field_name in DeviceEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"DeviceEventPayload contains forbidden field: {field_name}"
            )

    def test_beneficiary_payload_has_no_ground_truth_fields(self):
        for field_name in BeneficiaryEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"BeneficiaryEventPayload contains forbidden field: {field_name}"
            )

    def test_account_context_payload_has_no_ground_truth_fields(self):
        for field_name in AccountContextEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"AccountContextEventPayload contains forbidden field: {field_name}"
            )

    def test_relationship_payload_has_no_ground_truth_fields(self):
        for field_name in RelationshipEventPayload.model_fields:
            assert field_name not in FORBIDDEN_FIELDS, (
                f"RelationshipEventPayload contains forbidden field: {field_name}"
            )

    def test_serialized_event_has_no_ground_truth_fields(self):
        """Full serialized Event must not contain any ground-truth key."""
        ev = _make_event(
            EventType.TRANSACTION,
            _make_transaction_payload(),
            account_id="acct-001",
        )
        data = ev.model_dump()
        all_keys = _collect_keys_recursive(data)
        for key in all_keys:
            assert key not in FORBIDDEN_FIELDS, (
                f"Serialized Event contains forbidden key: {key}"
            )


def _collect_keys_recursive(d: dict) -> set[str]:
    """Recursively collect all keys from nested dicts/lists."""
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            keys.add(k)
            keys |= _collect_keys_recursive(v)
    elif isinstance(d, list):
        for item in d:
            keys |= _collect_keys_recursive(item)
    return keys
