"""Observable attack trace schemas — the Blue Team input contract.

Contains ONLY information visible to the future Blue Team detector.
No ground-truth fields, no attack labels, no generation metadata.

Design decisions:
    - Observable events use discriminated unions keyed on event_type,
      mirroring the internal Event model but exposing only observable fields.
    - All observable models use extra='forbid' to reject unknown fields at
      construction time, providing structural protection against leakage.
    - extract_observable() is the ONLY supported transformation from internal
      Events → ObservableAttackTrace. There is no reverse path.
    - No shared base class with AttackGroundTruth.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from red_team.schemas.events import (
    AccountContextEventPayload,
    BeneficiaryEventPayload,
    DeviceEventPayload,
    Event,
    RelationshipEventPayload,
    SessionEventPayload,
    TransactionEventPayload,
)

if TYPE_CHECKING:
    from red_team.schemas.entities import Merchant


# ---------------------------------------------------------------------------
# Observable event models — one per event category
# ---------------------------------------------------------------------------

class ObservableTransactionEvent(BaseModel):
    """Observable transaction — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["TRANSACTION"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    amount: Decimal = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    transaction_type: str = Field(..., min_length=1)
    merchant_category: str | None = None
    merchant_country: str | None = None
    beneficiary_id: str | None = Field(
        default=None,
        description=(
            "FK -> Beneficiary.beneficiary_id, observable for transfer "
            "transactions (any real bank logs the destination of a "
            "transfer it processes). None for non-transfer transaction "
            "types (purchase/withdrawal/payment/refund), which have no "
            "beneficiary. This is the field a cross-bank/graph detector "
            "needs to see a mule network's shared collector account -- "
            "previously stripped here, which is why cascade_with_graph.py "
            "had to fall back on a hand-injected synthetic overlay instead "
            "of the real Red Team MULE_NETWORK signal."
        ),
    )
    channel: str = Field(..., min_length=1)
    transaction_status: str = Field(..., min_length=1)


class ObservableSessionEvent(BaseModel):
    """Observable session event — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["SESSION_LOGIN", "SESSION_LOGOUT"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    account_id: str | None = None
    session_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    geo_country: str | None = None
    geo_city: str | None = None
    auth_method: str = Field(..., min_length=1)
    auth_success: bool
    login_attempt_count: int = Field(..., ge=0)


class ObservableDeviceEvent(BaseModel):
    """Observable device event — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["DEVICE_REGISTRATION", "DEVICE_CHANGE"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    device_type: str = Field(..., min_length=1)
    os: str | None = None
    browser: str | None = None
    fingerprint: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    previous_device_id: str | None = None


class ObservableBeneficiaryEvent(BaseModel):
    """Observable beneficiary event — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["BENEFICIARY_ADDITION", "BENEFICIARY_REMOVAL"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    beneficiary_id: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    is_verified: bool
    action: str = Field(..., min_length=1)


class ObservableAccountContextEvent(BaseModel):
    """Observable account context event — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["ACCOUNT_CONTEXT_CHANGE"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    account_id: str | None = None
    change_type: str = Field(..., min_length=1)
    field_changed: str = Field(..., min_length=1)


class ObservableRelationshipEvent(BaseModel):
    """Observable relationship event — Blue Team visible only."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["RELATIONSHIP_CHANGE"]
    event_id: str = Field(..., min_length=1)
    timestamp: datetime
    customer_id: str = Field(..., min_length=1)
    source_entity_type: str = Field(..., min_length=1)
    source_entity_id: str = Field(..., min_length=1)
    target_entity_type: str = Field(..., min_length=1)
    target_entity_id: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Discriminated union over all observable event types
# ---------------------------------------------------------------------------

ObservableEvent = Annotated[
    Union[
        ObservableTransactionEvent,
        ObservableSessionEvent,
        ObservableDeviceEvent,
        ObservableBeneficiaryEvent,
        ObservableAccountContextEvent,
        ObservableRelationshipEvent,
    ],
    Field(discriminator="event_type"),
]
"""Pydantic discriminated union of observable events, keyed on event_type."""


# ---------------------------------------------------------------------------
# Observable Attack Trace — the Blue Team contract
# ---------------------------------------------------------------------------

class ObservableAttackTrace(BaseModel):
    """The ONLY attack representation the Blue Team may receive.

    Contains observable events and metadata. No ground-truth fields.
    No attack labels. No generation metadata.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1, description="Unique trace identifier.")
    customer_id: str = Field(..., min_length=1, description="Customer this trace belongs to.")
    events: list[ObservableEvent] = Field(
        ..., min_length=1, description="Ordered observable events.",
    )
    observation_window: tuple[datetime, datetime] = Field(
        ..., description="(start, end) of the observation period.",
    )

    @model_validator(mode="after")
    def _validate_observation_window(self) -> ObservableAttackTrace:
        start, end = self.observation_window
        if end < start:
            raise ValueError(
                f"observation_window end ({end}) must be >= start ({start})"
            )
        return self


# ---------------------------------------------------------------------------
# One-way extraction: internal Events → ObservableAttackTrace
# ---------------------------------------------------------------------------

def extract_observable(
    events: list[Event],
    trace_id: str,
    merchant_lookup: dict[str, "Merchant"] | None = None,
) -> ObservableAttackTrace:
    """One-way transformation from internal Event list to ObservableAttackTrace.

    Extracts only Blue-Team-visible fields. Strips all internal metadata.

    There is NO reverse transformation (ObservableAttackTrace → Event).
    There is NO path from AttackGroundTruth → ObservableAttackTrace.
    Ground truth must be created separately.

    Args:
        events: Internal event objects from the simulation engine.
        trace_id: Unique identifier for this observable trace.
        merchant_lookup: Optional mapping of merchant_id -> Merchant, taken
            from the simulated world state. When provided, purchase
            transactions get their merchant_category/merchant_country
            populated from a real merchant record — this is information any
            bank has in its own systems (it operates the card network /
            merchant acquiring relationship), not a Blue-Team leak. When
            omitted, both fields stay None (unchanged legacy behavior).

    Returns:
        ObservableAttackTrace containing only observable information.

    Raises:
        ValueError: If events is empty or contains multiple customer IDs.
    """
    if not events:
        raise ValueError("Cannot extract observable trace from empty event list")

    customer_ids = {e.envelope.customer_id for e in events}
    if len(customer_ids) != 1:
        raise ValueError(
            f"All events must belong to the same customer, got: {customer_ids}"
        )

    customer_id = customer_ids.pop()
    observable_events: list[
        ObservableTransactionEvent
        | ObservableSessionEvent
        | ObservableDeviceEvent
        | ObservableBeneficiaryEvent
        | ObservableAccountContextEvent
        | ObservableRelationshipEvent
    ] = [_extract_single_event(e, merchant_lookup) for e in events]

    timestamps = [e.envelope.timestamp for e in events]
    observation_window = (min(timestamps), max(timestamps))

    return ObservableAttackTrace(
        trace_id=trace_id,
        customer_id=customer_id,
        events=observable_events,
        observation_window=observation_window,
    )


def _extract_single_event(
    event: Event,
    merchant_lookup: dict[str, "Merchant"] | None = None,
) -> (
    ObservableTransactionEvent
    | ObservableSessionEvent
    | ObservableDeviceEvent
    | ObservableBeneficiaryEvent
    | ObservableAccountContextEvent
    | ObservableRelationshipEvent
):
    """Map a single internal Event to its observable counterpart."""
    env = event.envelope
    payload = event.payload

    if isinstance(payload, TransactionEventPayload):
        tx = payload.transaction
        merchant_category = None
        merchant_country = None
        if merchant_lookup is not None and tx.merchant_id:
            merchant = merchant_lookup.get(tx.merchant_id)
            if merchant is not None:
                merchant_category = merchant.category
                merchant_country = merchant.country
        return ObservableTransactionEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            account_id=tx.account_id,
            amount=tx.amount,
            currency=tx.currency,
            transaction_type=tx.transaction_type,
            merchant_category=merchant_category,
            merchant_country=merchant_country,
            beneficiary_id=tx.beneficiary_id,
            channel=tx.channel,
            transaction_status=tx.status,
        )

    if isinstance(payload, SessionEventPayload):
        sess = payload.session
        return ObservableSessionEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            account_id=env.account_id,
            session_id=sess.session_id,
            device_id=sess.device_id,
            geo_country=sess.geo_country,
            geo_city=sess.geo_city,
            auth_method=sess.auth_method,
            auth_success=sess.auth_success,
            login_attempt_count=payload.login_attempt_count,
        )

    if isinstance(payload, DeviceEventPayload):
        dev = payload.device
        return ObservableDeviceEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            device_id=dev.device_id,
            device_type=dev.device_type,
            os=dev.os,
            browser=dev.browser,
            fingerprint=dev.fingerprint,
            action=payload.action,
            previous_device_id=payload.previous_device_id,
        )

    if isinstance(payload, BeneficiaryEventPayload):
        ben = payload.beneficiary
        return ObservableBeneficiaryEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            beneficiary_id=ben.beneficiary_id,
            relationship_type=ben.relationship_type,
            is_verified=ben.is_verified,
            action=payload.action,
        )

    if isinstance(payload, AccountContextEventPayload):
        return ObservableAccountContextEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            account_id=env.account_id,
            change_type=payload.change_type,
            field_changed=payload.field_changed,
        )

    if isinstance(payload, RelationshipEventPayload):
        rel = payload.relationship
        return ObservableRelationshipEvent(
            event_type=env.event_type.value,
            event_id=env.event_id,
            timestamp=env.timestamp,
            customer_id=env.customer_id,
            source_entity_type=rel.source_entity_type,
            source_entity_id=rel.source_entity_id,
            target_entity_type=rel.target_entity_type,
            target_entity_id=rel.target_entity_id,
            relationship_type=rel.relationship_type,
            action=payload.action,
        )

    raise ValueError(f"Unknown payload type: {type(payload).__name__}")
