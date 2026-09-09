"""Event schemas for the Red Team AI payment world event stream.

Implements a strongly-typed event model with discriminated union payloads.
Each event wraps Stage 2.1 entity models into a timestamped, typed event
with structural validation.

Design decisions:
    - EventType enum defines all supported event categories.
    - Each payload type carries a `payload_type` Literal field used as the
      Pydantic discriminated union discriminator.
    - The Event model validates that envelope.event_type agrees with
      payload.payload_type via EVENTTYPE_TO_PAYLOAD_TYPE mapping.
    - Reference coherence (e.g., envelope.account_id matches
      payload.transaction.account_id) is validated where cross-references
      are available.
    - No simulation clock or world-level validation — only local structural
      correctness.
    - No ground-truth / attack labels anywhere in the event model.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

from red_team.schemas.entities import (
    Beneficiary,
    Device,
    Relationship,
    Session,
    Transaction,
)
from red_team.schemas.id_generator import generate_id as _default_uuid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# _default_uuid() is the same shared, seedable generator entities.py uses --
# see red_team/schemas/id_generator.py.


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """All supported event types in the payment world event stream."""

    TRANSACTION = "TRANSACTION"
    SESSION_LOGIN = "SESSION_LOGIN"
    SESSION_LOGOUT = "SESSION_LOGOUT"
    DEVICE_REGISTRATION = "DEVICE_REGISTRATION"
    DEVICE_CHANGE = "DEVICE_CHANGE"
    BENEFICIARY_ADDITION = "BENEFICIARY_ADDITION"
    BENEFICIARY_REMOVAL = "BENEFICIARY_REMOVAL"
    ACCOUNT_CONTEXT_CHANGE = "ACCOUNT_CONTEXT_CHANGE"
    RELATIONSHIP_CHANGE = "RELATIONSHIP_CHANGE"


# ---------------------------------------------------------------------------
# Event Envelope
# ---------------------------------------------------------------------------

class EventEnvelope(BaseModel):
    """Common envelope for all events in the payment world.

    Validates local structural correctness only. No simulation-clock
    or world-state validation.
    """

    event_id: str = Field(
        default_factory=_default_uuid, min_length=1,
        description="Unique event identifier (UUID).",
    )
    timestamp: datetime = Field(..., description="When the event occurred.")
    event_type: EventType = Field(..., description="Discriminated event category.")
    customer_id: str = Field(
        ..., min_length=1, description="FK → Customer.customer_id.",
    )
    account_id: str | None = Field(
        default=None, description="FK → Account.account_id (when applicable).",
    )
    session_id: str | None = Field(
        default=None, description="FK → Session.session_id (when applicable).",
    )

    @model_validator(mode="after")
    def _validate_optional_ids(self) -> EventEnvelope:
        """Reject empty-string optional IDs (None is fine, '' is not)."""
        if self.account_id is not None and len(self.account_id.strip()) == 0:
            raise ValueError("account_id must be non-empty when provided")
        if self.session_id is not None and len(self.session_id.strip()) == 0:
            raise ValueError("session_id must be non-empty when provided")
        return self


# ---------------------------------------------------------------------------
# Event Payloads
# ---------------------------------------------------------------------------

class TransactionEventPayload(BaseModel):
    """Payload for TRANSACTION events."""

    payload_type: Literal["transaction"] = "transaction"
    transaction: Transaction = Field(..., description="The transaction entity.")
    pre_balance: Decimal = Field(..., description="Account balance before the transaction.")
    post_balance: Decimal = Field(..., description="Account balance after the transaction.")


class SessionEventPayload(BaseModel):
    """Payload for SESSION_LOGIN and SESSION_LOGOUT events."""

    payload_type: Literal["session"] = "session"
    session: Session = Field(..., description="The session entity.")
    login_attempt_count: int = Field(
        ..., ge=0,
        description="Number of login attempts (including failed ones).",
    )


class DeviceEventPayload(BaseModel):
    """Payload for DEVICE_REGISTRATION and DEVICE_CHANGE events."""

    payload_type: Literal["device"] = "device"
    device: Device = Field(..., description="The device entity.")
    action: Literal["register", "change_primary", "deactivate"] = Field(
        ..., description="What device action occurred.",
    )
    previous_device_id: str | None = Field(
        default=None,
        description="Previous primary device ID (conventionally present for change_primary).",
    )


class BeneficiaryEventPayload(BaseModel):
    """Payload for BENEFICIARY_ADDITION and BENEFICIARY_REMOVAL events."""

    payload_type: Literal["beneficiary"] = "beneficiary"
    beneficiary: Beneficiary = Field(..., description="The beneficiary entity.")
    action: Literal["add", "remove", "modify"] = Field(
        ..., description="What beneficiary action occurred.",
    )


class AccountContextEventPayload(BaseModel):
    """Payload for ACCOUNT_CONTEXT_CHANGE events."""

    payload_type: Literal["account_context"] = "account_context"
    change_type: Literal[
        "contact_info", "security_settings", "address", "limits", "status"
    ] = Field(..., description="Category of account context change.")
    field_changed: str = Field(
        ..., min_length=1,
        description="Specific field that was changed.",
    )


class RelationshipEventPayload(BaseModel):
    """Payload for RELATIONSHIP_CHANGE events."""

    payload_type: Literal["relationship"] = "relationship"
    relationship: Relationship = Field(..., description="The relationship entity.")
    action: Literal["establish", "strengthen", "weaken", "terminate"] = Field(
        ..., description="What relationship action occurred.",
    )


# ---------------------------------------------------------------------------
# Discriminated payload union
# ---------------------------------------------------------------------------

EventPayload = Annotated[
    Union[
        TransactionEventPayload,
        SessionEventPayload,
        DeviceEventPayload,
        BeneficiaryEventPayload,
        AccountContextEventPayload,
        RelationshipEventPayload,
    ],
    Field(discriminator="payload_type"),
]
"""Pydantic discriminated union over all event payload types, keyed on payload_type."""


# ---------------------------------------------------------------------------
# EventType → payload_type mapping
# ---------------------------------------------------------------------------

EVENTTYPE_TO_PAYLOAD_TYPE: dict[EventType, str] = {
    EventType.TRANSACTION: "transaction",
    EventType.SESSION_LOGIN: "session",
    EventType.SESSION_LOGOUT: "session",
    EventType.DEVICE_REGISTRATION: "device",
    EventType.DEVICE_CHANGE: "device",
    EventType.BENEFICIARY_ADDITION: "beneficiary",
    EventType.BENEFICIARY_REMOVAL: "beneficiary",
    EventType.ACCOUNT_CONTEXT_CHANGE: "account_context",
    EventType.RELATIONSHIP_CHANGE: "relationship",
}
"""Authoritative mapping from EventType to the required payload_type string."""


# ---------------------------------------------------------------------------
# Full Event model
# ---------------------------------------------------------------------------

class Event(BaseModel):
    """A fully-typed event in the payment world event stream.

    Combines an EventEnvelope with a discriminated payload and validates:
    1. event_type/payload_type agreement
    2. Cross-reference coherence between envelope and payload
    """

    envelope: EventEnvelope = Field(..., description="Common event metadata.")
    payload: EventPayload = Field(..., description="Event-specific payload (discriminated union).")

    @model_validator(mode="after")
    def _validate_type_payload_agreement(self) -> Event:
        """Ensure envelope.event_type maps to the correct payload_type."""
        expected = EVENTTYPE_TO_PAYLOAD_TYPE[self.envelope.event_type]
        if self.payload.payload_type != expected:
            raise ValueError(
                f"Event type {self.envelope.event_type.value} requires "
                f"payload_type '{expected}', got '{self.payload.payload_type}'"
            )
        return self

    @model_validator(mode="after")
    def _validate_reference_coherence(self) -> Event:
        """Validate internal coherence of references between envelope and payload.

        Only checks references that are available on both sides.
        Does NOT enforce world-level entity existence.
        """
        envelope = self.envelope
        payload = self.payload

        if isinstance(payload, TransactionEventPayload):
            # If envelope specifies account_id, it must match the transaction's
            if envelope.account_id is not None:
                if payload.transaction.account_id != envelope.account_id:
                    raise ValueError(
                        f"Envelope account_id '{envelope.account_id}' does not match "
                        f"transaction account_id '{payload.transaction.account_id}'"
                    )

        if isinstance(payload, SessionEventPayload):
            # Session's customer_id must match envelope's customer_id
            if payload.session.customer_id != envelope.customer_id:
                raise ValueError(
                    f"Envelope customer_id '{envelope.customer_id}' does not match "
                    f"session customer_id '{payload.session.customer_id}'"
                )
            # If envelope specifies session_id, it must match session's
            if envelope.session_id is not None:
                if payload.session.session_id != envelope.session_id:
                    raise ValueError(
                        f"Envelope session_id '{envelope.session_id}' does not match "
                        f"session session_id '{payload.session.session_id}'"
                    )

        return self
