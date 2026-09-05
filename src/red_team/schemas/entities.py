"""Canonical entity schemas for the Red Team AI payment world.

All entities are versionable Pydantic models with strict validation.
These form the foundation of the stateful payment world simulation.

Design decisions:
    - Beneficiary is an INDEPENDENT entity (no customer_id FK). Customer-to-
      beneficiary associations are represented via the Relationship model.
      This supports shared beneficiaries, mule accounts, coordinated fraud,
      and beneficiary fan-in/fan-out analysis.
    - Transaction requires merchant_id for purchases and beneficiary_id for
      transfers, enforced by validators.
    - Relationship prevents self-loops and enforces strength bounds.
    - All models carry schema_version for future migration support.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from red_team.schemas.id_generator import generate_id as _default_uuid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# _default_uuid() is now the shared, seedable generator from id_generator.py
# (see that module's docstring). Unseeded behavior is unchanged; call
# red_team.schemas.id_generator.seed_ids(seed) to make IDs reproducible.


# ---------------------------------------------------------------------------
# Entity: Customer
# ---------------------------------------------------------------------------

class Customer(BaseModel):
    """A customer in the synthetic payment world."""

    customer_id: str = Field(default_factory=_default_uuid, description="Unique customer identifier (UUID).")
    name: str = Field(..., min_length=1, description="Customer display name.")
    registration_date: datetime = Field(..., description="When the customer registered.")
    risk_profile: Literal["low", "medium", "high"] = Field(
        default="low", description="Current risk classification."
    )
    behavioral_segment: str = Field(
        ..., min_length=1, description="Behavioral persona segment identifier."
    )
    country: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code."
    )
    status: Literal["active", "suspended", "closed"] = Field(
        default="active", description="Account lifecycle status."
    )
    schema_version: str = Field(default="1.0", description="Schema version for migration.")


# ---------------------------------------------------------------------------
# Entity: Account
# ---------------------------------------------------------------------------

class Account(BaseModel):
    """A financial account owned by a customer."""

    account_id: str = Field(default_factory=_default_uuid, description="Unique account identifier (UUID).")
    customer_id: str = Field(..., min_length=1, description="FK → Customer.customer_id.")
    bank_id: str = Field(
        ..., min_length=1,
        description=(
            "Identifier of the bank/financial institution holding this account. "
            "A customer's accounts are not required to share a bank_id, since "
            "real customers bank at more than one institution. This field is what "
            "lets a network-level (multi-bank) analysis see a mule pattern that is "
            "invisible from any single institution's own data."
        ),
    )
    account_type: Literal["checking", "savings", "credit", "business"] = Field(
        ..., description="Type of financial account."
    )
    currency: str = Field(
        ..., min_length=3, max_length=3, description="ISO 4217 currency code."
    )
    status: Literal["active", "frozen", "closed"] = Field(
        default="active", description="Account status."
    )
    opened_date: datetime = Field(..., description="When the account was opened.")
    balance: Decimal = Field(default=Decimal("0.00"), description="Current balance.")
    schema_version: str = Field(default="1.0", description="Schema version for migration.")

    @model_validator(mode="after")
    def _validate_balance(self) -> "Account":
        if self.account_type in ("checking", "savings", "business") and self.balance < 0:
            raise ValueError(
                f"Balance must be non-negative for {self.account_type} accounts, "
                f"got {self.balance}"
            )
        return self


# ---------------------------------------------------------------------------
# Entity: Device
# ---------------------------------------------------------------------------

class Device(BaseModel):
    """A device used to access the payment system."""

    device_id: str = Field(default_factory=_default_uuid, description="Unique device identifier (UUID).")
    device_type: Literal["mobile", "desktop", "tablet", "pos_terminal"] = Field(
        ..., description="Physical device category."
    )
    os: str | None = Field(default=None, description="Operating system.")
    browser: str | None = Field(default=None, description="Browser name/version.")
    fingerprint: str = Field(..., min_length=1, description="Device fingerprint hash.")
    first_seen: datetime = Field(..., description="When this device was first observed.")
    last_seen: datetime = Field(..., description="When this device was last observed.")
    is_trusted: bool = Field(default=False, description="Whether the device is marked trusted.")
    schema_version: str = Field(default="1.0", description="Schema version for migration.")

    @model_validator(mode="after")
    def _validate_seen_dates(self) -> "Device":
        if self.last_seen < self.first_seen:
            raise ValueError(
                f"last_seen ({self.last_seen}) must be >= first_seen ({self.first_seen})"
            )
        return self


# ---------------------------------------------------------------------------
# Entity: Merchant
# ---------------------------------------------------------------------------

class Merchant(BaseModel):
    """A merchant accepting payments."""

    merchant_id: str = Field(default_factory=_default_uuid, description="Unique merchant identifier (UUID).")
    name: str = Field(..., min_length=1, description="Merchant display name.")
    mcc_code: str = Field(
        ..., min_length=4, max_length=4, pattern=r"^\d{4}$",
        description="4-digit Merchant Category Code."
    )
    category: str = Field(..., min_length=1, description="Human-readable category.")
    country: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low", description="Merchant risk classification."
    )
    schema_version: str = Field(default="1.0", description="Schema version for migration.")


# ---------------------------------------------------------------------------
# Entity: Beneficiary (INDEPENDENT — no customer_id FK)
# ---------------------------------------------------------------------------

class Beneficiary(BaseModel):
    """A payment beneficiary.

    This is an independent entity — NOT scoped to a single customer.
    Customer-to-beneficiary associations are modeled via the Relationship
    entity to support:
        - shared beneficiaries across customers
        - mule account detection
        - coordinated fraud patterns
        - beneficiary fan-in / fan-out analysis
    """

    beneficiary_id: str = Field(default_factory=_default_uuid, description="Unique beneficiary identifier (UUID).")
    name: str = Field(..., min_length=1, description="Beneficiary display name.")
    account_reference: str = Field(..., min_length=1, description="Target account/IBAN reference.")
    bank_code: str | None = Field(default=None, description="Bank/routing code.")
    created_date: datetime = Field(..., description="When this beneficiary record was created.")
    relationship_type: Literal["personal", "business", "utility", "government", "other"] = Field(
        ..., description="General category of beneficiary."
    )
    is_verified: bool = Field(default=False, description="Whether identity has been verified.")
    is_active: bool = Field(default=True, description="Whether this beneficiary is active.")
    schema_version: str = Field(default="1.0", description="Schema version for migration.")


# ---------------------------------------------------------------------------
# Entity: Session
# ---------------------------------------------------------------------------

class Session(BaseModel):
    """An authenticated session in the payment system."""

    session_id: str = Field(default_factory=_default_uuid, description="Unique session identifier (UUID).")
    customer_id: str = Field(..., min_length=1, description="FK → Customer.customer_id.")
    device_id: str = Field(..., min_length=1, description="FK → Device.device_id.")
    ip_address: str = Field(..., min_length=1, description="Client IP address (IPv4 or IPv6).")
    geo_country: str | None = Field(default=None, description="Geo-located country.")
    geo_city: str | None = Field(default=None, description="Geo-located city.")
    start_time: datetime = Field(..., description="Session start timestamp.")
    end_time: datetime | None = Field(default=None, description="Session end timestamp (None if still active).")
    auth_method: Literal["password", "biometric", "mfa", "token", "sso"] = Field(
        ..., description="Authentication method used."
    )
    auth_success: bool = Field(..., description="Whether authentication succeeded.")
    schema_version: str = Field(default="1.0", description="Schema version for migration.")

    @model_validator(mode="after")
    def _validate_end_time(self) -> "Session":
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )
        return self


# ---------------------------------------------------------------------------
# Entity: Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """A financial transaction in the payment system."""

    transaction_id: str = Field(default_factory=_default_uuid, description="Unique transaction identifier (UUID).")
    account_id: str = Field(..., min_length=1, description="FK → Account.account_id.")
    session_id: str | None = Field(default=None, description="FK → Session.session_id (None for batch/scheduled).")
    merchant_id: str | None = Field(default=None, description="FK → Merchant.merchant_id (required for purchases).")
    beneficiary_id: str | None = Field(default=None, description="FK → Beneficiary.beneficiary_id (required for transfers).")
    amount: Decimal = Field(..., gt=0, description="Transaction amount (must be positive).")
    currency: str = Field(
        ..., min_length=3, max_length=3, description="ISO 4217 currency code."
    )
    transaction_type: Literal["purchase", "transfer", "withdrawal", "payment", "refund"] = Field(
        ..., description="Type of transaction."
    )
    status: Literal["pending", "completed", "failed", "reversed"] = Field(
        default="pending", description="Transaction lifecycle status."
    )
    channel: Literal["online", "pos", "atm", "mobile", "branch"] = Field(
        ..., description="Channel through which the transaction occurred."
    )
    timestamp: datetime = Field(..., description="When the transaction occurred.")
    schema_version: str = Field(default="1.0", description="Schema version for migration.")

    @model_validator(mode="after")
    def _validate_type_references(self) -> "Transaction":
        if self.transaction_type == "purchase" and not self.merchant_id:
            raise ValueError("purchase transactions require merchant_id")
        if self.transaction_type == "transfer" and not self.beneficiary_id:
            raise ValueError("transfer transactions require beneficiary_id")
        return self


# ---------------------------------------------------------------------------
# Entity: Relationship
# ---------------------------------------------------------------------------

# Canonical entity types that can participate in relationships.
VALID_SOURCE_ENTITY_TYPES = frozenset({"customer", "account", "device"})
VALID_TARGET_ENTITY_TYPES = frozenset(
    {"customer", "account", "device", "merchant", "beneficiary"}
)


class Relationship(BaseModel):
    """A directed relationship between two entities in the payment world.

    Used to model connections such as:
        Customer → Device       (uses)
        Customer → Beneficiary  (knows / pays)
        Customer → Merchant     (transacts_with)
        Account  → Customer     (owned_by)
        Device   → Customer     (used_by)
    """

    relationship_id: str = Field(default_factory=_default_uuid, description="Unique relationship identifier (UUID).")
    source_entity_type: Literal["customer", "account", "device"] = Field(
        ..., description="Type of the source entity."
    )
    source_entity_id: str = Field(..., min_length=1, description="ID of the source entity.")
    target_entity_type: Literal["customer", "account", "device", "merchant", "beneficiary"] = Field(
        ..., description="Type of the target entity."
    )
    target_entity_id: str = Field(..., min_length=1, description="ID of the target entity.")
    relationship_type: str = Field(
        ..., min_length=1,
        description="Relationship label (e.g. 'owns', 'uses', 'transacts_with', 'knows')."
    )
    established_date: datetime = Field(..., description="When the relationship was established.")
    last_activity_date: datetime | None = Field(
        default=None, description="When the relationship was last active."
    )
    is_active: bool = Field(default=True, description="Whether the relationship is currently active.")
    strength: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Interaction frequency weight (0.0 = dormant, 1.0 = very active)."
    )
    schema_version: str = Field(default="1.0", description="Schema version for migration.")

    @model_validator(mode="after")
    def _validate_no_self_loop(self) -> "Relationship":
        if (
            self.source_entity_type == self.target_entity_type
            and self.source_entity_id == self.target_entity_id
        ):
            raise ValueError(
                "Self-loops are not allowed: source and target must differ "
                f"(both are {self.source_entity_type}:{self.source_entity_id})"
            )
        return self

    @model_validator(mode="after")
    def _validate_activity_date(self) -> "Relationship":
        if self.last_activity_date is not None and self.last_activity_date < self.established_date:
            raise ValueError(
                f"last_activity_date ({self.last_activity_date}) must be >= "
                f"established_date ({self.established_date})"
            )
        return self
