"""Stage 2.1 — Entity schema tests.

Covers:
    - Valid construction of all 8 entity models
    - Required field enforcement
    - Invalid value rejection (enums, constraints)
    - Validator logic (dates, amounts, self-loops, balance)
    - Serialization round-trip (model_dump → reconstruct)
    - schema_version presence
    - Beneficiary independence (no customer_id)
    - Relationship constraints (self-loop, strength bounds, date ordering)
    - Transaction type-reference enforcement (merchant_id / beneficiary_id)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from red_team.schemas.entities import (
    Account,
    Beneficiary,
    Customer,
    Device,
    Merchant,
    Relationship,
    Session,
    Transaction,
)


# =========================================================================
# Fixtures — valid entity factories
# =========================================================================

NOW = datetime(2025, 6, 15, 12, 0, 0)
YESTERDAY = NOW - timedelta(days=1)
TOMORROW = NOW + timedelta(days=1)


def _make_customer(**overrides) -> Customer:
    defaults = dict(
        name="Alice Smith",
        registration_date=YESTERDAY,
        risk_profile="low",
        behavioral_segment="digital_native",
        country="US",
        status="active",
    )
    defaults.update(overrides)
    return Customer(**defaults)


def _make_account(**overrides) -> Account:
    defaults = dict(
        customer_id="cust-001",
        bank_id="BANK_A",
        account_type="checking",
        currency="USD",
        status="active",
        opened_date=YESTERDAY,
        balance=Decimal("1000.00"),
    )
    defaults.update(overrides)
    return Account(**defaults)


def _make_device(**overrides) -> Device:
    defaults = dict(
        device_type="mobile",
        os="iOS 17",
        browser=None,
        fingerprint="fp-abc123",
        first_seen=YESTERDAY,
        last_seen=NOW,
        is_trusted=False,
    )
    defaults.update(overrides)
    return Device(**defaults)


def _make_merchant(**overrides) -> Merchant:
    defaults = dict(
        name="Coffee Shop",
        mcc_code="5812",
        category="Restaurants",
        country="US",
        risk_level="low",
    )
    defaults.update(overrides)
    return Merchant(**defaults)


def _make_beneficiary(**overrides) -> Beneficiary:
    defaults = dict(
        name="Bob Jones",
        account_reference="IBAN-DE89370400440532013000",
        bank_code="COBADEFF",
        created_date=YESTERDAY,
        relationship_type="personal",
        is_verified=True,
        is_active=True,
    )
    defaults.update(overrides)
    return Beneficiary(**defaults)


def _make_session(**overrides) -> Session:
    defaults = dict(
        customer_id="cust-001",
        device_id="dev-001",
        ip_address="192.168.1.1",
        geo_country="US",
        start_time=NOW,
        end_time=NOW + timedelta(minutes=30),
        auth_method="password",
        auth_success=True,
    )
    defaults.update(overrides)
    return Session(**defaults)


def _make_transaction(**overrides) -> Transaction:
    defaults = dict(
        account_id="acct-001",
        session_id="sess-001",
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


def _make_relationship(**overrides) -> Relationship:
    defaults = dict(
        source_entity_type="customer",
        source_entity_id="cust-001",
        target_entity_type="beneficiary",
        target_entity_id="ben-001",
        relationship_type="knows",
        established_date=YESTERDAY,
        last_activity_date=NOW,
        is_active=True,
        strength=0.7,
    )
    defaults.update(overrides)
    return Relationship(**defaults)


# =========================================================================
# Customer tests
# =========================================================================

class TestCustomer:
    def test_valid_construction(self):
        c = _make_customer()
        assert c.name == "Alice Smith"
        assert c.risk_profile == "low"
        assert c.country == "US"
        assert c.schema_version == "1.0"
        assert c.customer_id  # auto-generated UUID

    def test_schema_version_present(self):
        c = _make_customer()
        assert "schema_version" in Customer.model_fields
        assert c.schema_version == "1.0"

    def test_auto_uuid(self):
        c1 = _make_customer()
        c2 = _make_customer()
        assert c1.customer_id != c2.customer_id

    def test_invalid_risk_profile(self):
        with pytest.raises(ValidationError, match="risk_profile"):
            _make_customer(risk_profile="extreme")

    def test_invalid_status(self):
        with pytest.raises(ValidationError, match="status"):
            _make_customer(status="deleted")

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            _make_customer(name="")

    def test_invalid_country_code_too_long(self):
        with pytest.raises(ValidationError, match="country"):
            _make_customer(country="USA")

    def test_invalid_country_code_too_short(self):
        with pytest.raises(ValidationError, match="country"):
            _make_customer(country="U")

    def test_empty_behavioral_segment_rejected(self):
        with pytest.raises(ValidationError, match="behavioral_segment"):
            _make_customer(behavioral_segment="")

    def test_serialization_round_trip(self):
        c = _make_customer()
        data = c.model_dump()
        c2 = Customer(**data)
        assert c == c2


# =========================================================================
# Account tests
# =========================================================================

class TestAccount:
    def test_valid_construction(self):
        a = _make_account()
        assert a.customer_id == "cust-001"
        assert a.balance == Decimal("1000.00")
        assert a.schema_version == "1.0"

    def test_negative_balance_rejected_for_checking(self):
        with pytest.raises(ValidationError, match="non-negative"):
            _make_account(account_type="checking", balance=Decimal("-50.00"))

    def test_negative_balance_rejected_for_savings(self):
        with pytest.raises(ValidationError, match="non-negative"):
            _make_account(account_type="savings", balance=Decimal("-1.00"))

    def test_negative_balance_rejected_for_business(self):
        with pytest.raises(ValidationError, match="non-negative"):
            _make_account(account_type="business", balance=Decimal("-0.01"))

    def test_negative_balance_allowed_for_credit(self):
        a = _make_account(account_type="credit", balance=Decimal("-500.00"))
        assert a.balance == Decimal("-500.00")

    def test_invalid_account_type(self):
        with pytest.raises(ValidationError, match="account_type"):
            _make_account(account_type="investment")

    def test_invalid_currency_too_long(self):
        with pytest.raises(ValidationError, match="currency"):
            _make_account(currency="USDT")

    def test_zero_balance_allowed(self):
        a = _make_account(balance=Decimal("0.00"))
        assert a.balance == Decimal("0.00")

    def test_serialization_round_trip(self):
        a = _make_account()
        data = a.model_dump()
        a2 = Account(**data)
        assert a == a2


# =========================================================================
# Device tests
# =========================================================================

class TestDevice:
    def test_valid_construction(self):
        d = _make_device()
        assert d.device_type == "mobile"
        assert d.fingerprint == "fp-abc123"
        assert d.schema_version == "1.0"

    def test_last_seen_before_first_seen_rejected(self):
        with pytest.raises(ValidationError, match="last_seen"):
            _make_device(first_seen=NOW, last_seen=YESTERDAY)

    def test_same_first_last_seen_allowed(self):
        d = _make_device(first_seen=NOW, last_seen=NOW)
        assert d.first_seen == d.last_seen

    def test_invalid_device_type(self):
        with pytest.raises(ValidationError, match="device_type"):
            _make_device(device_type="smartwatch")

    def test_empty_fingerprint_rejected(self):
        with pytest.raises(ValidationError, match="fingerprint"):
            _make_device(fingerprint="")

    def test_optional_os_and_browser(self):
        d = _make_device(os=None, browser=None)
        assert d.os is None
        assert d.browser is None

    def test_serialization_round_trip(self):
        d = _make_device()
        data = d.model_dump()
        d2 = Device(**data)
        assert d == d2


# =========================================================================
# Merchant tests
# =========================================================================

class TestMerchant:
    def test_valid_construction(self):
        m = _make_merchant()
        assert m.mcc_code == "5812"
        assert m.category == "Restaurants"
        assert m.schema_version == "1.0"

    def test_invalid_mcc_code_non_digit(self):
        with pytest.raises(ValidationError, match="mcc_code"):
            _make_merchant(mcc_code="58AB")

    def test_invalid_mcc_code_too_short(self):
        with pytest.raises(ValidationError, match="mcc_code"):
            _make_merchant(mcc_code="581")

    def test_invalid_mcc_code_too_long(self):
        with pytest.raises(ValidationError, match="mcc_code"):
            _make_merchant(mcc_code="58121")

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            _make_merchant(name="")

    def test_invalid_risk_level(self):
        with pytest.raises(ValidationError, match="risk_level"):
            _make_merchant(risk_level="critical")

    def test_serialization_round_trip(self):
        m = _make_merchant()
        data = m.model_dump()
        m2 = Merchant(**data)
        assert m == m2


# =========================================================================
# Beneficiary tests — INDEPENDENT entity (no customer_id)
# =========================================================================

class TestBeneficiary:
    def test_valid_construction(self):
        b = _make_beneficiary()
        assert b.name == "Bob Jones"
        assert b.is_verified is True
        assert b.schema_version == "1.0"

    def test_no_customer_id_field(self):
        """Beneficiary must NOT have a customer_id field — it is independent."""
        assert "customer_id" not in Beneficiary.model_fields

    def test_beneficiary_independence(self):
        """Two separate beneficiary instances are not scoped to any customer."""
        b1 = _make_beneficiary(name="Bob")
        b2 = _make_beneficiary(name="Carol")
        # Neither has customer_id
        assert "customer_id" not in Beneficiary.model_fields
        assert b1.beneficiary_id != b2.beneficiary_id

    def test_invalid_relationship_type(self):
        with pytest.raises(ValidationError, match="relationship_type"):
            _make_beneficiary(relationship_type="crypto_exchange")

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            _make_beneficiary(name="")

    def test_empty_account_reference_rejected(self):
        with pytest.raises(ValidationError, match="account_reference"):
            _make_beneficiary(account_reference="")

    def test_optional_bank_code(self):
        b = _make_beneficiary(bank_code=None)
        assert b.bank_code is None

    def test_customer_beneficiary_via_relationship(self):
        """Customer → Beneficiary association must use the Relationship model."""
        b = _make_beneficiary()
        r = _make_relationship(
            source_entity_type="customer",
            source_entity_id="cust-001",
            target_entity_type="beneficiary",
            target_entity_id=b.beneficiary_id,
            relationship_type="knows",
        )
        assert r.target_entity_id == b.beneficiary_id
        assert r.source_entity_type == "customer"
        assert r.target_entity_type == "beneficiary"

    def test_shared_beneficiary_across_customers(self):
        """Multiple customers can reference the same beneficiary via Relationship."""
        b = _make_beneficiary()
        r1 = _make_relationship(
            source_entity_id="cust-001",
            target_entity_id=b.beneficiary_id,
            target_entity_type="beneficiary",
        )
        r2 = _make_relationship(
            source_entity_id="cust-002",
            target_entity_id=b.beneficiary_id,
            target_entity_type="beneficiary",
        )
        assert r1.target_entity_id == r2.target_entity_id
        assert r1.source_entity_id != r2.source_entity_id

    def test_serialization_round_trip(self):
        b = _make_beneficiary()
        data = b.model_dump()
        b2 = Beneficiary(**data)
        assert b == b2


# =========================================================================
# Session tests
# =========================================================================

class TestSession:
    def test_valid_construction(self):
        s = _make_session()
        assert s.auth_method == "password"
        assert s.auth_success is True
        assert s.schema_version == "1.0"

    def test_end_time_before_start_rejected(self):
        with pytest.raises(ValidationError, match="end_time"):
            _make_session(start_time=NOW, end_time=YESTERDAY)

    def test_none_end_time_allowed(self):
        s = _make_session(end_time=None)
        assert s.end_time is None

    def test_same_start_end_time_allowed(self):
        s = _make_session(start_time=NOW, end_time=NOW)
        assert s.start_time == s.end_time

    def test_invalid_auth_method(self):
        with pytest.raises(ValidationError, match="auth_method"):
            _make_session(auth_method="face_id")

    def test_failed_auth(self):
        s = _make_session(auth_success=False)
        assert s.auth_success is False

    def test_optional_geo_fields(self):
        s = _make_session(geo_country=None, geo_city=None)
        assert s.geo_country is None
        assert s.geo_city is None

    def test_serialization_round_trip(self):
        s = _make_session()
        data = s.model_dump()
        s2 = Session(**data)
        assert s == s2


# =========================================================================
# Transaction tests
# =========================================================================

class TestTransaction:
    def test_valid_purchase(self):
        t = _make_transaction(transaction_type="purchase", merchant_id="merch-001")
        assert t.amount == Decimal("42.50")
        assert t.schema_version == "1.0"

    def test_valid_transfer(self):
        t = _make_transaction(
            transaction_type="transfer",
            merchant_id=None,
            beneficiary_id="ben-001",
        )
        assert t.transaction_type == "transfer"
        assert t.beneficiary_id == "ben-001"

    def test_purchase_without_merchant_rejected(self):
        with pytest.raises(ValidationError, match="merchant_id"):
            _make_transaction(transaction_type="purchase", merchant_id=None)

    def test_transfer_without_beneficiary_rejected(self):
        with pytest.raises(ValidationError, match="beneficiary_id"):
            _make_transaction(
                transaction_type="transfer",
                merchant_id=None,
                beneficiary_id=None,
            )

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="amount"):
            _make_transaction(amount=Decimal("0.00"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="amount"):
            _make_transaction(amount=Decimal("-10.00"))

    def test_invalid_transaction_type(self):
        with pytest.raises(ValidationError, match="transaction_type"):
            _make_transaction(transaction_type="crypto_swap")

    def test_invalid_channel(self):
        with pytest.raises(ValidationError, match="channel"):
            _make_transaction(channel="carrier_pigeon")

    def test_invalid_status(self):
        with pytest.raises(ValidationError, match="status"):
            _make_transaction(status="cancelled")

    def test_withdrawal_without_merchant_allowed(self):
        t = _make_transaction(transaction_type="withdrawal", merchant_id=None)
        assert t.transaction_type == "withdrawal"

    def test_payment_without_beneficiary_allowed(self):
        t = _make_transaction(
            transaction_type="payment",
            merchant_id="merch-001",
            beneficiary_id=None,
        )
        assert t.transaction_type == "payment"

    def test_optional_session_id(self):
        t = _make_transaction(session_id=None)
        assert t.session_id is None

    def test_serialization_round_trip(self):
        t = _make_transaction()
        data = t.model_dump()
        t2 = Transaction(**data)
        assert t == t2


# =========================================================================
# Relationship tests
# =========================================================================

class TestRelationship:
    def test_valid_construction(self):
        r = _make_relationship()
        assert r.source_entity_type == "customer"
        assert r.target_entity_type == "beneficiary"
        assert r.strength == 0.7
        assert r.schema_version == "1.0"

    def test_self_loop_rejected(self):
        with pytest.raises(ValidationError, match="Self-loops"):
            _make_relationship(
                source_entity_type="customer",
                source_entity_id="cust-001",
                target_entity_type="customer",
                target_entity_id="cust-001",
            )

    def test_same_type_different_id_allowed(self):
        r = _make_relationship(
            source_entity_type="customer",
            source_entity_id="cust-001",
            target_entity_type="customer",
            target_entity_id="cust-002",
        )
        assert r.source_entity_id != r.target_entity_id

    def test_strength_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="strength"):
            _make_relationship(strength=-0.1)

    def test_strength_above_one_rejected(self):
        with pytest.raises(ValidationError, match="strength"):
            _make_relationship(strength=1.1)

    def test_strength_boundary_zero(self):
        r = _make_relationship(strength=0.0)
        assert r.strength == 0.0

    def test_strength_boundary_one(self):
        r = _make_relationship(strength=1.0)
        assert r.strength == 1.0

    def test_last_activity_before_established_rejected(self):
        with pytest.raises(ValidationError, match="last_activity_date"):
            _make_relationship(
                established_date=NOW,
                last_activity_date=YESTERDAY,
            )

    def test_last_activity_none_allowed(self):
        r = _make_relationship(last_activity_date=None)
        assert r.last_activity_date is None

    def test_same_established_and_activity_allowed(self):
        r = _make_relationship(
            established_date=NOW,
            last_activity_date=NOW,
        )
        assert r.established_date == r.last_activity_date

    def test_invalid_source_entity_type(self):
        with pytest.raises(ValidationError, match="source_entity_type"):
            _make_relationship(source_entity_type="merchant")

    def test_invalid_target_entity_type(self):
        with pytest.raises(ValidationError, match="target_entity_type"):
            _make_relationship(target_entity_type="session")

    def test_customer_to_device_relationship(self):
        r = _make_relationship(
            source_entity_type="customer",
            source_entity_id="cust-001",
            target_entity_type="device",
            target_entity_id="dev-001",
            relationship_type="uses",
        )
        assert r.relationship_type == "uses"

    def test_customer_to_merchant_relationship(self):
        r = _make_relationship(
            source_entity_type="customer",
            source_entity_id="cust-001",
            target_entity_type="merchant",
            target_entity_id="merch-001",
            relationship_type="transacts_with",
        )
        assert r.relationship_type == "transacts_with"

    def test_serialization_round_trip(self):
        r = _make_relationship()
        data = r.model_dump()
        r2 = Relationship(**data)
        assert r == r2


# =========================================================================
# Cross-entity integration tests
# =========================================================================

class TestCrossEntityIntegration:
    """Verify that entity references work correctly across models."""

    def test_customer_account_reference(self):
        c = _make_customer()
        a = _make_account(customer_id=c.customer_id)
        assert a.customer_id == c.customer_id

    def test_session_references_customer_and_device(self):
        c = _make_customer()
        d = _make_device()
        s = _make_session(customer_id=c.customer_id, device_id=d.device_id)
        assert s.customer_id == c.customer_id
        assert s.device_id == d.device_id

    def test_transaction_references_account_and_merchant(self):
        a = _make_account()
        m = _make_merchant()
        t = _make_transaction(
            account_id=a.account_id,
            merchant_id=m.merchant_id,
            transaction_type="purchase",
        )
        assert t.account_id == a.account_id
        assert t.merchant_id == m.merchant_id

    def test_transfer_references_beneficiary_via_relationship(self):
        """Full chain: Customer → (Relationship) → Beneficiary → Transaction."""
        c = _make_customer()
        b = _make_beneficiary()
        a = _make_account(customer_id=c.customer_id)

        # Link customer to beneficiary via Relationship
        r = _make_relationship(
            source_entity_type="customer",
            source_entity_id=c.customer_id,
            target_entity_type="beneficiary",
            target_entity_id=b.beneficiary_id,
            relationship_type="knows",
        )

        # Transfer to beneficiary
        t = _make_transaction(
            account_id=a.account_id,
            transaction_type="transfer",
            merchant_id=None,
            beneficiary_id=b.beneficiary_id,
        )

        assert r.source_entity_id == c.customer_id
        assert r.target_entity_id == b.beneficiary_id
        assert t.beneficiary_id == b.beneficiary_id

    def test_all_models_have_schema_version(self):
        """Every entity model must have schema_version field."""
        models = [
            _make_customer(),
            _make_account(),
            _make_device(),
            _make_merchant(),
            _make_beneficiary(),
            _make_session(),
            _make_transaction(),
            _make_relationship(),
        ]
        for model in models:
            assert hasattr(model, "schema_version"), (
                f"{type(model).__name__} missing schema_version"
            )
            assert model.schema_version == "1.0"

    def test_all_models_generate_unique_ids(self):
        """Auto-generated IDs should be unique across instances."""
        ids = set()
        factories = [
            _make_customer,
            _make_account,
            _make_device,
            _make_merchant,
            _make_beneficiary,
            _make_session,
            _make_transaction,
            _make_relationship,
        ]
        for factory in factories:
            for _ in range(3):
                obj = factory()
                # Get the primary ID field (first field ending in _id)
                model_cls = type(obj)
                id_fields = [
                    f for f in model_cls.model_fields
                    if f.endswith("_id") and f == list(model_cls.model_fields.keys())[0]
                ]
                if id_fields:
                    val = getattr(obj, id_fields[0])
                    assert val not in ids, f"Duplicate ID {val} in {type(obj).__name__}"
                    ids.add(val)
