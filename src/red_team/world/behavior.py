"""Behavioral simulator generating legitimate events based on World State."""

import random
import heapq
from datetime import timedelta, datetime
from decimal import Decimal
from typing import List, Optional, Tuple

from red_team.world.state import WorldState
from red_team.world.persona import PersonaParameters
from red_team.world.behavior_state import BehavioralModelConfig, CustomerBehaviorState
from red_team.schemas.entities import Session, Transaction, Device, Relationship
from red_team.schemas.events import (
    Event, EventEnvelope, EventType,
    TransactionEventPayload, SessionEventPayload, DeviceEventPayload, RelationshipEventPayload
)
from red_team.schemas.entities import Session, Transaction, Device, Relationship, Beneficiary
from red_team.schemas.events import (
    Event, EventEnvelope, EventType,
    TransactionEventPayload, SessionEventPayload, DeviceEventPayload,
    RelationshipEventPayload, BeneficiaryEventPayload
)


class BehavioralSimulator:
    """Simulates stateful, legitimate payment behavior."""

    def __init__(self, random_seed: int, personas: List[PersonaParameters], config: Optional[BehavioralModelConfig] = None):
        self.rng = random.Random(random_seed)
        self.personas = {p.segment_id: p for p in personas}
        self.config = config or BehavioralModelConfig()
        # Priority queue for scheduling: list of (next_event_time, customer_id)
        self._event_queue: List[Tuple[float, str]] = []
        self._initialized = False

    def _initialize_states_if_needed(self, state: WorldState):
        """Idempotently initialize customer behavioral states and the scheduling heap."""
        if self._initialized:
            # We must maintain the heap if new customers are added dynamically,
            # but for now assume population is static or we just check missing
            pass
        
        needs_heapify = False
        for customer_id, customer in state.customers.items():
            if customer_id not in state.customer_behavior:
                persona = self.personas[customer.behavioral_segment]
                
                # Initialize state
                # Transaction type weights: DOMAIN_MODELED assumption
                # We give them a random preference split between purchase/transfer
                base_purchase_weight = self.rng.uniform(0.5, 1.0)
                if persona.segment_id == "LOW_FREQUENCY":
                    base_purchase_weight = self.rng.uniform(0.8, 1.0)
                tx_type_weights = {
                    "purchase": base_purchase_weight,
                    "transfer": 1.0 - base_purchase_weight
                }
                
                typical_amount_anchor = self.rng.uniform(persona.typical_amount_range[0], persona.typical_amount_range[1])
                amount_variability = typical_amount_anchor * self.config.amount_variance_factor
                
                # Calculate initial next_event_time
                avg_events_per_week = self.rng.uniform(persona.tx_frequency_per_week[0], persona.tx_frequency_per_week[1])
                avg_gap_seconds = (7 * 24 * 3600) / max(0.1, avg_events_per_week)
                # Random initial offset
                initial_offset = self.rng.uniform(0, avg_gap_seconds)
                next_time = state.current_time + timedelta(seconds=initial_offset)
                
                cb = CustomerBehaviorState(
                    customer_id=customer_id,
                    next_event_time=next_time,
                    tx_type_weights=tx_type_weights,
                    typical_amount_anchor=typical_amount_anchor,
                    amount_variability=amount_variability
                )
                state.customer_behavior[customer_id] = cb
                self._event_queue.append((next_time.timestamp(), customer_id))
                needs_heapify = True
                
        if needs_heapify:
            heapq.heapify(self._event_queue)
        self._initialized = True

    def _schedule_next_event(self, state: WorldState, customer_id: str):
        """Schedule the customer's next event and push to heap."""
        cb = state.customer_behavior[customer_id]
        customer = state.customers[customer_id]
        persona = self.personas[customer.behavioral_segment]
        
        avg_events_per_week = self.rng.uniform(persona.tx_frequency_per_week[0], persona.tx_frequency_per_week[1])
        avg_gap_seconds = (7 * 24 * 3600) / max(0.1, avg_events_per_week)
        
        # Check burst
        if not cb.in_burst and self.rng.random() < self.config.burst_prob:
            cb.in_burst = True
            cb.burst_events_remaining = self.rng.randint(2, 5)
            
        if cb.in_burst:
            gap_seconds = avg_gap_seconds * self.config.burst_time_multiplier
            cb.burst_events_remaining -= 1
            if cb.burst_events_remaining <= 0:
                cb.in_burst = False
        else:
            # Exponential distribution around mean for normal gaps
            gap_seconds = self.rng.expovariate(1.0 / avg_gap_seconds)
            
        next_time = state.current_time + timedelta(seconds=gap_seconds)
        cb.next_event_time = next_time
        heapq.heappush(self._event_queue, (next_time.timestamp(), customer_id))

    def generate_next_event(self, state: WorldState) -> Optional[Event]:
        """Generate the next legitimate event by pulling from the scheduling heap."""
        if not state.customers:
            return None
            
        self._initialize_states_if_needed(state)
        
        if not self._event_queue:
            return None
            
        # Pop the earliest customer
        ts, customer_id = heapq.heappop(self._event_queue)
        next_time = datetime.fromtimestamp(ts)
        
        # Advance global time to this customer's next event
        if next_time > state.current_time:
            delta = (next_time - state.current_time).total_seconds()
            state.advance_time(int(delta))
            
        customer = state.customers[customer_id]
        persona = self.personas[customer.behavioral_segment]
        
        # Decide action type
        if self.rng.random() < self.config.drift_prob:
            event = self._generate_drift_event(state, customer_id)
        else:
            if customer_id not in state.active_sessions:
                event = self._generate_session_login(state, customer_id)
            else:
                action_choice = self.rng.choices(
                    ["transact", "logout", "add_beneficiary"],
                    weights=[0.8, 0.2, self.config.beneficiary_addition_prob],
                )[0]
                if action_choice == "logout":
                    event = self._generate_session_logout(state, customer_id)
                elif action_choice == "add_beneficiary":
                    event = self._generate_beneficiary_addition_event(state, customer_id)
                else:
                    event = self._generate_transaction(state, customer_id, persona)
        # If generation failed (e.g. no accounts), event might be None. Still schedule next.
        self._schedule_next_event(state, customer_id)
        return event

    def _generate_beneficiary_addition_event(self, state: WorldState, customer_id: str) -> Event:
        """Legit customers occasionally add a brand-new beneficiary during a session,
        distinct from the pre-seeded beneficiary pool established at population time.
        This gives NormalWorld a real BENEFICIARY_ADDITION signal so that "customer
        touches multiple beneficiaries" is not, by itself, a fraud-only shortcut."""
        beneficiary = Beneficiary(
            name=f"Beneficiary_{self.rng.randint(10000, 99999)}",
            account_reference=f"ACCT_REF_{self.rng.randint(100000, 999999)}",
            created_date=state.current_time,
            relationship_type=self.rng.choice(
                ["personal", "business", "utility", "government", "other"]
            ),
            is_verified=self.rng.random() < 0.7,
        )
        state.beneficiaries[beneficiary.beneficiary_id] = beneficiary

        envelope = EventEnvelope(
            timestamp=state.current_time,
            event_type=EventType.BENEFICIARY_ADDITION,
            customer_id=customer_id,
        )
        payload = BeneficiaryEventPayload(beneficiary=beneficiary, action="add")
        return Event(envelope=envelope, payload=payload)

    def _generate_session_login(self, state: WorldState, customer_id: str) -> Event:
        cb = state.customer_behavior[customer_id]
        
        device_id = None
        if cb.primary_device_id and self.rng.random() < self.config.device_reuse_prob:
            device_id = cb.primary_device_id
        else:
            # Pick alternate or primary if none exists
            devices = state.customer_devices.get(customer_id)
            if devices:
                device_id = self.rng.choice(devices)
                cb.primary_device_id = device_id
                
        session = Session(
            customer_id=customer_id,
            device_id=device_id,
            ip_address="192.168.1.1",
            start_time=state.current_time,
            auth_method="password",
            auth_success=True,
        )
        state.active_sessions[customer_id] = session
        
        envelope = EventEnvelope(
            timestamp=state.current_time,
            event_type=EventType.SESSION_LOGIN,
            customer_id=customer_id,
            session_id=session.session_id,
        )
        payload = SessionEventPayload(session=session, action="login", login_attempt_count=1)
        return Event(envelope=envelope, payload=payload)

    def _generate_session_logout(self, state: WorldState, customer_id: str) -> Event:
        session = state.active_sessions.pop(customer_id)
        session.end_time = state.current_time
        
        envelope = EventEnvelope(
            timestamp=state.current_time,
            event_type=EventType.SESSION_LOGOUT,
            customer_id=customer_id,
            session_id=session.session_id,
        )
        payload = SessionEventPayload(session=session, action="logout", login_attempt_count=1)
        return Event(envelope=envelope, payload=payload)

    def _generate_transaction(self, state: WorldState, customer_id: str, persona: PersonaParameters) -> Optional[Event]:
        cb = state.customer_behavior[customer_id]
        
        customer_accts = [a for a in state.accounts.values() if a.customer_id == customer_id]
        if not customer_accts:
            return None
        account = self.rng.choice(customer_accts)
        session = state.active_sessions[customer_id]
        
        # Decide type (purchase vs transfer) based on personalized weights
        types = list(cb.tx_type_weights.keys())
        weights = list(cb.tx_type_weights.values())
        tx_type = self.rng.choices(types, weights=weights)[0]
        
        # Statefully anchored amount (Gaussian around anchor)
        amount_val = self.rng.gauss(cb.typical_amount_anchor, cb.amount_variability)
        amount_val = max(1.0, amount_val) # prevent negative/zero amounts
        amount = Decimal(str(round(amount_val, 2)))
        
        merchant_id = None
        beneficiary_id = None
        
        if tx_type == "purchase" and state.merchants:
            merchant_id = self.rng.choice(list(state.merchants.keys()))
        elif tx_type == "transfer" and state.beneficiaries:
            # Stateful beneficiary selection
            if cb.beneficiary_affinities and self.rng.random() < self.config.beneficiary_reuse_prob:
                # Reuse known beneficiary
                b_ids = list(cb.beneficiary_affinities.keys())
                b_weights = list(cb.beneficiary_affinities.values())
                beneficiary_id = self.rng.choices(b_ids, weights=b_weights)[0]
                cb.beneficiary_affinities[beneficiary_id] += 1
            else:
                # Find new beneficiary
                rels = [r for r in state.relationships.values() 
                        if r.source_entity_id == customer_id and r.target_entity_type == "beneficiary"]
                if rels:
                    beneficiary_id = self.rng.choice(rels).target_entity_id
                else:
                    beneficiary_id = self.rng.choice(list(state.beneficiaries.keys()))
                cb.beneficiary_affinities[beneficiary_id] = cb.beneficiary_affinities.get(beneficiary_id, 0) + 1
        else:
            return None
            
        pre_balance = account.balance
        
        # Check balance
        if account.account_type in ("checking", "savings", "business"):
            if account.balance < amount:
                # Top up balance (simulate legitimate deposit)
                account.balance += amount * 2
                pre_balance = account.balance
                
        account.balance -= amount
        
        tx = Transaction(
            account_id=account.account_id,
            session_id=session.session_id,
            amount=amount,
            currency="USD",
            transaction_type=tx_type,
            merchant_id=merchant_id,
            beneficiary_id=beneficiary_id,
            channel="mobile",
            timestamp=state.current_time,
        )
        
        envelope = EventEnvelope(
            timestamp=state.current_time,
            event_type=EventType.TRANSACTION,
            customer_id=customer_id,
            account_id=account.account_id,
            session_id=session.session_id,
        )
        payload = TransactionEventPayload(transaction=tx, pre_balance=pre_balance, post_balance=account.balance)
        return Event(envelope=envelope, payload=payload)

    def _generate_drift_event(self, state: WorldState, customer_id: str) -> Event:
        # Register a new device to simulate drift
        device = Device(
            device_type="mobile",
            fingerprint=f"NEW_FINGERPRINT_{self.rng.randint(1000,9999)}",
            first_seen=state.current_time,
            last_seen=state.current_time,
            is_trusted=False
        )
        state.devices[device.device_id] = device
        
        if customer_id not in state.customer_devices:
            state.customer_devices[customer_id] = []
        state.customer_devices[customer_id].append(device.device_id)
        
        # Update stateful primary device optionally
        cb = state.customer_behavior[customer_id]
        if cb.primary_device_id is None:
            cb.primary_device_id = device.device_id
        
        envelope = EventEnvelope(
            timestamp=state.current_time,
            event_type=EventType.DEVICE_REGISTRATION,
            customer_id=customer_id,
        )
        payload = DeviceEventPayload(device=device, action="register")
        return Event(envelope=envelope, payload=payload)