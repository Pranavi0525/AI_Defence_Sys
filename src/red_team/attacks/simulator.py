"""Stateful ATO Attack Simulator.

Executes a structured attack plan against a legitimate customer's world state,
following the constrained state graph of an Attack Signature, producing
a strictly isolated ObservableAttackTrace and AttackGroundTruth pair.
"""

import uuid
import random
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Literal, Any
from pydantic import BaseModel, Field
from decimal import Decimal

from red_team.world.state import WorldState
from red_team.attacks.signature_library import AttackSignature, Observability
from red_team.schemas.events import (
    Event, EventEnvelope, EventType,
    Session, Device, Beneficiary,
    Transaction, Relationship,
    SessionEventPayload, DeviceEventPayload,
    BeneficiaryEventPayload, TransactionEventPayload,
    AccountContextEventPayload, RelationshipEventPayload
)
from red_team.schemas.observable import ObservableAttackTrace, extract_observable
from red_team.schemas.ground_truth import (
    AttackGroundTruth, AttackPhaseRecord,
    GenerationMetadata, PlannerMetadata, EvaluationMetadata
)


class AttackPlan(BaseModel):
    """Structured instruction for the Simulator."""
    attack_family: str
    entry_state: Optional[str] = None
    difficulty: Literal["easy", "medium", "hard", "advanced"]
    variation_settings: Dict[str, str] = Field(default_factory=dict)
    target_signal_intensity: str = "MEDIUM"
    affected_entity_preferences: List[str] = Field(default_factory=list)
    max_phases: int = 50
    max_events: int = 100
    max_simulation_duration_minutes: int = 1440
    # MULE_NETWORK only: identifies which ring this single mule's trace
    # belongs to, so a ring stays recoverable even after its traces are
    # stored as independent flat AttackRecords. None for ATO/APP.
    network_id: Optional[str] = None
    mule_hop_index: Optional[int] = None


class VariationProfile(BaseModel):
    # DOMAIN_MODELED parameter scales
    timing_type: Literal["RAPID", "NORMAL", "SLOW", "BURSTY"]
    device_new_prob: float
    beneficiary_new_prob: float
    amount_scale: Tuple[float, float]
    splits: Tuple[int, int]
    end_early_prob: float = 0.0
    loop_prob_mult: float = 1.0
    
    # APP-specific Social Engineering dimensions
    app_hesitation_prob: float = 0.0
    app_retry_prob: float = 0.0
    app_amount_trend: Literal["flat", "escalating", "fragmented", "decreasing"] = "flat"


DIFFICULTY_PROFILES = {
    # EASY: Fast, all new entities, large amounts, no splitting, eager to finish
    "easy": VariationProfile(
        timing_type="RAPID", device_new_prob=1.0, beneficiary_new_prob=1.0, 
        amount_scale=(0.8, 1.0), splits=(1, 1), end_early_prob=0.3, loop_prob_mult=0.5,
        app_hesitation_prob=0.0, app_retry_prob=0.0, app_amount_trend="flat"
    ),
    # MEDIUM: Normal speed, mostly new, moderate amounts, occasional split
    "medium": VariationProfile(
        timing_type="NORMAL", device_new_prob=0.8, beneficiary_new_prob=0.8, 
        amount_scale=(0.4, 0.7), splits=(1, 2), end_early_prob=0.1, loop_prob_mult=1.0,
        app_hesitation_prob=0.3, app_retry_prob=0.2, app_amount_trend="escalating"
    ),
    # HARD: Slow speed, reuses entities to blend in, small amounts, split up, avoids ending
    "hard": VariationProfile(
        timing_type="SLOW", device_new_prob=0.4, beneficiary_new_prob=0.3, 
        amount_scale=(0.1, 0.3), splits=(2, 3), end_early_prob=0.0, loop_prob_mult=2.0,
        app_hesitation_prob=0.7, app_retry_prob=0.5, app_amount_trend="fragmented"
    ),
    # ADVANCED: Bursty timing, almost entirely reuses entities, random amounts, many splits, high loops
    "advanced": VariationProfile(
        timing_type="BURSTY", device_new_prob=0.1, beneficiary_new_prob=0.1, 
        amount_scale=(0.2, 0.9), splits=(3, 5), end_early_prob=0.0, loop_prob_mult=3.0,
        app_hesitation_prob=0.5, app_retry_prob=0.8, app_amount_trend="decreasing"
    ),
}


class StatefulSimulator:
    def __init__(self, state: WorldState, signature: AttackSignature, seed: int):
        self.state = state
        self.signature = signature
        
        # IMPLEMENTED FIX: Salt the seed deterministically using the attack family
        salt_str = f"{seed}_{getattr(signature, 'attack_family', 'UNKNOWN')}"
        salted_seed = int(hashlib.sha256(salt_str.encode()).hexdigest()[:15], 16)
        
        self.seed = salted_seed
        self.rng = random.Random(self.seed)
        
        self.generated_events: List[Event] = []
        self.phase_records: List[AttackPhaseRecord] = []
        
    def _generate_event_id(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128)))

    def generate_attack(
        self, plan: AttackPlan, customer_id: str, forced_beneficiary: Optional[Beneficiary] = None
    ) -> Tuple[ObservableAttackTrace, AttackGroundTruth]:
        """Execute the stateful simulation.

        forced_beneficiary: if provided, seeds the attack's beneficiary
        slot with this entity instead of letting the simulator create or
        pick one on its own. Used by MuleNetworkOrchestrator so several
        different mule customers genuinely transact to the SAME shared
        collector beneficiary -- the correlating signal a graph model
        needs -- generated through this real event pipeline, not patched
        into already-generated traces afterward. Default (None) leaves
        ATO/APP behavior completely unchanged.
        """
        
        if customer_id not in self.state.customers:
            raise ValueError(f"Customer {customer_id} not found in WorldState")
            
        attack_id = f"atk-{self._generate_event_id()[:8]}"
        
        self.profile = DIFFICULTY_PROFILES[plan.difficulty]
        
        # 1. Determine entry state
        current_state_name = plan.entry_state
        if not current_state_name:
            current_state_name = self.rng.choice(self.signature.entry_states)
            
        phases_executed = 0
        events_generated = 0
        start_time = self.state.current_time
        
        attacker_session: Optional[Session] = None
        attacker_device: Optional[Device] = None
        attacker_beneficiary: Optional[Beneficiary] = forced_beneficiary

        # Enforce APP constraint: must use known primary device
        if plan.attack_family == "AUTHORIZED_PUSH_PAYMENT":
            known_devices = []
            for r in self.state.relationships.values():
                if r.source_entity_id == customer_id and r.target_entity_type == "device":
                    d_id = r.target_entity_id
                    if d_id in self.state.devices:
                        known_devices.append(self.state.devices[d_id])
            if known_devices:
                attacker_device = known_devices[0]

        # Enforce MULE_NETWORK constraint: mules transact from their own
        # known device too -- there's no credential compromise here.
        if plan.attack_family == "MULE_NETWORK":
            known_devices = []
            for r in self.state.relationships.values():
                if r.source_entity_id == customer_id and r.target_entity_type == "device":
                    d_id = r.target_entity_id
                    if d_id in self.state.devices:
                        known_devices.append(self.state.devices[d_id])
            if known_devices:
                attacker_device = known_devices[0]
        
        max_duration = plan.max_simulation_duration_minutes
        if self.profile.timing_type == "SLOW":
            max_duration = 1440 * 7 # DOMAIN_MODELED: 7 days to permit slow multi-phase attacks
            
        while current_state_name != "END":
            # Safety limits
            if phases_executed >= plan.max_phases:
                break
            if events_generated >= plan.max_events:
                break
            if (self.state.current_time - start_time).total_seconds() / 60 > max_duration:
                break
                
            phases_executed += 1
            attack_state = self.signature.states[current_state_name]
            
            phase_start = self.state.current_time
            
            # DOMAIN_MODELED: Timing scale based on difficulty
            if self.profile.timing_type == "RAPID":
                gap = self.rng.randint(1, 10) * 60
            elif self.profile.timing_type == "NORMAL":
                gap = self.rng.randint(30, 120) * 60
            elif self.profile.timing_type == "SLOW":
                gap = self.rng.randint(720, 1440) * 60
            elif self.profile.timing_type == "BURSTY":
                if self.rng.random() < 0.2:
                    gap = self.rng.randint(1440, 2880) * 60 # 1-2 days
                else:
                    gap = self.rng.randint(1, 10) * 60 # burst
            else:
                gap = 60
                
            # STAGE 28: APP Hesitation 
            if plan.attack_family == "AUTHORIZED_PUSH_PAYMENT" and current_state_name == "PAYMENT_EXECUTION":
                if self.rng.random() < self.profile.app_hesitation_prob:
                    # Inject hesitation (10 minutes to 2 hours)
                    gap += self.rng.randint(600, 7200)
                
            self.state.advance_time(gap)
            
            for consequence in attack_state.observable_consequences:
                new_events = self._synthesize_events_for_consequence(
                    consequence, customer_id, attacker_device, attacker_session, attacker_beneficiary
                )
                
                for ev in new_events:
                    payload = ev.payload
                    if isinstance(payload, DeviceEventPayload) and payload.action == "register":
                        attacker_device = payload.device
                    elif isinstance(payload, SessionEventPayload):
                        attacker_session = payload.session
                    elif isinstance(payload, BeneficiaryEventPayload) and payload.action == "add":
                        attacker_beneficiary = payload.beneficiary
                    
                    self.state.append_event(ev)  # Mutates world state graph
                    self.generated_events.append(ev)
                    events_generated += 1
            
            phase_end = self.state.current_time
            
            # 3. Transition Selection
            transitions = attack_state.transitions
            if not transitions:
                break
                
            # Bias weights based on difficulty
            weights = []
            for t in transitions:
                w = self.rng.uniform(t.min_weight, t.max_weight)
                if t.target_state == "END":
                    # Easy might end early, hard avoids ending
                    w *= (1.0 + self.profile.end_early_prob)
                    if self.profile.end_early_prob == 0.0:
                        w *= 0.5 
                elif t.target_state == current_state_name:
                    w *= self.profile.loop_prob_mult
                weights.append(w)
                
            total_w = sum(weights)
            if total_w <= 0:
                break
                
            normalized = [w / total_w for w in weights]
            
            r = self.rng.random()
            cumulative = 0.0
            next_state_name = "END"
            for t, p in zip(transitions, normalized):
                cumulative += p
                if r <= cumulative:
                    next_state_name = t.target_state
                    break
            
            was_optional = False
            
            self.phase_records.append(
                AttackPhaseRecord(
                    phase=current_state_name,
                    entered_at=phase_start,
                    exited_at=phase_end,
                    transition_to=next_state_name,
                    was_optional=was_optional
                )
            )
            
            current_state_name = next_state_name

        if not self.generated_events:
            raise ValueError("Attack generated zero events")

        trace = extract_observable(
            self.generated_events,
            trace_id=attack_id,
            merchant_lookup=self.state.merchants,
        )
        ground_truth = self._create_ground_truth(attack_id, plan)
        
        return trace, ground_truth
        
    def _synthesize_events_for_consequence(
        self, consequence, customer_id: str, 
        device: Optional[Device], session: Optional[Session], beneficiary: Optional[Beneficiary]
    ) -> List[Event]:
        events = []
        
        # Recon / Access -> Session/Device
        if "device" in consequence.affected_entities or "session" in consequence.affected_entities:
            # DOMAIN_MODELED: Device selection
            if device is None:
                if self.rng.random() >= self.profile.device_new_prob:
                    known_devices = []
                    for r in self.state.relationships.values():
                        if r.source_entity_id == customer_id and r.target_entity_type == "device":
                            d_id = r.target_entity_id
                            if d_id in self.state.devices:
                                known_devices.append(self.state.devices[d_id])
                    if known_devices:
                        device = self.rng.choice(known_devices)
                        
            if device is None:
                device = Device(
                    device_id=self._generate_event_id(),
                    device_type="desktop",
                    fingerprint=f"atk_fp_{self.rng.randint(100,999)}",
                    first_seen=self.state.current_time,
                    last_seen=self.state.current_time,
                    is_trusted=False
                )
                self.state.devices[device.device_id] = device
                env = EventEnvelope(
                    event_id=self._generate_event_id(),
                    timestamp=self.state.current_time,
                    event_type=EventType.DEVICE_REGISTRATION,
                    customer_id=customer_id
                )
                events.append(Event(envelope=env, payload=DeviceEventPayload(device=device, action="register")))
                
            if session is None:
                session = Session(
                    session_id=self._generate_event_id(),
                    customer_id=customer_id,
                    device_id=device.device_id,
                    ip_address="203.0.113.5",
                    start_time=self.state.current_time,
                    auth_method="password",
                    auth_success=True
                )
                self.state.active_sessions[customer_id] = session
                env = EventEnvelope(
                    event_id=self._generate_event_id(),
                    timestamp=self.state.current_time,
                    event_type=EventType.SESSION_LOGIN,
                    customer_id=customer_id,
                    session_id=session.session_id,
                    account_id=list(self.state.accounts.keys())[0] if self.state.accounts else None
                )
                events.append(Event(envelope=env, payload=SessionEventPayload(session=session, login_attempt_count=1)))
                
        # Modification -> Beneficiary/Context
        elif "beneficiary" in consequence.affected_entities and not "transaction" in consequence.affected_entities:
            # DOMAIN_MODELED: Beneficiary Selection
            if beneficiary is None:
                if self.rng.random() >= self.profile.beneficiary_new_prob:
                    known_bens = []
                    for r in self.state.relationships.values():
                        if r.source_entity_id == customer_id and r.target_entity_type == "beneficiary":
                            b_id = r.target_entity_id
                            if b_id in self.state.beneficiaries:
                                known_bens.append(self.state.beneficiaries[b_id])
                    if known_bens:
                        beneficiary = self.rng.choice(known_bens)

            if beneficiary is None:
                beneficiary = Beneficiary(
                    beneficiary_id=self._generate_event_id(),
                    name="Mock Beneficiary",
                    account_reference="offshore_acct",
                    created_date=self.state.current_time,
                    relationship_type="other",
                    is_verified=False
                )
                self.state.beneficiaries[beneficiary.beneficiary_id] = beneficiary
                env = EventEnvelope(
                    event_id=self._generate_event_id(),
                    timestamp=self.state.current_time,
                    event_type=EventType.BENEFICIARY_ADDITION,
                    customer_id=customer_id
                )
                events.append(Event(envelope=env, payload=BeneficiaryEventPayload(beneficiary=beneficiary, action="add")))
                
        # Exploitation -> Transaction
        elif "transaction" in consequence.affected_entities:
            acct_id = None
            for a_id, acct in self.state.accounts.items():
                if acct.account_type in ("checking", "savings"):
                    acct_id = a_id
                    break
            if not acct_id and self.state.accounts:
                acct_id = list(self.state.accounts.keys())[0]
                
            if acct_id:
                acct = self.state.accounts[acct_id]
                
                # DOMAIN_MODELED: Target amount relative to balance
                target_pct = self.rng.uniform(self.profile.amount_scale[0], self.profile.amount_scale[1])
                
                # Friction/Failure: 10% chance to erroneously overestimate balance
                if self.rng.random() < 0.1:
                    target_pct = self.rng.uniform(1.1, 1.5)
                    
                if acct.balance > Decimal("50.00"):
                    target_amt = acct.balance * Decimal(str(target_pct))
                else:
                    target_amt = Decimal("100.00") # flat attempt that will fail
                    
                target_amt = target_amt.quantize(Decimal("0.01"))
                
                # DOMAIN_MODELED: Transaction Splitting
                num_splits = self.rng.randint(self.profile.splits[0], self.profile.splits[1])
                
                split_amts = []
                remaining = target_amt
                for i in range(num_splits - 1):
                    portion = (remaining / (num_splits - i)) * Decimal(str(self.rng.uniform(0.8, 1.2)))
                    portion = portion.quantize(Decimal("0.01"))
                    if portion < Decimal("1.00"): portion = Decimal("1.00")
                    split_amts.append(portion)
                    remaining -= portion
                split_amts.append(max(Decimal("1.00"), remaining))
                
                # STAGE 28: APP Amount Trends
                if getattr(self.signature, "attack_family", "") == "AUTHORIZED_PUSH_PAYMENT":
                    if self.profile.app_amount_trend == "escalating":
                        split_amts.sort()
                    elif self.profile.app_amount_trend == "decreasing":
                        split_amts.sort(reverse=True)
                
                for amt in split_amts:
                    # Optional gap between splits
                    if num_splits > 1 and len(split_amts) > 1:
                        self.state.advance_time(self.rng.randint(5, 60))
                        
                    # STAGE 28: APP Retry Behavior (Simulate Bank Decline followed by victim retry)
                    if getattr(self.signature, "attack_family", "") == "AUTHORIZED_PUSH_PAYMENT":
                        if self.rng.random() < self.profile.app_retry_prob:
                            # Generate a declined transaction first
                            fail_amt = (amt * Decimal("1.5")).quantize(Decimal("0.01"))
                            fail_tx = Transaction(
                                account_id=acct_id, session_id=session.session_id if session else None,
                                amount=fail_amt, currency="USD", transaction_type="transfer" if beneficiary else "purchase",
                                beneficiary_id=beneficiary.beneficiary_id if beneficiary else None, merchant_id=None if beneficiary else f"merch-{self._generate_event_id()[:8]}",
                                timestamp=self.state.current_time, channel="online", status="failed"
                            )
                            fail_env = EventEnvelope(
                                event_id=self._generate_event_id(), timestamp=self.state.current_time, event_type=EventType.TRANSACTION,
                                customer_id=customer_id, account_id=acct_id, session_id=session.session_id if session else None
                            )
                            events.append(Event(envelope=fail_env, payload=TransactionEventPayload(
                                transaction=fail_tx, pre_balance=acct.balance, post_balance=acct.balance
                            )))
                            # Brief gap as they panic/negotiate
                            self.state.advance_time(self.rng.randint(30, 300))
                        
                    pre_balance = acct.balance
                    
                    if amt < Decimal("1.00"):
                        amt = Decimal("1.00") # Enforce schema constraint > 0
                    
                    if acct.balance < amt:
                        tx_status = "failed"
                        post_balance = acct.balance
                    else:
                        tx_status = "completed"
                        acct.balance -= amt
                        post_balance = acct.balance
                    
                    tx = Transaction(
                        account_id=acct_id,
                        session_id=session.session_id if session else None,
                        amount=amt,
                        currency="USD",
                        transaction_type="transfer" if beneficiary else "purchase",
                        beneficiary_id=beneficiary.beneficiary_id if beneficiary else None,
                        merchant_id=None if beneficiary else f"merch-{self._generate_event_id()[:8]}",
                        timestamp=self.state.current_time,
                        channel="online",
                        status=tx_status
                    )
                    
                    env = EventEnvelope(
                        event_id=self._generate_event_id(),
                        timestamp=self.state.current_time,
                        event_type=EventType.TRANSACTION,
                        customer_id=customer_id,
                        account_id=acct_id,
                        session_id=session.session_id if session else None
                    )
                    events.append(Event(envelope=env, payload=TransactionEventPayload(
                        transaction=tx, pre_balance=pre_balance, post_balance=post_balance
                    )))
                
        return events

    def _create_ground_truth(self, attack_id: str, plan: AttackPlan) -> AttackGroundTruth:
        conf_hash = hashlib.sha256(str(self.seed).encode()).hexdigest()[:8]
        
        gen_meta = GenerationMetadata(
            random_seed=self.seed,
            generator_version="1.0",
            signature_version=self.signature.version,
            provenance_registry_version="1.0",
            configuration_hash=conf_hash,
            generated_at=datetime.now(timezone.utc)
        )
        
        plan_meta = PlannerMetadata(
            planner_type="mock",
            plan_json=plan.model_dump(),
        )
        
        eval_meta = EvaluationMetadata(
            structural_valid=True,
        )
        
        return AttackGroundTruth(
            attack_id=attack_id,
            attack_family=self.signature.attack_family,
            attack_difficulty=plan.difficulty,
            hidden_objective="extract_funds",
            phases_executed=self.phase_records,
            linked_event_ids=[e.envelope.event_id for e in self.generated_events],
            generation_metadata=gen_meta,
            planner_metadata=plan_meta,
            evaluation_metadata=eval_meta
        )
