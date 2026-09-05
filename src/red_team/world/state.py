"""World State maintaining the synthetic payment simulation."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from red_team.schemas.entities import (
    Customer,
    Account,
    Device,
    Merchant,
    Beneficiary,
    Relationship,
    Session,
)
from red_team.schemas.events import Event
from red_team.world.behavior_state import CustomerBehaviorState

class WorldState(BaseModel):
    """The authoritative state of the Normal World simulation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_time: datetime
    
    customers: Dict[str, Customer] = Field(default_factory=dict)
    accounts: Dict[str, Account] = Field(default_factory=dict)
    devices: Dict[str, Device] = Field(default_factory=dict)
    merchants: Dict[str, Merchant] = Field(default_factory=dict)
    beneficiaries: Dict[str, Beneficiary] = Field(default_factory=dict)
    relationships: Dict[str, Relationship] = Field(default_factory=dict)
    
    customer_behavior: Dict[str, CustomerBehaviorState] = Field(default_factory=dict)
    
    # Active sessions keyed by customer_id
    active_sessions: Dict[str, Session] = Field(default_factory=dict)
    
    # Customer devices mapping
    customer_devices: Dict[str, List[str]] = Field(default_factory=dict)
    
    event_history: List[Event] = Field(default_factory=list)
    
    # Graph representing relationships
    graph: Any = Field(default=None, description="RelationshipGraph instance")
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize the graph upon state creation."""
        from red_team.world.graph import RelationshipGraph
        if self.graph is None:
            self.graph = RelationshipGraph()

    def advance_time(self, delta_seconds: int) -> None:
        """Advance the simulation clock."""
        from datetime import timedelta
        self.current_time += timedelta(seconds=delta_seconds)

    def append_event(self, event: Event) -> None:
        """Append an event to the chronological history and update the graph."""
        self.event_history.append(event)
        
        # Event-driven graph updates
        payload = event.payload
        from red_team.schemas.events import (
            DeviceEventPayload, TransactionEventPayload, RelationshipEventPayload,
            SessionEventPayload, BeneficiaryEventPayload
        )
        
        if isinstance(payload, BeneficiaryEventPayload):
            if payload.action == "add":
                self.graph.add_entity("beneficiary", payload.beneficiary.beneficiary_id)
                rel = Relationship(
                    source_entity_type="customer",
                    source_entity_id=event.envelope.customer_id,
                    target_entity_type="beneficiary",
                    target_entity_id=payload.beneficiary.beneficiary_id,
                    relationship_type="transacts_with",
                    established_date=event.envelope.timestamp,
                    last_activity_date=event.envelope.timestamp,
                )
                self.relationships[rel.relationship_id] = rel
                self.graph.add_relationship(rel)

        elif isinstance(payload, DeviceEventPayload):
            self.graph.add_entity("device", payload.device.device_id)
            # Add relationship between customer and device
            rel = Relationship(
                source_entity_type="customer",
                source_entity_id=event.envelope.customer_id,
                target_entity_type="device",
                target_entity_id=payload.device.device_id,
                relationship_type="uses",
                established_date=event.envelope.timestamp,
                last_activity_date=event.envelope.timestamp,
            )
            self.relationships[rel.relationship_id] = rel
            self.graph.add_relationship(rel)
            
        elif isinstance(payload, TransactionEventPayload):
            tx = payload.transaction
            if tx.merchant_id:
                # transacts_with relationship
                # find existing relationship
                existing_rel = None
                for r in self.relationships.values():
                    if (r.source_entity_id == event.envelope.customer_id and
                        r.target_entity_id == tx.merchant_id and
                        r.relationship_type == "transacts_with"):
                        existing_rel = r
                        break
                        
                if existing_rel:
                    existing_rel.last_activity_date = tx.timestamp
                    self.graph.update_relationship(existing_rel)
                else:
                    rel = Relationship(
                        source_entity_type="customer",
                        source_entity_id=event.envelope.customer_id,
                        target_entity_type="merchant",
                        target_entity_id=tx.merchant_id,
                        relationship_type="transacts_with",
                        established_date=tx.timestamp,
                        last_activity_date=tx.timestamp,
                    )
                    self.relationships[rel.relationship_id] = rel
                    self.graph.add_relationship(rel)
                    
            elif tx.beneficiary_id:
                # transacts_with relationship for beneficiary
                existing_rel = None
                for r in self.relationships.values():
                    if (r.source_entity_id == event.envelope.customer_id and
                        r.target_entity_id == tx.beneficiary_id and
                        r.relationship_type == "transacts_with"):
                        existing_rel = r
                        break
                        
                if existing_rel:
                    existing_rel.last_activity_date = tx.timestamp
                    self.graph.update_relationship(existing_rel)
                else:
                    rel = Relationship(
                        source_entity_type="customer",
                        source_entity_id=event.envelope.customer_id,
                        target_entity_type="beneficiary",
                        target_entity_id=tx.beneficiary_id,
                        relationship_type="transacts_with",
                        established_date=tx.timestamp,
                        last_activity_date=tx.timestamp,
                    )
                    self.relationships[rel.relationship_id] = rel
                    self.graph.add_relationship(rel)
                    
        elif isinstance(payload, RelationshipEventPayload):
            rel = payload.relationship
            if payload.action == "establish":
                self.relationships[rel.relationship_id] = rel
                self.graph.add_relationship(rel)
            elif payload.action == "terminate":
                if rel.relationship_id in self.relationships:
                    self.relationships[rel.relationship_id].is_active = False
                    self.graph.update_relationship(self.relationships[rel.relationship_id])