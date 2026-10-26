"""Attack Signature Library schemas.

Defines the generic, strictly-typed Pydantic models for Attack Signatures.
These signatures represent research-grounded constrained state graphs,
not deterministic scripts.
"""

from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, model_validator


class Observability(str, Enum):
    DIRECTLY_OBSERVABLE = "DIRECTLY_OBSERVABLE"
    INDIRECTLY_OBSERVABLE = "INDIRECTLY_OBSERVABLE"
    POTENTIALLY_UNOBSERVABLE = "POTENTIALLY_UNOBSERVABLE"


class SignalFamily(str, Enum):
    BEHAVIORAL = "BEHAVIORAL"
    DEVICE_SESSION = "DEVICE_SESSION"
    TRANSACTION = "TRANSACTION"
    BENEFICIARY = "BENEFICIARY"
    CONTEXT = "CONTEXT"
    VELOCITY = "VELOCITY"
    RELATIONSHIP = "RELATIONSHIP"


class ObservableConsequence(BaseModel):
    """An observable manifestation of an attack step."""
    description: str = Field(..., min_length=1)
    observability: Observability = Field(...)
    signal_families: List[SignalFamily] = Field(..., min_length=1)
    affected_entities: List[str] = Field(..., min_length=1)  # e.g., ["customer", "session"]


class AttackTransition(BaseModel):
    """A weighted transition between attack states."""
    target_state: str = Field(..., min_length=1)
    min_weight: float = Field(..., ge=0.0)
    max_weight: float = Field(..., ge=0.0)
    reason: str = Field(..., min_length=1)
    condition: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_weights(self) -> "AttackTransition":
        if self.min_weight > self.max_weight:
            raise ValueError(f"min_weight {self.min_weight} cannot exceed max_weight {self.max_weight}")
        return self


class AttackState(BaseModel):
    """A state within an attack state graph."""
    state_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    transitions: List[AttackTransition] = Field(default_factory=list)
    observable_consequences: List[ObservableConsequence] = Field(default_factory=list)
    affected_entities: List[str] = Field(default_factory=list)


class VariationAxis(BaseModel):
    """An axis of variation supported by this attack."""
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    allowed_values: List[str] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class AttackConstraint(BaseModel):
    """A simulation constraint the engine must respect."""
    description: str = Field(..., min_length=1)
    enforcement_layer: str = Field(..., min_length=1)


class ResearchSource(BaseModel):
    """A legitimate research source validating the attack behavior."""
    source_name: str = Field(..., min_length=1)
    source_url: Optional[str] = Field(default=None)
    title: str = Field(..., min_length=1)
    publication_year: int = Field(..., gt=1990)
    relevant_claim: str = Field(..., min_length=1)


class AttackSignature(BaseModel):
    """The complete specification for an Attack Signature."""
    attack_family: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    
    entry_states: List[str] = Field(..., min_length=1)
    states: Dict[str, AttackState] = Field(...)
    
    variation_axes: List[VariationAxis] = Field(default_factory=list)
    constraints: List[AttackConstraint] = Field(default_factory=list)
    research_sources: List[ResearchSource] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_graph(self) -> "AttackSignature":
        # Ensure entry states are defined
        for est in self.entry_states:
            if est not in self.states:
                raise ValueError(f"Entry state '{est}' not found in states.")
                
        # Validate transitions
        for state_name, state in self.states.items():
            if state.state_name != state_name:
                raise ValueError(f"State key '{state_name}' mismatch with state_name '{state.state_name}'.")
                
            for t in state.transitions:
                if t.target_state != "END" and t.target_state not in self.states:
                    raise ValueError(
                        f"Transition from '{state_name}' to unknown state '{t.target_state}'."
                    )
                    
        # (Optional) We could validate reachability, but the prompt says:
        # "Do not require all states to be reachable from every entry point."
        return self
