"""Persona generator for the Normal World.

Personas define the legitimate behavioral profiles of synthetic customers.
"""

from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field

class PersonaParameters(BaseModel):
    """Parameters governing a customer's legitimate behavior."""
    
    segment_id: str = Field(..., description="Unique persona segment name.")
    
    # Financial behavior
    typical_amount_range: tuple[float, float] = Field(..., description="Min/max range for typical transactions.")
    tx_frequency_per_week: tuple[int, int] = Field(..., description="Min/max typical transactions per week.")
    
    # Structural tendencies
    device_count_tendency: Literal["single", "multiple"] = Field(..., description="Likelihood of using multiple devices.")
    beneficiary_count_range: tuple[int, int] = Field(..., description="How many beneficiaries they maintain.")
    
    # Preferences
    preferred_channels: List[str] = Field(..., description="Channels used by this persona.")
    
    # Provenance tags
    provenance_notes: str = Field(..., description="Explain where these parameters come from.")


def get_default_personas() -> List[PersonaParameters]:
    """Return the baseline persona configurations for the Normal World.
    
    Note: Many parameters are DOMAIN_MODELED since the raw datasets (PaySim)
    do not contain explicit device or rich multi-channel behavior.
    """
    return [
        PersonaParameters(
            segment_id="LOW_FREQUENCY",
            typical_amount_range=(10.0, 500.0),
            tx_frequency_per_week=(1, 3),
            device_count_tendency="single",
            beneficiary_count_range=(1, 2),
            preferred_channels=["mobile"],
            provenance_notes="DOMAIN_MODELED: Represents minimal active users.",
        ),
        PersonaParameters(
            segment_id="REGULAR_CONSUMER",
            typical_amount_range=(50.0, 2000.0),
            tx_frequency_per_week=(5, 12),
            device_count_tendency="multiple",
            beneficiary_count_range=(3, 10),
            preferred_channels=["mobile", "web"],
            provenance_notes="DOMAIN_MODELED: Represents average active users.",
        ),
        PersonaParameters(
            segment_id="HIGH_FREQUENCY",
            typical_amount_range=(20.0, 5000.0),
            tx_frequency_per_week=(15, 30),
            device_count_tendency="multiple",
            beneficiary_count_range=(5, 15),
            preferred_channels=["mobile", "pos", "web"],
            provenance_notes="DOMAIN_MODELED: Represents very active or power users.",
        ),
        PersonaParameters(
            segment_id="DIGITAL_HEAVY",
            typical_amount_range=(50.0, 3000.0),
            tx_frequency_per_week=(10, 25),
            device_count_tendency="multiple",
            beneficiary_count_range=(2, 8),
            preferred_channels=["mobile", "api"],
            provenance_notes="DOMAIN_MODELED: Represents users interacting mostly through mobile/API.",
        )
    ]
