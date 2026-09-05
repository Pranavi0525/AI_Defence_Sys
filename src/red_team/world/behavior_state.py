from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class BehavioralModelConfig(BaseModel):
    """Configurable parameters for behavioral modeling assumptions.
    
    All values are DOMAIN_MODELED and should be updated when empirical data becomes available.
    """
    device_reuse_prob: float = Field(default=0.90, description="DOMAIN_MODELED: Probability of reusing primary device.")
    beneficiary_reuse_prob: float = Field(default=0.95, description="DOMAIN_MODELED: Probability of reusing known beneficiary.")
    beneficiary_addition_prob: float = Field(default=0.03, description="DOMAIN_MODELED: Probability, within an active session, of adding a brand-new beneficiary instead of transacting or logging out.")
    burst_prob: float = Field(default=0.10, description="DOMAIN_MODELED: Probability of entering a burst mode.")
    burst_time_multiplier: float = Field(default=0.05, description="DOMAIN_MODELED: Timing multiplier during a burst.")
    amount_variance_factor: float = Field(default=0.20, description="DOMAIN_MODELED: Standard deviation as a fraction of mean.")
    drift_prob: float = Field(default=0.01, description="DOMAIN_MODELED: Probability of generating a drift event instead of normal behavior.")

class CustomerBehaviorState(BaseModel):
    """Persistent state driving a specific customer's chronological behavior."""
    customer_id: str
    next_event_time: datetime
    tx_type_weights: Dict[str, float]
    typical_amount_anchor: float
    amount_variability: float
    
    primary_device_id: Optional[str] = None
    beneficiary_affinities: Dict[str, int] = Field(default_factory=dict)
    
    in_burst: bool = False
    burst_events_remaining: int = 0
    
    recent_tx_types: List[str] = Field(default_factory=list)
    recent_amounts: List[float] = Field(default_factory=list)
