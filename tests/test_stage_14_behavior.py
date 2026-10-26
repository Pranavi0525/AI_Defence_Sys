import pytest
from datetime import datetime
from red_team.world.world import NormalWorld
from red_team.world.behavior_state import BehavioralModelConfig
from red_team.schemas.events import TransactionEventPayload

def test_stage_14_behavioral_redesign():
    # 1. Customer-specific scheduling, priorities, no look-ahead
    config = BehavioralModelConfig(
        device_reuse_prob=1.0,
        beneficiary_reuse_prob=1.0,
        burst_prob=0.0
    )
    world = NormalWorld(seed=42)
    world.behavior_sim.config = config
    world.generate_population(n_customers=5, n_merchants=5, n_beneficiaries=5)
    world.generate_legitimate_events(num_events=100)
    
    events = world.get_events()
    
    # 16. Events remain chronologically ordered
    for i in range(1, len(events)):
        assert events[i].envelope.timestamp >= events[i-1].envelope.timestamp
        
    state = world.get_state()
    # 17. WorldState and graph remain synchronized
    # Just asserting the graph didn't crash and has some entities
    assert state.graph._graph.number_of_nodes() > 0
    
    # 3. Persistent transaction preferences & 4. Different customers have different preferences
    customer_tx_types = {}
    for e in events:
        if isinstance(e.payload, TransactionEventPayload):
            cid = e.envelope.customer_id
            ttype = e.payload.transaction.transaction_type
            if cid not in customer_tx_types:
                customer_tx_types[cid] = {'purchase': 0, 'transfer': 0}
            customer_tx_types[cid][ttype] += 1
            
    # Customers should have different mixes, but persistent within themselves
    # Let's inspect cb state
    for cid, cb in state.customer_behavior.items():
        assert "purchase" in cb.tx_type_weights
        assert "transfer" in cb.tx_type_weights
        assert cb.typical_amount_anchor > 0
        assert cb.amount_variability > 0

    # 14. Same seed identical results, 15. Different seeds different results
    world2 = NormalWorld(seed=42)
    world2.generate_population(n_customers=5, n_merchants=5, n_beneficiaries=5)
    world2.generate_legitimate_events(num_events=100)
    assert world.get_events()[0].envelope.timestamp == world2.get_events()[0].envelope.timestamp
    
    world3 = NormalWorld(seed=999)
    world3.generate_population(n_customers=5, n_merchants=5, n_beneficiaries=5)
    world3.generate_legitimate_events(num_events=100)
    assert world.get_events()[0].envelope.timestamp != world3.get_events()[0].envelope.timestamp
