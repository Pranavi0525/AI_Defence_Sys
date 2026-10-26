import pytest
from datetime import datetime
from red_team.world.world import NormalWorld

def test_device_metric_extraction():
    # Setup world
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=5)
    world.generate_legitimate_events(num_events=50)
    
    events = world.get_events()
    
    # Run the same extraction logic
    session_to_device = {}
    for e in events:
        if e.envelope.event_type.value == "SESSION_LOGIN":
            session_to_device[e.envelope.session_id] = e.payload.session.device_id
            
    device_extracted_count = 0
    for e in events:
        if e.envelope.event_type.value == "TRANSACTION":
            sid = e.envelope.session_id
            dev_id = session_to_device.get(sid)
            if dev_id is not None:
                device_extracted_count += 1
                
    # Prove that we successfully extracted device IDs for transactions
    assert device_extracted_count > 0
