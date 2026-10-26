"""Tests for Stage 6 — Relationship Graph."""

import pytest
from datetime import datetime
import uuid

from red_team.world.world import NormalWorld
from red_team.world.graph import validate_consistency
from red_team.schemas.entities import Customer, Device, Relationship
from red_team.schemas.events import (
    Event, EventEnvelope, EventType,
    DeviceEventPayload, RelationshipEventPayload
)


def test_graph_initialization_sync():
    world = NormalWorld(seed=123)
    world.generate_population(n_customers=10, n_merchants=5, n_beneficiaries=10)
    
    state = world.get_state()
    graph = state.graph
    
    # Check graph sync
    val = validate_consistency(state)
    assert val["is_consistent"], f"Graph inconsistent: {val}"
    
    # Check basic stats
    stats = graph.get_statistics()
    assert stats["node_count"] > 0
    assert stats["edge_count"] > 0
    assert stats["nodes_by_type"].get("customer", 0) == 10
    
    # Ensure domain metrics are calculated
    assert "devices_per_customer" in stats
    assert "beneficiaries_per_customer" in stats


def test_device_event_updates_graph():
    world = NormalWorld(seed=123)
    world.generate_population(n_customers=2, n_merchants=1, n_beneficiaries=1)
    
    state = world.get_state()
    customer_id = list(state.customers.keys())[0]
    
    dev = Device(
        device_type="mobile",
        fingerprint="TEST_FINGERPRINT",
        first_seen=state.current_time,
        last_seen=state.current_time,
        is_trusted=False
    )
    
    envelope = EventEnvelope(
        timestamp=state.current_time,
        event_type=EventType.DEVICE_REGISTRATION,
        customer_id=customer_id
    )
    payload = DeviceEventPayload(device=dev, action="register")
    event = Event(envelope=envelope, payload=payload)
    
    # Pre-check graph
    assert not state.graph.has_entity("device", dev.device_id)
    
    # Append event
    state.append_event(event)
    
    # Post-check graph
    assert state.graph.has_entity("device", dev.device_id)
    
    neighbors = state.graph.get_neighbors("customer", customer_id)
    assert f"device:{dev.device_id}" in neighbors
    
    # Check consistency
    val = validate_consistency(state)
    assert val["is_consistent"]


def test_transaction_updates_graph_last_activity():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=5)
    
    state = world.get_state()
    
    # Initial graph state
    initial_stats = state.graph.get_statistics()
    
    # Generate behavior
    world.generate_legitimate_events(num_events=50)
    
    # Check consistency
    val = validate_consistency(state)
    assert val["is_consistent"], f"Inconsistent after events: {val}"
    
    post_stats = state.graph.get_statistics()
    
    # We expect edges to maybe increase if they transacted with new merchants
    # But node_count should basically only increase if new drift devices appeared
    
    # Find a relationship and check its last activity
    rels = list(state.relationships.values())
    tx_rels = [r for r in rels if r.relationship_type == "transacts_with"]
    
    if tx_rels:
        r = tx_rels[0]
        # graph should have updated date
        g_rel = state.graph.get_relationship(r.relationship_id)
        assert g_rel["last_activity_date"] == r.last_activity_date


def test_relationship_removal_updates_graph():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    state = world.get_state()
    
    rel_id = list(state.relationships.keys())[0]
    rel = state.relationships[rel_id]
    
    assert state.graph.has_relationship(rel_id)
    
    envelope = EventEnvelope(
        timestamp=state.current_time,
        event_type=EventType.RELATIONSHIP_CHANGE,
        customer_id=rel.source_entity_id
    )
    payload = RelationshipEventPayload(relationship=rel, action="terminate")
    event = Event(envelope=envelope, payload=payload)
    
    state.append_event(event)
    
    # Checking state. The relationship should still exist but be inactive.
    g_rel = state.graph.get_relationship(rel_id)
    assert not g_rel["is_active"]
    assert not state.relationships[rel_id].is_active
    
    val = validate_consistency(state)
    assert val["is_consistent"]
