"""Tests for Stage 8 — Account Takeover Attack Signature."""

import pytest
from pydantic import ValidationError

from red_team.attacks.signature_library import (
    AttackSignature,
    AttackState,
    AttackTransition,
    ObservableConsequence,
    Observability,
    SignalFamily,
    VariationAxis,
    AttackConstraint,
    ResearchSource,
)
from red_team.attacks.ato_signature import get_ato_signature


def test_schema_valid_signature_creation():
    sig = get_ato_signature()
    assert sig.attack_family == "ACCOUNT_TAKEOVER"
    assert len(sig.states) == 5
    assert len(sig.entry_states) == 2


def test_schema_invalid_state_references_rejected():
    with pytest.raises(ValidationError, match="unknown state"):
        AttackSignature(
            attack_family="TEST",
            version="1.0",
            description="Test",
            entry_states=["A"],
            states={
                "A": AttackState(
                    state_name="A",
                    description="A",
                    transitions=[
                        AttackTransition(
                            target_state="B",  # B doesn't exist
                            min_weight=0, max_weight=1,
                            reason="test"
                        )
                    ]
                )
            },
            research_sources=[ResearchSource(source_name="X", title="X", publication_year=2020, relevant_claim="X")]
        )


def test_schema_invalid_entry_state_rejected():
    with pytest.raises(ValidationError, match="Entry state 'INVALID' not found"):
        AttackSignature(
            attack_family="TEST",
            version="1.0",
            description="Test",
            entry_states=["INVALID"],
            states={
                "A": AttackState(state_name="A", description="A", transitions=[])
            },
            research_sources=[ResearchSource(source_name="X", title="X", publication_year=2020, relevant_claim="X")]
        )


def test_schema_invalid_weight_ranges_rejected():
    with pytest.raises(ValidationError):
        AttackTransition(
            target_state="END",
            min_weight=0.9, max_weight=0.1,  # min > max
            reason="test"
        )


def test_schema_invalid_signal_family_rejected():
    with pytest.raises(ValidationError):
        ObservableConsequence(
            description="Test",
            observability=Observability.DIRECTLY_OBSERVABLE,
            signal_families=["INVALID_FAMILY"],  # type: ignore
            affected_entities=["session"]
        )


def test_graph_all_declared_states_represented():
    sig = get_ato_signature()
    expected_states = {"RECONNAISSANCE", "ACCOUNT_ACCESS", "ACCOUNT_MODIFICATION", "EXPLOITATION", "PERSISTENCE"}
    assert set(sig.states.keys()) == expected_states


def test_graph_valid_entry_states():
    sig = get_ato_signature()
    assert "RECONNAISSANCE" in sig.entry_states
    assert "ACCOUNT_ACCESS" in sig.entry_states


def test_graph_end_transitions_valid():
    sig = get_ato_signature()
    for state in sig.states.values():
        for t in state.transitions:
            if t.target_state == "END":
                assert t.min_weight >= 0
                assert t.max_weight >= 0


def test_optional_phases():
    sig = get_ato_signature()
    # verify that ACCOUNT_ACCESS can skip MODIFICATION and go to EXPLOITATION
    access_state = sig.states["ACCOUNT_ACCESS"]
    targets = [t.target_state for t in access_state.transitions]
    assert "ACCOUNT_MODIFICATION" in targets
    assert "EXPLOITATION" in targets  # skips modification


def test_loops_permitted():
    sig = get_ato_signature()
    # RECONNAISSANCE -> RECONNAISSANCE
    recon = sig.states["RECONNAISSANCE"]
    assert any(t.target_state == "RECONNAISSANCE" for t in recon.transitions)
    
    # EXPLOITATION -> EXPLOITATION
    exploit = sig.states["EXPLOITATION"]
    assert any(t.target_state == "EXPLOITATION" for t in exploit.transitions)


def test_observable_consequences():
    sig = get_ato_signature()
    for state in sig.states.values():
        for c in state.observable_consequences:
            assert isinstance(c.observability, Observability)
            assert len(c.signal_families) > 0
            assert len(c.affected_entities) > 0


def test_variation_axes():
    sig = get_ato_signature()
    assert len(sig.variation_axes) >= 7
    axis_names = [a.name for a in sig.variation_axes]
    assert "ENTRY_PATH" in axis_names
    assert "PHASE_SKIPPING" in axis_names
    assert "LOOPING" in axis_names


def test_constraints():
    sig = get_ato_signature()
    assert len(sig.constraints) >= 7
    for c in sig.constraints:
        assert len(c.description) > 0
        assert len(c.enforcement_layer) > 0


def test_research_sources():
    sig = get_ato_signature()
    assert len(sig.research_sources) > 0
    for rs in sig.research_sources:
        assert rs.publication_year > 1990


def test_serialization():
    sig = get_ato_signature()
    dump = sig.model_dump()
    sig_reloaded = AttackSignature.model_validate(dump)
    assert sig_reloaded.attack_family == sig.attack_family
    assert len(sig_reloaded.states) == len(sig.states)
