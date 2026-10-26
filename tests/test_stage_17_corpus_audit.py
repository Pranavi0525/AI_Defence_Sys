import pytest
from scripts.run_stage_17_duplicates import extract_structured_plan, calculate_similarity, StructuredPlan

class MockEvent:
    def __init__(self, event_type, transaction_status="none"):
        self.event_type = event_type
        self.transaction_status = transaction_status

class MockTrace:
    def __init__(self, events):
        self.events = events

class MockPhase:
    def __init__(self, phase):
        self.phase = phase

class MockGT:
    def __init__(self, phases_executed, attack_family="ACCOUNT_TAKEOVER", attack_difficulty="easy"):
        self.phases_executed = phases_executed
        self.attack_family = attack_family
        self.attack_difficulty = attack_difficulty

class MockRecord:
    def __init__(self, trace, gt):
        self.observable_trace = trace
        self.ground_truth = gt

def test_extract_structured_plan():
    rec = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN"), MockEvent("TRANSACTION")]),
        MockGT([MockPhase("ACCOUNT_ACCESS"), MockPhase("EXPLOITATION"), MockPhase("EXPLOITATION")])
    )
    plan = extract_structured_plan(rec)
    
    assert plan.attack_family == "ACCOUNT_TAKEOVER"
    assert plan.entry_path == "ACCOUNT_ACCESS"
    assert set(plan.phases) == {"ACCOUNT_ACCESS", "EXPLOITATION"}
    assert plan.phase_order == ["ACCOUNT_ACCESS", "EXPLOITATION", "EXPLOITATION"]
    assert plan.loops == 1
    assert plan.path_length == 3
    assert set(plan.affected_entity_categories) == {"SESSION_LOGIN", "TRANSACTION"}

def test_calculate_similarity_identical():
    rec1 = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN"), MockEvent("TRANSACTION")]),
        MockGT([MockPhase("ACCOUNT_ACCESS"), MockPhase("EXPLOITATION")])
    )
    rec2 = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN"), MockEvent("TRANSACTION")]),
        MockGT([MockPhase("ACCOUNT_ACCESS"), MockPhase("EXPLOITATION")])
    )
    
    sim = calculate_similarity(rec1, rec2)
    assert sim == 1.0

def test_calculate_similarity_different():
    rec1 = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN")]),
        MockGT([MockPhase("ACCOUNT_ACCESS")])
    )
    rec2 = MockRecord(
        MockTrace([MockEvent("TRANSACTION")]),
        MockGT([MockPhase("EXPLOITATION")])
    )
    
    sim = calculate_similarity(rec1, rec2)
    assert sim == 0.0

def test_calculate_similarity_partial():
    rec1 = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN"), MockEvent("TRANSACTION")]),
        MockGT([MockPhase("ACCOUNT_ACCESS"), MockPhase("EXPLOITATION")])
    )
    rec2 = MockRecord(
        MockTrace([MockEvent("SESSION_LOGIN"), MockEvent("BENEFICIARY_ADDITION")]),
        MockGT([MockPhase("ACCOUNT_ACCESS"), MockPhase("MODIFICATION")])
    )
    
    sim = calculate_similarity(rec1, rec2)
    # Entry is the same (ACCOUNT_ACCESS).
    # phase_seq is different (0.0).
    # phase_set_sim: {ACCESS, EXPL} vs {ACCESS, MOD} -> 1/3
    # event_seq is different (0.0).
    # outcome_sim: [none] vs [none] (Wait, rec2 has NO transaction. o1=[none], o2=[]. Different.) -> 0.0
    # Expected sim: (0.0 + 1/3 + 0.0 + 1.0 + 0.0) / 5.0 = (1.333) / 5.0 = 0.2666...
    assert abs(sim - (1.3333333333333333 / 5.0)) < 0.01

