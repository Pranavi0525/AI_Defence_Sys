"""
MULE_NETWORK Corpus Generation & Qualification
================================================

Generates and freezes the third Red Team attack family corpus --
MULE_NETWORK -- using the exact same pattern already proven for
ACCOUNT_TAKEOVER / AUTHORIZED_PUSH_PAYMENT in
run_final_red_team_qualification.py: a world with trusted devices and
funded accounts, individual trace validation against realism +
leakage/chronology invariants, then a frozen JSON handoff.

Output:
  - reports/mule_corpus_raw.json        (accepted traces, Blue Team
    consumable -- same {"observable_trace", "ground_truth"} shape as
    ato_corpus_raw.json / app_corpus_raw.json)
  - reports/stage_32_mule_corpus_freeze.md   (qualification note)

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 scripts/run_mule_network_qualification.py
"""
import json
import logging
import time
from collections import Counter
from decimal import Decimal

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_mule_network_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MULE_TARGET_TRACES = 120     # comparable order of magnitude to ATO(97)/APP(156)
MULE_SEED = 44                 # distinct from ato_seed=42, app_seed=43
WORLD_SEED = 100                # same world seed used by the ATO/APP qualification run
N_CUSTOMERS = 60                  # enough pool for varied 2-4 person rings


def setup_world(seed: int, n_customers: int = N_CUSTOMERS, n_events: int = 200):
    """Identical setup to run_final_red_team_qualification.py's setup_world:
    trusted device per customer, funded accounts. Reused verbatim rather
    than re-derived, so the MULE_NETWORK corpus is validated against the
    same kind of world ATO/APP already were.
    """
    world = NormalWorld(seed=seed)
    world.generate_population(n_customers=n_customers)
    world.generate_legitimate_events(num_events=n_events)
    ws = world.get_state()
    from red_team.schemas.entities import Device, Relationship
    for c_id in ws.customers:
        dev_id = f"dev_{c_id[:5]}"
        ws.devices[dev_id] = Device(
            device_id=dev_id, device_type="mobile", fingerprint="fp1",
            first_seen=ws.current_time, last_seen=ws.current_time, is_trusted=True
        )
        rel_id = f"rel_{c_id[:5]}"
        ws.relationships[rel_id] = Relationship(
            relationship_id=rel_id, source_entity_type="customer", source_entity_id=c_id,
            target_entity_type="device", target_entity_id=dev_id, relationship_type="owns",
            established_date=ws.current_time
        )
    for acct in ws.accounts.values():
        acct.balance = Decimal("25000.00")
    return ws


def verify_invariants_and_leakage(trace_events):
    """Same check ATO/APP traces were qualified against: no planner/ground-
    truth fields leaked into the observable trace, and events are in
    chronological order."""
    forbidden_keys = [
        "attack_family", "difficulty", "variation_profile",
        "planner", "strategy", "target_amount", "logical_split", "sim_"
    ]
    timestamps = []
    for event in trace_events:
        timestamps.append(event.timestamp)
        dump = event.model_dump()
        for k in dump.keys():
            for fb in forbidden_keys:
                if fb in k.lower():
                    return False, "LEAKAGE"
    if sorted(timestamps) != timestamps:
        return False, "CHRONOLOGY"
    return True, "CLEAN"


def save_corpus(res, filepath):
    traces = []
    for t in res.accepted_traces:
        traces.append({
            "observable_trace": t.observable_trace.model_dump(mode="json"),
            "ground_truth": t.ground_truth.model_dump(mode="json"),
        })
    with open(filepath, "w") as f:
        json.dump(traces, f, indent=2)
    return len(traces)


def run():
    logging.info("Starting MULE_NETWORK corpus generation + qualification...")
    ws = setup_world(WORLD_SEED)

    t0 = time.time()
    res = generate_mule_network_corpus(
        world_state=ws,
        target_traces=MULE_TARGET_TRACES,
        master_seed=MULE_SEED,
    )
    duration = time.time() - t0

    n_saved = save_corpus(res, "reports/mule_corpus_raw.json")

    difficulty_counts = Counter(t.ground_truth.attack_difficulty for t in res.accepted_traces)
    network_ids = {
        t.ground_truth.planner_metadata.plan_json["network_id"]
        for t in res.accepted_traces
        if t.ground_truth.planner_metadata
        and t.ground_truth.planner_metadata.plan_json.get("network_id")
    }

    clean = True
    failed_reason = None
    for t in res.accepted_traces:
        c, r = verify_invariants_and_leakage(t.observable_trace.events)
        if not c:
            clean = False
            failed_reason = r
            break

    acceptance_rate_vs_rings = round(len(res.accepted_traces) / max(len(network_ids), 1), 2)

    report = f"""# Stage 32 -- MULE_NETWORK Corpus Freeze

Status: {"QUALIFIED for freeze" if clean else "BLOCKED -- leakage/chronology failure"}
Generated: reports/mule_corpus_raw.json
Duration: {duration:.1f}s

## Generation parameters
- world_seed: {WORLD_SEED}, n_customers: {N_CUSTOMERS}
- master_seed: {MULE_SEED}
- target_traces: {MULE_TARGET_TRACES}
- attempted: {res.generation_statistics.attempted}

## Results
- accepted traces: {len(res.accepted_traces)}
- rejected attempts: {len(res.rejected_attempts)}
- distinct rings (network_id) recovered from flat storage: {len(network_ids)}
- traces per ring (accepted / distinct rings): {acceptance_rate_vs_rings}
  (NOTE: this ratio is expected to exceed 1 -- each accepted ring
  contributes 2-4 individual mule traces, unlike ATO/APP where one
  attempt == one trace. Not a correctness bug, see corpus.py docstring.)
- difficulty distribution: {dict(difficulty_counts)}
- invariant + leakage check: {"CLEAN" if clean else f"FAILED ({failed_reason})"}

## Integration note
Every trace here is stored as an ordinary AttackRecord
({{"observable_trace", "ground_truth"}}), identical in shape to
ato_corpus_raw.json / app_corpus_raw.json. Blue Team's load_attack_corpus()
requires zero special-case code to consume this file -- rings remain
recoverable downstream via ground_truth.planner_metadata.plan_json.network_id
for anyone who wants ring-level (not just trace-level) analysis.
"""
    with open("reports/stage_32_mule_corpus_freeze.md", "w") as f:
        f.write(report)

    logging.info(
        f"Done. accepted={len(res.accepted_traces)} rejected={len(res.rejected_attempts)} "
        f"rings={len(network_ids)} clean={clean} saved_to=reports/mule_corpus_raw.json"
    )
    return res


if __name__ == "__main__":
    run()
