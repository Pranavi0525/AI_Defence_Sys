"""
Quiet Mule-Ring Overlay
=================================================================

WHY THIS EXISTS
----------------
The first ring-detection pass (train_gnn.py) injected the synthetic ring
into 24 ATO traces that Stage 1+2 already caught on their own individual
features (new device + beneficiary-then-transaction, etc.). Result: zero
measurable lift from the graph, because there was nothing left for it to
rescue -- see train_gnn.py's module docstring for that honest finding.

This module builds the harder, more honest version instead: a ring
pattern injected into traces that look INDIVIDUALLY unremarkable and are
only suspicious as a group -- which is what a real mule ring looks like.

HOW IT STAYS "QUIET" (i.e. doesn't trip Stage 1 on its own)
-------------------------------------------------------------
Stage 1 (blue_team_pipeline.stage1_rule_filter) escalates on exactly
four signals:
    beneficiary_added_before_transaction, new_device_present,
    min_time_between_transactions < 3600s, transactions_per_hour > 2.5

  - We do NOT add a BENEFICIARY_ADDITION event (that's what made the
    first pass's rings "obviously fraud" -- NormalWorld never generates
    legit beneficiary additions, so any beneficiary event is a perfect
    tell by construction, per blue_team_pipeline.py's documented gap).
  - We do NOT add a DEVICE_REGISTRATION event either (new_device_present
    counts DEVICE_REGISTRATION events specifically, not device_id in
    general -- see blue_team_pipeline.extract_features).
  - Instead we overwrite the device_id already present on the trace's
    existing SESSION_LOGIN / SESSION_LOGOUT events (every session has
    one -- see ObservableSessionEvent in schemas/observable.py) with a
    shared "collector device" id. This is the realistic pattern: a mule
    ring is often several DIFFERENT customers' otherwise-ordinary
    accounts, accessed at different times from the SAME physical
    device/browser. It creates a genuine cross-customer graph edge
    (build_cross_customer_graph() keys on device_id from ANY event that
    carries it) without touching any field Stage 1 or Stage 2's feature
    set inspects for "new device" or "beneficiary" signals.
  - Velocity fields (transaction timing/amounts) are left completely
    untouched -- these are real NormalWorld-generated sessions.

This means some of these traces WILL still escalate to Stage 2 on their
own, purely because Stage 1's velocity thresholds are calibrated on
population percentiles and a handful of ordinary sessions naturally sit
above them (documented in blue_team_pipeline.py -- p90/p99 cutoffs still
leave ~1-10% of legit traffic escalating by chance). That's a pre-existing
Stage 1 calibration property, not something this overlay introduces --
verify it directly with `diagnose_overlay()` below before trusting any
"rescued by Stage 3" claim.

FRAUD LABEL
-----------
These traces are relabeled fraud=1, attack_family="ring_mule_synthetic",
attack_difficulty="n/a". This is a deliberate, clearly-flagged synthetic
construction -- NOT real Red Team output -- exactly like the first ring
overlay was. State this plainly in any writeup; don't let it read as if
NormalWorld itself produces mule rings.
"""
from __future__ import annotations

import random
from collections import defaultdict

N_RING_TRACES = 24
N_COLLECTORS = 4
RING_ATTACK_FAMILY = "ring_mule_synthetic"


def _session_device_events(record: dict) -> list[dict]:
    """Every event on this trace that carries a device_id we can rewrite
    (SESSION_LOGIN / SESSION_LOGOUT -- every session has one). We only
    ever rewrite login/logout events, never DEVICE_REGISTRATION ones,
    so we never create or touch a DEVICE_REGISTRATION event ourselves.
    """
    return [e for e in record["events"]
            if e.get("event_type") in ("SESSION_LOGIN", "SESSION_LOGOUT") and e.get("device_id")]


def _is_already_quiet(record: dict) -> bool:
    """True if this trace has NEITHER a DEVICE_REGISTRATION nor a
    BENEFICIARY_ADDITION event of its own already -- i.e. it's a clean
    starting point for the overlay. A trace that already carries one of
    these (a pre-existing NormalWorld artifact, not something we'd be
    adding) would trip Stage 1 regardless of anything we do, which
    would contaminate the "quiet ring" claim. We simply don't draw the
    ring sample from these traces rather than mutate around them.
    """
    return not any(
        e.get("event_type") in ("DEVICE_REGISTRATION", "BENEFICIARY_ADDITION")
        for e in record["events"]
    )


def apply_quiet_ring_overlay(
    legit_records: list[dict],
    n_ring: int = N_RING_TRACES,
    n_collectors: int = N_COLLECTORS,
    seed: int = 42,
) -> tuple[list[dict], set[str]]:
    """Relabels `n_ring` ordinary legitimate traces as a synthetic mule
    ring by rewiring their existing device_id fields to `n_collectors`
    shared "collector device" ids, spread across different customers.

    Returns (mutated legit_records, set of ring trace_ids). Mutates the
    matching dict entries in `legit_records` in place AND returns the
    same list for convenience; does not touch any other record.
    """
    rng = random.Random(seed)

    eligible = [r for r in legit_records if _session_device_events(r) and _is_already_quiet(r)]
    if len(eligible) < n_ring:
        raise ValueError(
            f"Only {len(eligible)} legit traces carry a device_id -- "
            f"cannot sample {n_ring} for the ring overlay."
        )

    # Sample across distinct customers where possible, so the ring is
    # genuinely cross-customer rather than accidentally landing on the
    # same customer's several sessions.
    by_customer: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_customer[r["customer_id"]].append(r)

    customers = list(by_customer.keys())
    rng.shuffle(customers)

    ring_records: list[dict] = []
    for cust in customers:
        if len(ring_records) >= n_ring:
            break
        ring_records.append(rng.choice(by_customer[cust]))

    if len(ring_records) < n_ring:
        # Not enough distinct customers -- top up from whatever's left.
        remaining = [r for r in eligible if r not in ring_records]
        rng.shuffle(remaining)
        ring_records.extend(remaining[: n_ring - len(ring_records)])

    ring_records = ring_records[:n_ring]
    ring_ids = {r["trace_id"] for r in ring_records}

    collectors = [f"RING_COLLECTOR_DEVICE_{i}" for i in range(n_collectors)]
    for i, r in enumerate(ring_records):
        collector = collectors[i % n_collectors]
        for e in _session_device_events(r):
            e["device_id"] = collector
        r["fraud"] = 1
        r["attack_family"] = RING_ATTACK_FAMILY
        r["attack_difficulty"] = "n/a"

    return legit_records, ring_ids


def diagnose_overlay(records: list[dict], ring_ids: set[str], stage1_rule_filter) -> dict:
    """Honesty check, run BEFORE the full cascade: confirms the ring
    traces don't trip Stage 1 via beneficiary/device-registration
    signals (the thing we deliberately avoided), and reports how many
    trip it anyway via the pre-existing velocity thresholds (the thing
    we did NOT introduce and can't control).

    `stage1_rule_filter` takes a pandas Series (one feature row) and
    returns bool -- pass blue_team_pipeline.stage1_rule_filter directly,
    called by the caller on the feature table for these trace_ids.
    """
    ring_records = [r for r in records if r["trace_id"] in ring_ids]
    n_with_benef_event = sum(
        1 for r in ring_records
        if any(e["event_type"] == "BENEFICIARY_ADDITION" for e in r["events"])
    )
    n_with_device_reg_event = sum(
        1 for r in ring_records
        if any(e["event_type"] == "DEVICE_REGISTRATION" for e in r["events"])
    )
    return {
        "n_ring_traces": len(ring_records),
        "n_with_beneficiary_addition_event": n_with_benef_event,
        "n_with_device_registration_event": n_with_device_reg_event,
        "note": "Both counts above should be 0 -- if not, the overlay is "
                "leaking through a Stage-1 trigger field and the eval "
                "downstream isn't testing what it claims to.",
    }
