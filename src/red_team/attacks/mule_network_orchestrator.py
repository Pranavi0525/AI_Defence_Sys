"""Mule Network Orchestrator.

Coordinates several single-mule StatefulSimulator runs into one
MULE_NETWORK ring, genuinely correlated through a shared entity (a
common collector beneficiary_id, or a shared bank_id corridor) --
generated through the real event pipeline as each mule's trace is
built, not patched into already-generated traces after the fact.

This is the real replacement for quiet_ring_overlay.py's approach.
quiet_ring_overlay.py is honest that it isn't real Red Team output: it
rewrites device_id on already-generated LEGIT traces post-hoc. This
module instead runs MULE_NETWORK's own AttackSignature through the same
StatefulSimulator/WorldState path every other Red Team attack uses, so
the correlating signal (shared beneficiary_id or bank_id) exists
because real, distinct customers really transacted with it -- visible
to WorldState.append_event's normal relationship-wiring logic, not
invented by string-rewriting a JSON record afterward.
"""

import random
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel

from red_team.world.state import WorldState
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.attacks.mule_network_signature import get_mule_network_signature
from red_team.schemas.entities import Beneficiary
from red_team.schemas.observable import ObservableAttackTrace
from red_team.schemas.ground_truth import AttackGroundTruth


class MuleRingResult(BaseModel):
    """One generated mule ring: several correlated per-mule traces."""
    model_config = {"arbitrary_types_allowed": True}

    network_id: str
    correlation_type: Literal["shared_beneficiary", "shared_bank_corridor"]
    mule_customer_ids: List[str]
    traces: List[ObservableAttackTrace]
    ground_truths: List[AttackGroundTruth]


class MuleNetworkOrchestrator:
    """Coordinates several single-mule attack traces into one ring."""

    def __init__(self, state: WorldState, seed: int):
        self.state = state
        self.seed = seed
        self.rng = random.Random(seed)
        self.signature = get_mule_network_signature()

    def generate_ring(
        self,
        mule_customer_ids: List[str],
        difficulty: str = "medium",
        correlation_type: Literal["shared_beneficiary", "shared_bank_corridor"] = "shared_beneficiary",
    ) -> MuleRingResult:
        """Generate one mule ring across `mule_customer_ids`.

        Requires >= 2 distinct, already-populated customers (i.e.
        present in world_state.customers -- same precondition
        StatefulSimulator.generate_attack already enforces per-customer).

        Returns a MuleRingResult carrying one ObservableAttackTrace +
        AttackGroundTruth pair PER mule, plus network_id: the shared
        collector beneficiary_id (correlation_type="shared_beneficiary")
        or the shared bank_id string (correlation_type="shared_bank_corridor")
        that a downstream graph model can key on directly to recover the
        ring -- the same way build_cross_customer_graph() keys on
        device_id today, but for a real generated signal instead of an
        overlaid one.
        """
        if len(set(mule_customer_ids)) < 2:
            raise ValueError("A mule network requires at least 2 distinct customers.")

        for cid in mule_customer_ids:
            if cid not in self.state.customers:
                raise ValueError(f"Customer {cid} not found in WorldState")

        collector_beneficiary: Optional[Beneficiary] = None
        shared_bank_id: Optional[str] = None

        if correlation_type == "shared_beneficiary":
            collector_beneficiary = Beneficiary(
                name="Collector Account",
                account_reference=f"COLLECTOR_REF_{self.rng.randint(100000, 999999)}",
                created_date=self.state.current_time,
                relationship_type="other",
                is_verified=False,
            )
            # Registered once, shared across every mule's trace below --
            # this is what makes the correlation real rather than cosmetic.
            self.state.beneficiaries[collector_beneficiary.beneficiary_id] = collector_beneficiary
            network_id = collector_beneficiary.beneficiary_id
        else:
            shared_bank_id = f"MULE_CORRIDOR_BANK_{self.rng.randint(1, 999)}"
            network_id = shared_bank_id

        traces: List[ObservableAttackTrace] = []
        ground_truths: List[AttackGroundTruth] = []

        for i, customer_id in enumerate(mule_customer_ids):
            if shared_bank_id is not None:
                self._apply_bank_corridor(customer_id, shared_bank_id)

            sim = StatefulSimulator(self.state, self.signature, seed=self.seed + i)
            plan = AttackPlan(
                attack_family="MULE_NETWORK",
                difficulty=difficulty,
                max_phases=6,
                max_events=15,
                network_id=network_id,
                mule_hop_index=i,
            )
            trace, ground_truth = sim.generate_attack(
                plan, customer_id, forced_beneficiary=collector_beneficiary
            )
            traces.append(trace)
            ground_truths.append(ground_truth)

        return MuleRingResult(
            network_id=network_id,
            correlation_type=correlation_type,
            mule_customer_ids=list(mule_customer_ids),
            traces=traces,
            ground_truths=ground_truths,
        )

    def _apply_bank_corridor(self, customer_id: str, shared_bank_id: str) -> None:
        """Retarget one of this customer's real accounts onto the shared
        cross-bank corridor, so a network-level (multi-bank) view can see
        the correlation that no single institution's own data would."""
        for account in self.state.accounts.values():
            if account.customer_id == customer_id:
                account.bank_id = shared_bank_id
                return
        raise ValueError(f"Customer {customer_id} has no account to retarget onto {shared_bank_id}")
