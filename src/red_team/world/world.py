"""Orchestrator API for the Minimal Normal World."""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple

from red_team.schemas.events import Event
from red_team.schemas.id_generator import seed_ids
from red_team.world.state import WorldState
from red_team.world.persona import PersonaParameters, get_default_personas
from red_team.world.entity_generator import EntityGenerator
from red_team.world.behavior import BehavioralSimulator


logger = logging.getLogger(__name__)


class NormalWorld:
    """The Legitimate Normal World generator."""

    def __init__(self, seed: int = 42, start_time: datetime = None):
        self.seed = seed
        # Makes every entity_id/event_id generated from here on a
        # deterministic function of `seed`, not just the behavioral
        # choices -- see red_team/schemas/id_generator.py.
        seed_ids(self.seed)
        self.start_time = start_time or datetime(2025, 1, 1, 0, 0, 0)
        self.personas = get_default_personas()
        
        self.state = WorldState(current_time=self.start_time)
        self.entity_gen = EntityGenerator(random_seed=self.seed, start_time=self.start_time)
        self.behavior_sim = BehavioralSimulator(random_seed=self.seed, personas=self.personas)

    def generate_population(
        self,
        n_customers: int = 100,
        n_merchants: int = 20,
        n_beneficiaries: int = 50,
        bank_pool: List[str] | None = None,
        cross_bank_rate: float = 0.15,
    ) -> None:
        """Populate the world with synthetic entities.

        bank_pool: simulated banks accounts can belong to. Defaults to a
            single bank (single-institution simulation, backward compatible).
            Pass e.g. ["BANK_A", "BANK_B", "BANK_C"] to simulate a multi-bank
            population.
        cross_bank_rate: see EntityGenerator.generate_population.
        """
        logger.info(f"Generating population with {n_customers} customers.")
        
        custs, accts, devs, merchs, bens, rels = self.entity_gen.generate_population(
            self.personas, n_customers, n_merchants, n_beneficiaries,
            bank_pool=bank_pool, cross_bank_rate=cross_bank_rate,
        )
        
        self.state.customers = {c.customer_id: c for c in custs}
        self.state.accounts = {a.account_id: a for a in accts}
        self.state.devices = {d.device_id: d for d in devs}
        self.state.merchants = {m.merchant_id: m for m in merchs}
        self.state.beneficiaries = {b.beneficiary_id: b for b in bens}
        self.state.relationships = {r.relationship_id: r for r in rels}
        
        # Populate customer_devices index mapping generically 
        # (Assuming all created devices belong to customers sequentially/randomly)
        # For simplicity in this slice, we will randomly assign devices to customers 
        # based on the Entity Generator's intended distribution.
        
        # In entity_generator, we actually generated devices per customer but didn't map them explicitly.
        # Let's fix that mapping generically:
        dev_list = list(self.state.devices.keys())
        import random
        rng = random.Random(self.seed)
        for c_id in self.state.customers:
            # assign 1 device randomly
            d_id = rng.choice(dev_list)
            self.state.customer_devices[c_id] = [d_id]
            
        # Sync to graph
        for c in custs:
            self.state.graph.add_entity("customer", c.customer_id)
        for a in accts:
            self.state.graph.add_entity("account", a.account_id)
        for d in devs:
            self.state.graph.add_entity("device", d.device_id)
        for m in merchs:
            self.state.graph.add_entity("merchant", m.merchant_id)
        for b in bens:
            self.state.graph.add_entity("beneficiary", b.beneficiary_id)
        for r in rels:
            self.state.graph.add_relationship(r)
            
    def generate_legitimate_events(self, num_events: int = 1000) -> None:
        """Generate a chronological sequence of legitimate events."""
        logger.info(f"Generating {num_events} legitimate events.")
        
        for _ in range(num_events):
            event = self.behavior_sim.generate_next_event(self.state)
            if event:
                self.state.append_event(event)
                
    def get_state(self) -> WorldState:
        """Return the current WorldState."""
        return self.state

    def get_events(self) -> List[Event]:
        """Return the history of generated events."""
        return self.state.event_history
