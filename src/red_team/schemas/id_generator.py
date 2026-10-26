"""Deterministic, seedable ID generation shared by entities.py and events.py.

WHY THIS EXISTS
----------------
Every entity (Customer, Account, Device, ...) and every Event previously
got its ID from a bare `uuid.uuid4()` call. uuid4() draws from the OS's
random source, completely independent of any `seed` passed to
NormalWorld or a corpus generator's `master_seed`. That RNG only ever
controlled *behavioral* choices (which event fires, transaction
amounts, timing) -- the ID *strings* were random every single process
run regardless of seed. Two runs with `NormalWorld(seed=7)` produced two
completely different populations of customer_ids/account_ids/event_ids,
which silently broke every downstream "same seed -> same result" claim
(CV fold assignment via trace_id hashing, per-customer feature
baselines, frozen-corpus network_id values on regeneration, etc.).

This module gives every entity/event ID generator a single shared,
reseedable RNG instead. Unseeded (default) behavior is unchanged --
`generate_id()` still produces a random-looking UUID4-shaped string
using system randomness if `seed_ids()` is never called, so any code
that doesn't opt in behaves exactly as it always has. Call `seed_ids(n)`
once at the start of a run you want reproducible (NormalWorld.__init__,
or the top of generate_attack_corpus()/generate_mule_network_corpus())
and every ID generated afterward in that process becomes a deterministic
function of that seed and the exact call order -- the same guarantee the
rest of the simulator already assumed it had.
"""

import random
import uuid

_rng = random.Random()  # unseeded by default -- same behavior as uuid.uuid4()


def seed_ids(seed: int) -> None:
    """Reseed the shared ID generator. Every ID generated after this call,
    in this process, is deterministic given the same seed and the same
    sequence of generate_id() calls. Safe to call multiple times (e.g.
    once per corpus-generation run) -- each call starts a fresh,
    independent stream with no memory of IDs generated before it."""
    global _rng
    _rng = random.Random(seed)


def generate_id() -> str:
    """Generate a UUID4-shaped string from the shared RNG."""
    return str(uuid.UUID(int=_rng.getrandbits(128), version=4))
