"""
v2/serving/store.py -- history-only entity graph used for graph features, in TRAINING and in SERVING.

For a trace being scored, features come ONLY from traces ingested before it (no future information, unlike a batch
graph built over the whole dataset). Training replays the dataset in time order through this same class, so the
distribution of graph features seen in training matches what the API computes online (no train/serve skew).

Entity types: device, payee (beneficiary), merchant, address, phone.
  g_device_shared_customers   = max over this trace's devices of the number of OTHER customers seen with that device
  g_payee_shared_customers    = same for payees;  g_address_...  / g_phone_... likewise
  g_merchant_payers           = max over merchants of distinct customers who paid it (including this one)
  g_component_size            = size of the customer component this trace would join, linking via device/payee/address/phone
                                (merchants are hubs and are excluded from connectivity on purpose)
"""
from __future__ import annotations

import json
from collections import defaultdict

TYPES = ("device", "payee", "merchant", "address", "phone")
LINKING = ("device", "payee", "address", "phone")


def entities_of(rec: dict) -> dict:
    ents = {t: set() for t in TYPES}
    for e in rec["events"]:
        et = e["event_type"]
        if et in ("SESSION_LOGIN", "DEVICE_REGISTRATION") and e.get("device_id"):
            ents["device"].add(e["device_id"])
        if et in ("BENEFICIARY_ADDITION", "TRANSACTION") and e.get("beneficiary_id"):
            ents["payee"].add(e["beneficiary_id"])
        if et == "TRANSACTION" and e.get("merchant_id"):
            ents["merchant"].add(e["merchant_id"])
        if et == "ACCOUNT_OPENING":
            if e.get("address_hash"):
                ents["address"].add(e["address_hash"])
            if e.get("phone_hash"):
                ents["phone"].add(e["phone_hash"])
    return ents


class EntityStore:
    def __init__(self):
        self.owners = {t: defaultdict(set) for t in TYPES}
        self.parent: dict = {}
        self.size: dict = {}

    # -- union-find
    def _find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            if (self.size[ra], ra) < (self.size[rb], rb):     # deterministic root choice (size, then name)
                ra, rb = rb, ra
            self.parent[rb] = ra
            self.size[ra] += self.size[rb]

    # -- read-only feature computation (history only)
    def features(self, rec: dict) -> dict:
        cust = rec["customer_id"]
        ents = entities_of(rec)

        def others(t):
            vals = [len(self.owners[t][x] - {cust}) for x in ents[t] if x in self.owners[t]]
            return max(vals) if vals else 0

        payers = [len(self.owners["merchant"].get(x, set()) | {cust}) for x in ents["merchant"]]
        linked = {cust}
        for t in LINKING:
            for x in ents[t]:
                linked |= self.owners[t].get(x, set())
        roots = {self._find(c) for c in linked if c in self.parent}
        comp = sum(self.size[r] for r in roots) + (0 if cust in self.parent else 1)
        return {"g_device_shared_customers": others("device"), "g_payee_shared_customers": others("payee"),
                "g_merchant_payers": max(payers) if payers else 0, "g_address_shared_customers": others("address"),
                "g_phone_shared_customers": others("phone"), "g_component_size": comp}

    # -- mutation
    def ingest(self, rec: dict) -> None:
        cust = rec["customer_id"]
        if cust not in self.parent:
            self.parent[cust], self.size[cust] = cust, 1
        ents = entities_of(rec)
        for t in TYPES:
            for x in ents[t]:
                if t in LINKING:
                    for other in sorted(self.owners[t][x]):
                        self._union(cust, other)
                self.owners[t][x].add(cust)

    # -- persistence
    def to_json(self) -> str:
        return json.dumps({"owners": {t: {k: sorted(v) for k, v in d.items()} for t, d in self.owners.items()},
                           "parent": dict(sorted(self.parent.items())),
                           "size": dict(sorted(self.size.items()))}, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "EntityStore":
        d, st = json.loads(s), cls()
        for t in TYPES:
            for k, v in d["owners"][t].items():
                st.owners[t][k] = set(v)
        st.parent, st.size = d["parent"], d["size"]
        return st

    def stats(self) -> dict:
        return {"customers": len(self.parent), **{f"{t}_entities": len(self.owners[t]) for t in TYPES}}


def replay_features(records: list[dict]) -> list[dict]:
    """Time-ordered replay: features for each record from history only, then ingest it. Returns rows in INPUT order."""
    order = sorted(range(len(records)), key=lambda i: (records[i]["events"][0]["timestamp"] if records[i]["events"] else "", records[i]["trace_id"]))
    store, out = EntityStore(), [None] * len(records)
    for i in order:
        out[i] = store.features(records[i])
        store.ingest(records[i])
    return out, store
