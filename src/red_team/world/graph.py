"""NetworkX relationship graph for the Normal World simulation.

This module provides a MultiDiGraph to maintain a live representation of entities
and their relationships within the WorldState.
"""

from typing import Any, Dict, List, Optional
import networkx as nx

from red_team.schemas.entities import Relationship


class RelationshipGraph:
    """Live relationship graph mirroring the WorldState.
    
    Uses nx.MultiDiGraph because multiple relationships can exist between
    the same two entities (e.g. 'owns' and 'primary_contact').
    """

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        # Lookup mapping for relationships: relationship_id -> (u, v, key)
        self._edge_index: Dict[str, tuple[str, str, str]] = {}

    def _node_id(self, entity_type: str, entity_id: str) -> str:
        """Format a heterogeneous node ID (e.g., 'customer:uuid')."""
        return f"{entity_type.lower()}:{entity_id}"

    def add_entity(self, entity_type: str, entity_id: str, **metadata: Any) -> None:
        """Add an entity as a node to the graph."""
        nid = self._node_id(entity_type, entity_id)
        if not self._graph.has_node(nid):
            self._graph.add_node(nid, entity_type=entity_type.lower(), **metadata)
        else:
            self._graph.nodes[nid].update(metadata)

    def remove_entity(self, entity_type: str, entity_id: str) -> None:
        """Remove an entity and all its incident edges."""
        nid = self._node_id(entity_type, entity_id)
        if self._graph.has_node(nid):
            # Also remove edges from index
            edges_to_remove = []
            for u, v, key in self._graph.edges(nid, keys=True):
                edges_to_remove.append(key)
            for key in set(edges_to_remove):
                if key in self._edge_index:
                    del self._edge_index[key]
            self._graph.remove_node(nid)

    def has_entity(self, entity_type: str, entity_id: str) -> bool:
        """Check if an entity node exists."""
        return self._graph.has_node(self._node_id(entity_type, entity_id))

    def add_relationship(self, rel: Relationship) -> None:
        """Add a Relationship as an edge to the graph."""
        if rel.relationship_id in self._edge_index:
            return self.update_relationship(rel)

        u = self._node_id(rel.source_entity_type, rel.source_entity_id)
        v = self._node_id(rel.target_entity_type, rel.target_entity_id)
        
        # Ensure nodes exist
        self.add_entity(rel.source_entity_type, rel.source_entity_id)
        self.add_entity(rel.target_entity_type, rel.target_entity_id)

        key = rel.relationship_id
        
        self._graph.add_edge(
            u, v, key=key,
            relationship_type=rel.relationship_type,
            established_date=rel.established_date,
            last_activity_date=rel.last_activity_date,
            is_active=rel.is_active,
            strength_score=getattr(rel, "strength_score", 0.0),
        )
        self._edge_index[key] = (u, v, key)

    def update_relationship(self, rel: Relationship) -> None:
        """Update metadata on an existing relationship edge."""
        if rel.relationship_id not in self._edge_index:
            return self.add_relationship(rel)
            
        u, v, key = self._edge_index[rel.relationship_id]
        
        # We only update attributes; we don't change endpoints.
        # If endpoints changed, it's technically a new relationship id in Stage 2.
        edge_data = self._graph[u][v][key]
        edge_data["relationship_type"] = rel.relationship_type
        edge_data["last_activity_date"] = rel.last_activity_date
        edge_data["is_active"] = rel.is_active
        edge_data["strength_score"] = getattr(rel, "strength_score", 0.0)

    def remove_relationship(self, relationship_id: str) -> None:
        """Remove a relationship edge by ID."""
        if relationship_id in self._edge_index:
            u, v, key = self._edge_index[relationship_id]
            self._graph.remove_edge(u, v, key=key)
            del self._edge_index[relationship_id]

    def has_relationship(self, relationship_id: str) -> bool:
        """Check if a relationship edge exists."""
        return relationship_id in self._edge_index

    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve relationship metadata from the graph."""
        if relationship_id in self._edge_index:
            u, v, key = self._edge_index[relationship_id]
            return self._graph[u][v][key]
        return None

    def get_neighbors(self, entity_type: str, entity_id: str) -> List[str]:
        """Get the node IDs of all adjacent neighbors (ignoring edge direction here)."""
        nid = self._node_id(entity_type, entity_id)
        if self._graph.has_node(nid):
            succ = list(self._graph.successors(nid))
            pred = list(self._graph.predecessors(nid))
            return list(set(succ + pred))
        return []
        
    def snapshot(self) -> nx.MultiDiGraph:
        """Return a copy of the graph, without exposing internal references."""
        return self._graph.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return basic graph statistics."""
        stats = {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_relationship_type": {},
            "degree_distribution": {},
            "in_degree_distribution": {},
            "out_degree_distribution": {},
        }
        
        # Node stats
        for n, data in self._graph.nodes(data=True):
            ntype = data.get("entity_type", "unknown")
            stats["nodes_by_type"][ntype] = stats["nodes_by_type"].get(ntype, 0) + 1
            
            deg = self._graph.degree(n)
            in_deg = self._graph.in_degree(n)
            out_deg = self._graph.out_degree(n)
            
            stats["degree_distribution"][deg] = stats["degree_distribution"].get(deg, 0) + 1
            stats["in_degree_distribution"][in_deg] = stats["in_degree_distribution"].get(in_deg, 0) + 1
            stats["out_degree_distribution"][out_deg] = stats["out_degree_distribution"].get(out_deg, 0) + 1
            
        # Edge stats
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            rtype = data.get("relationship_type", "unknown")
            stats["edges_by_relationship_type"][rtype] = stats["edges_by_relationship_type"].get(rtype, 0) + 1
            
        # Domain specific metrics
        customers = [n for n, data in self._graph.nodes(data=True) if data.get("entity_type") == "customer"]
        devices = [n for n, data in self._graph.nodes(data=True) if data.get("entity_type") == "device"]
        beneficiaries = [n for n, data in self._graph.nodes(data=True) if data.get("entity_type") == "beneficiary"]
        
        dev_per_cust = []
        ben_per_cust = []
        for c in customers:
            # Succ + Pred where neighbor is device
            neighbors = self.get_neighbors("customer", c.split(":")[1])
            c_devices = sum(1 for n in neighbors if n.startswith("device:"))
            c_bens = sum(1 for n in neighbors if n.startswith("beneficiary:"))
            dev_per_cust.append(c_devices)
            ben_per_cust.append(c_bens)
            
        stats["devices_per_customer"] = sum(dev_per_cust) / len(dev_per_cust) if dev_per_cust else 0.0
        stats["beneficiaries_per_customer"] = sum(ben_per_cust) / len(ben_per_cust) if ben_per_cust else 0.0
        
        cust_per_dev = []
        for d in devices:
            neighbors = self.get_neighbors("device", d.split(":")[1])
            d_custs = sum(1 for n in neighbors if n.startswith("customer:"))
            cust_per_dev.append(d_custs)
            
        stats["customers_per_device"] = sum(cust_per_dev) / len(cust_per_dev) if cust_per_dev else 0.0
        
        cust_per_ben = []
        for b in beneficiaries:
            neighbors = self.get_neighbors("beneficiary", b.split(":")[1])
            b_custs = sum(1 for n in neighbors if n.startswith("customer:"))
            cust_per_ben.append(b_custs)
            
        stats["customers_per_beneficiary"] = sum(cust_per_ben) / len(cust_per_ben) if cust_per_ben else 0.0
        
        return stats


def validate_consistency(world_state: Any) -> Dict[str, Any]:
    """Check consistency between WorldState and its RelationshipGraph."""
    graph = world_state.graph
    
    missing_nodes = []
    missing_edges = []
    
    # Check nodes
    for c_id in world_state.customers:
        if not graph.has_entity("customer", c_id):
            missing_nodes.append(f"customer:{c_id}")
    for a_id in world_state.accounts:
        if not graph.has_entity("account", a_id):
            missing_nodes.append(f"account:{a_id}")
    for d_id in world_state.devices:
        if not graph.has_entity("device", d_id):
            missing_nodes.append(f"device:{d_id}")
    for m_id in world_state.merchants:
        if not graph.has_entity("merchant", m_id):
            missing_nodes.append(f"merchant:{m_id}")
    for b_id in world_state.beneficiaries:
        if not graph.has_entity("beneficiary", b_id):
            missing_nodes.append(f"beneficiary:{b_id}")
            
    # Check edges
    for r_id, rel in world_state.relationships.items():
        if not graph.has_relationship(r_id):
            missing_edges.append(r_id)
        else:
            edge = graph.get_relationship(r_id)
            if edge["is_active"] != rel.is_active:
                missing_edges.append(f"{r_id}_inactive_mismatch")
                
    is_consistent = (len(missing_nodes) == 0 and len(missing_edges) == 0)
    return {
        "is_consistent": is_consistent,
        "missing_nodes": missing_nodes,
        "missing_edges": missing_edges,
    }
