"""Network topology graph + fault root-cause / propagation modeling.

The production system stores topology in Neo4j; for a self-contained repo we
build the same graph in networkx and expose a `to_cypher()` helper so the exact
graph can be loaded into Neo4j. Root-cause localization scores each node by how
well its own anomaly state explains anomalies observed across its dependents.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def build_topology(n_sites: int, seed: int = 7) -> nx.DiGraph:
    """Build a directed dependency graph: core -> aggregation -> access sites.

    Edges point upstream->downstream so a fault at a parent can propagate to
    children (dependents).
    """
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    n_core = max(2, n_sites // 100)
    n_agg = max(4, n_sites // 20)

    core = [f"core-{i}" for i in range(n_core)]
    agg = [f"agg-{i}" for i in range(n_agg)]
    access = [i for i in range(n_sites)]  # access nodes keyed by site_id

    for c in core:
        g.add_node(c, tier="core")
    for a in agg:
        parent = core[rng.integers(0, n_core)]
        g.add_node(a, tier="agg")
        g.add_edge(parent, a)
    for s in access:
        parent = agg[rng.integers(0, n_agg)]
        g.add_node(s, tier="access", site_id=s)
        g.add_edge(parent, s)
    return g


def localize_root_cause(g: nx.DiGraph, anomaly_by_site: dict[int, bool]) -> list[tuple[str, float]]:
    """Score each non-leaf node by the fraction of its subtree showing anomalies.

    A high score on an upstream node suggests it is the common root cause of the
    downstream anomalies (one fault explaining many dependents).
    """
    scores = {}
    for node in g.nodes:
        descendants = nx.descendants(g, node)
        leaf_sites = [d for d in descendants if g.nodes[d].get("tier") == "access"]
        if len(leaf_sites) < 2:
            continue
        anom = sum(1 for s in leaf_sites if anomaly_by_site.get(g.nodes[s]["site_id"], False))
        scores[node] = anom / len(leaf_sites)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(n, round(s, 4)) for n, s in ranked if s > 0]


def to_cypher(g: nx.DiGraph) -> str:
    """Emit Cypher to recreate the topology in Neo4j."""
    lines = []
    for n, d in g.nodes(data=True):
        tier = d.get("tier", "node")
        lines.append(f"MERGE (n:{tier.capitalize()} {{id: '{n}'}});")
    for u, v in g.edges:
        lines.append(
            f"MATCH (a {{id:'{u}'}}),(b {{id:'{v}'}}) MERGE (a)-[:FEEDS]->(b);"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    g = build_topology(50)
    print(f"nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    fake = {i: (i % 7 == 0) for i in range(50)}
    print("root-cause ranking:", localize_root_cause(g, fake)[:5])
