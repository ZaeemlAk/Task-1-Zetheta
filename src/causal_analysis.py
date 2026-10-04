"""Causal-analysis scaffolding for the audit."""

from __future__ import annotations

from typing import Any

import networkx as nx


def build_credit_dag() -> nx.DiGraph:
    """Create a directed acyclic graph showing the audit workflow."""
    graph = nx.DiGraph()
    nodes = [
        "Protected attributes",
        "Proxy variables",
        "Credit features",
        "Model score",
        "Credit decision",
        "Default",
    ]
    graph.add_nodes_from(nodes)
    edges = [
        ("Protected attributes", "Proxy variables"),
        ("Proxy variables", "Credit features"),
        ("Protected attributes", "Credit features"),
        ("Credit features", "Model score"),
        ("Model score", "Credit decision"),
        ("Credit decision", "Default"),
    ]
    graph.add_edges_from(edges)
    return graph


def describe_causal_paths() -> list[str]:
    """Return a short list of plausible causal pathways to investigate with data."""
    return [
        "Gender -> dependents -> model score",
        "Gender -> marital status -> model score",
        "Geography -> city tier -> model score",
        "Geography -> pincode -> model score",
        "Age -> relationship duration -> model score",
        "Income -> mobile OS -> model score",
    ]
