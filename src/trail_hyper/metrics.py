"""Compact metrics for the required evaluation axes."""
from __future__ import annotations

from .coarsen import Cluster
from .data import Snapshot


def hyperedge_retention(snapshot: Snapshot, clusters: list[Cluster]) -> float:
    parent = {node: i for i, c in enumerate(clusters) for node in c}
    if not snapshot.hyperedges:
        return 0.0
    return sum(max(list(parent[n] for n in edge.members).count(p) for p in set(parent[n] for n in edge.members)) / edge.arity
               for edge in snapshot.hyperedges) / len(snapshot.hyperedges)


def temporal_membership_agreement(previous: dict[str, Cluster], current: dict[str, Cluster]) -> float:
    shared = set().union(*previous.values()) & set().union(*current.values()) if previous and current else set()
    if not shared:
        return 0.0
    prev_owner = {node: ident for ident, group in previous.items() for node in group}
    curr_owner = {node: ident for ident, group in current.items() for node in group}
    return sum(prev_owner[node] == curr_owner[node] for node in shared) / len(shared)


def label_support_rate(label_terms: set[str], evidence_terms: set[str]) -> float:
    return len(label_terms & evidence_terms) / len(label_terms) if label_terms else 1.0

