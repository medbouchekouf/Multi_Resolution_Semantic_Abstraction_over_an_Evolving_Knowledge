"""Laminar, native-hypergraph coarsening with an RL-compatible action interface."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import combinations
from math import log
from typing import Callable
import numpy as np

from .data import Snapshot


Cluster = frozenset[str]


@dataclass(frozen=True)
class CoarseEdge:
    relation_type: str
    endpoints: tuple[int, ...]
    multiplicities: tuple[int, ...]
    original_arity: int
    count: int = 1


@dataclass
class Level:
    clusters: list[Cluster]
    parent_of: dict[str, int]
    coarse_edges: list[CoarseEdge]
    internal_edges: int


def _tokens(text: str) -> set[str]:
    return {word.strip(".,;:()[]{}-/").lower() for word in text.split() if len(word) > 2}


def _semantic_similarity(a: Cluster, b: Cluster, snapshot: Snapshot, recurrent_states: dict[str, np.ndarray] | None = None) -> float:
    # Bounded evidence summaries prevent a giant prior cluster from making a
    # later coarsening level quadratic in its member text.
    ta = set().union(*(_tokens(snapshot.nodes[x].surface_form) for x in sorted(a)[:24]))
    tb = set().union(*(_tokens(snapshot.nodes[x].surface_form) for x in sorted(b)[:24]))
    lexical = len(ta & tb) / len(ta | tb) if ta or tb else 0.0
    if not recurrent_states:
        return lexical
    va = np.mean([recurrent_states[n] for n in a], axis=0); vb = np.mean([recurrent_states[n] for n in b], axis=0)
    recurrent = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    return 0.5 * lexical + 0.5 * max(0.0, recurrent)


def _structural_similarity(a: Cluster, b: Cluster, snapshot: Snapshot) -> float:
    score = 0.0
    for edge in snapshot.hyperedges:
        members = set(edge.members)
        if members & a and members & b:
            score += 1.0 / max(1, edge.arity - 1)
    return score / (1.0 + log(1.0 + len(a) + len(b)))


def candidate_merges(clusters: list[Cluster], snapshot: Snapshot, top_k: int = 8, recurrent_states: dict[str, np.ndarray] | None = None) -> list[tuple[int, int, float, float]]:
    """Generate a bounded action set rather than all cluster pairs."""
    # Generate candidates from native incidence, not from every possible pair
    # of clusters. A pair enters the action set only when a hyperedge supports it.
    owner = {node: index for index, cluster in enumerate(clusters) for node in cluster}
    structural: defaultdict[tuple[int, int], float] = defaultdict(float)
    for edge in snapshot.hyperedges:
        endpoints = sorted({owner[node] for node in edge.members})
        for i, j in combinations(endpoints, 2):
            structural[(i, j)] += 1.0 / max(1, edge.arity - 1)
    candidates = [
        (i, j, value, _semantic_similarity(clusters[i], clusters[j], snapshot, recurrent_states))
        for (i, j), value in structural.items()
    ]
    return sorted(candidates, key=lambda x: x[2] + x[3], reverse=True)[:top_k * max(1, len(clusters))]


def collapse_hyperedges(snapshot: Snapshot, parent_of: dict[str, int]) -> tuple[list[CoarseEdge], int]:
    """Apply the specified hyperedge-collapse rule without pairwise projection."""
    aggregate: Counter[tuple[str, tuple[int, ...], tuple[int, ...], int]] = Counter()
    internal = 0
    for edge in snapshot.hyperedges:
        mapped = [parent_of[node] for node in edge.members]
        counts = Counter(mapped)
        if len(counts) == 1:
            internal += 1  # fully internal: retained as cluster evidence, not an intercluster edge
            continue
        endpoints = tuple(sorted(counts))
        multiplicities = tuple(counts[node] for node in endpoints)
        aggregate[(edge.relation_type, endpoints, multiplicities, edge.arity)] += 1
    return [CoarseEdge(*key, count=value) for key, value in aggregate.items()], internal


def _choose(candidates: list[tuple[int, int, float, float]], alpha: float,
            policy: Callable[[list[tuple[int, int, float, float]]], int] | None) -> int:
    if policy is not None:
        return policy(candidates)
    return max(range(len(candidates)), key=lambda k: alpha * candidates[k][2] + (1 - alpha) * candidates[k][3])


def coarsen_once(snapshot: Snapshot, clusters: list[Cluster], target_count: int,
                 alpha: float = 0.65, policy: Callable | None = None, recurrent_states: dict[str, np.ndarray] | None = None) -> Level:
    """Merge only existing groups, guaranteeing a laminar hierarchy."""
    # Candidate generation is performed once per level. This keeps the starter
    # scalable on the 5,798-node corpus while still making sequential, bounded
    # merge decisions. A production policy can refresh candidates periodically.
    initial = list(clusters)
    candidates = candidate_merges(initial, snapshot, recurrent_states=recurrent_states)
    parent = list(range(len(initial)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    remaining = len(initial)
    if policy is None:
        # Fast deterministic warm start: a Kruskal-style pass over native
        # hyperedge-supported actions. RL policies use the branch below.
        ranked = sorted(candidates, key=lambda x: alpha * x[2] + (1 - alpha) * x[3], reverse=True)
        for i, j, _, _ in ranked:
            if remaining <= target_count:
                break
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                remaining -= 1
    else:
        # RL controls an early bounded decision budget; the deterministic
        # warm-start completes the level. This avoids an impractical policy
        # action space on thousands of singleton clusters.
        decisions = 0
        while remaining > target_count and decisions < 128:
            valid = [item for item in candidates if find(item[0]) != find(item[1])]
            if not valid:
                break
            i, j, _, _ = valid[_choose(valid, alpha, policy)]
            root_i, root_j = find(i), find(j)
            parent[root_j] = root_i
            remaining -= 1
            decisions += 1
        ranked = sorted(candidates, key=lambda x: alpha * x[2] + (1 - alpha) * x[3], reverse=True)
        for i, j, _, _ in ranked:
            if remaining <= target_count:
                break
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                remaining -= 1
    grouped: defaultdict[int, set[str]] = defaultdict(set)
    for index, cluster in enumerate(initial):
        grouped[find(index)].update(cluster)
    working = [frozenset(group) for group in grouped.values()]
    parent_of = {node: index for index, group in enumerate(working) for node in group}
    coarse_edges, internal = collapse_hyperedges(snapshot, parent_of)
    return Level(working, parent_of, coarse_edges, internal)


def build_hierarchy(snapshot: Snapshot, levels: int = 2, shrink: float = 0.5,
                    alpha: float = 0.65, policy: Callable | None = None, recurrent_states: dict[str, np.ndarray] | None = None) -> list[Level]:
    """Level 0 is singleton membership; subsequent levels are coarsened partitions."""
    current = [frozenset([node]) for node in sorted(snapshot.nodes)]
    hierarchy = [coarsen_once(snapshot, current, len(current), alpha, policy, recurrent_states)]
    for _ in range(levels):
        target = max(1, int(len(current) * shrink))
        level = coarsen_once(snapshot, current, target, alpha, policy, recurrent_states)
        hierarchy.append(level)
        current = level.clusters
        if len(current) == 1:
            break
    return hierarchy


def labeller_input(snapshot: Snapshot, cluster: Cluster, max_members: int = 20) -> dict:
    """Auditable, time-safe input for an external labeller."""
    members = sorted(cluster)[:max_members]
    return {
        "snapshot_cutoff": snapshot.cutoff,
        "instruction": "Return a short label and one factual gloss. Do not add facts absent from the evidence.",
        "node_type_counts": dict(Counter(snapshot.nodes[n].type for n in cluster)),
        "members": [{"id": n, "type": snapshot.nodes[n].type, "surface_form": snapshot.nodes[n].surface_form} for n in members],
    }
