"""Persistent super-node identity and event logging across snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from .data import Snapshot
from .coarsen import Cluster, Level


@dataclass(frozen=True)
class ClusterEvent:
    snapshot: int
    kind: str
    cluster_id: str
    related_ids: tuple[str, ...]
    overlap: float


class RecurrentNodeMemory:
    """Deterministic GRU memory used by the temporal coarsening baseline.

    Input features contain hashed semantic tokens, node type, and native
    hyperedge degree. A learned implementation can train these weights; fixed
    seeds make this baseline fully reproducible before training.
    """
    def __init__(self, input_size: int = 20, hidden_size: int = 16, seed: int = 7) -> None:
        self.input_size, self.hidden_size = input_size, hidden_size
        rng = np.random.default_rng(seed)
        self.wz = rng.normal(0, .1, (input_size + hidden_size, hidden_size))
        self.wr = rng.normal(0, .1, (input_size + hidden_size, hidden_size))
        self.wh = rng.normal(0, .1, (input_size + hidden_size, hidden_size))
        self.states: dict[str, np.ndarray] = {}

    def _features(self, snapshot: Snapshot, node_id: str, degree: int) -> np.ndarray:
        node = snapshot.nodes[node_id]; value = np.zeros(self.input_size)
        for token in (node.type + " " + node.surface_form).lower().split():
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % (self.input_size - 1)
            value[bucket] += 1.0
        value[-1] = degree / max(1, len(snapshot.hyperedges))
        return value / max(1.0, np.linalg.norm(value))

    def update(self, snapshot: Snapshot) -> dict[str, np.ndarray]:
        output = {}
        degrees = {node_id: 0 for node_id in snapshot.nodes}
        for edge in snapshot.hyperedges:
            for node_id in edge.members:
                degrees[node_id] += 1
        for node_id in snapshot.nodes:
            x = self._features(snapshot, node_id, degrees[node_id]); old = self.states.get(node_id, np.zeros(self.hidden_size))
            joined = np.concatenate([x, old]); z = 1 / (1 + np.exp(-(joined @ self.wz))); r = 1 / (1 + np.exp(-(joined @ self.wr)))
            proposal = np.tanh(np.concatenate([x, r * old]) @ self.wh)
            output[node_id] = (1 - z) * old + z * proposal
        self.states = output
        return output


def _jaccard(a: Cluster, b: Cluster) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def match_levels(snapshot: int, previous: dict[str, Cluster], current: Level,
                 threshold: float = 0.35) -> tuple[dict[str, Cluster], list[ClusterEvent]]:
    assigned: dict[str, Cluster] = {}
    events: list[ClusterEvent] = []
    unused = set(previous)
    # Only clusters sharing a node can have positive Jaccard overlap. Indexing
    # by member avoids a dense previous-by-current comparison.
    prior_owner = {node: old_id for old_id, cluster in previous.items() for node in cluster}
    for index, cluster in enumerate(current.clusters):
        overlap_counts: dict[str, int] = {}
        for node in cluster:
            old_id = prior_owner.get(node)
            if old_id is not None:
                overlap_counts[old_id] = overlap_counts.get(old_id, 0) + 1
        matches = sorted(
            ((old_id, count / len(previous[old_id] | cluster)) for old_id, count in overlap_counts.items()),
            key=lambda x: x[1], reverse=True,
        )
        viable = [m for m in matches if m[1] >= threshold]
        if viable:
            chosen, overlap = viable[0]
            if chosen in unused:
                cluster_id = chosen
                unused.remove(chosen)
                kind = "growth" if len(cluster) >= len(previous[chosen]) else "shrinkage"
                related = tuple(x[0] for x in viable[1:])
            else:
                cluster_id, overlap, kind, related = f"S{snapshot}_{index}", viable[0][1], "split", (chosen,)
        else:
            cluster_id, overlap, kind, related = f"S{snapshot}_{index}", 0.0, "birth", ()
        assigned[cluster_id] = cluster
        events.append(ClusterEvent(snapshot, kind, cluster_id, related, overlap))
    for old_id in unused:
        events.append(ClusterEvent(snapshot, "death", old_id, (), 0.0))
    return assigned, events
