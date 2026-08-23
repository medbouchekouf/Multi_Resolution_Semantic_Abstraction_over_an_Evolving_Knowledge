"""Loading and leakage-safe snapshot construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    surface_form: str
    first_seen_year: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Hyperedge:
    id: str
    relation_type: str
    members: tuple[str, ...]
    year: int | None
    raw: dict[str, Any]

    @property
    def arity(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class TemporalHypergraph:
    meta: dict[str, Any]
    nodes: dict[str, Node]
    hyperedges: tuple[Hyperedge, ...]

    def snapshot(self, cutoff: int) -> "Snapshot":
        available = {
            node_id for node_id, node in self.nodes.items()
            if node.first_seen_year is not None and node.first_seen_year <= cutoff
        }
        edges = tuple(
            edge for edge in self.hyperedges
            if edge.year is not None and edge.year <= cutoff and set(edge.members).issubset(available)
        )
        return Snapshot(cutoff, {node_id: self.nodes[node_id] for node_id in available}, edges)


@dataclass(frozen=True)
class Snapshot:
    cutoff: int
    nodes: dict[str, Node]
    hyperedges: tuple[Hyperedge, ...]

    def describe(self) -> dict[str, Any]:
        def counts(values: list[str | int]) -> dict[str, int]:
            result: dict[str, int] = {}
            for value in values:
                result[str(value)] = result.get(str(value), 0) + 1
            return dict(sorted(result.items()))
        years = [e.year for e in self.hyperedges if e.year is not None]
        pairwise_edges = sum(e.arity * (e.arity - 1) // 2 for e in self.hyperedges)
        return {
            "cutoff": self.cutoff,
            "nodes": len(self.nodes),
            "hyperedges": len(self.hyperedges),
            "node_types": counts([n.type for n in self.nodes.values()]),
            "arity_distribution": counts([e.arity for e in self.hyperedges]),
            "edge_time_span": [min(years), max(years)] if years else None,
            "projection_loss": {
                "native_hyperedges": len(self.hyperedges),
                "clique_expansion_pairwise_edges": pairwise_edges,
                "mean_pairwise_edges_per_hyperedge": pairwise_edges / max(1, len(self.hyperedges)),
                "lost_fields": ["relation-level multi-way identity", "endpoint multiplicity", "hyperedge arity"],
            },
        }


def load_hypergraph(path: str | Path) -> TemporalHypergraph:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {
        item["id"]: Node(
            id=item["id"], type=item["type"], surface_form=item.get("surface_form", ""),
            first_seen_year=item.get("first_seen_year"), raw=item,
        ) for item in raw["nodes"]
    }
    edges = tuple(
        Hyperedge(
            id=item["id"], relation_type=item["relation_type"],
            members=tuple(item["members"]), year=item.get("year"), raw=item,
        ) for item in raw["hyperedges"]
    )
    return TemporalHypergraph(raw.get("meta", {}), nodes, edges)

