"""Dependency-free DOT exports for hierarchy and coarse-hypergraph diagrams."""
from __future__ import annotations

from pathlib import Path
from .coarsen import Level


def write_level_dot(level: Level, path: str | Path, title: str = "Coarse hypergraph") -> None:
    """Export hyperedges as diamond relation nodes, preserving their multi-way form."""
    lines = ["graph hierarchy {", f'  label="{title}";', "  labelloc=t;", "  node [fontname=Helvetica];"]
    for index, cluster in enumerate(level.clusters):
        label = f"S{index}\\n{len(cluster)} members"
        lines.append(f'  s{index} [shape=ellipse,label="{label}"];')
    for index, edge in enumerate(level.coarse_edges):
        label = f"{edge.relation_type}\\narity={edge.original_arity}, count={edge.count}"
        lines.append(f'  e{index} [shape=diamond,label="{label}"];')
        for endpoint, multiplicity in zip(edge.endpoints, edge.multiplicities):
            lines.append(f'  e{index} -- s{endpoint} [label="{multiplicity}"];')
    lines.append("}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
