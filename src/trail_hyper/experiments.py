"""Registered baselines and key metrics for reproducible experiments."""
from __future__ import annotations

from .data import TemporalHypergraph
from .coarsen import build_hierarchy
from .metrics import hyperedge_retention, temporal_membership_agreement
from .temporal import RecurrentNodeMemory, match_levels


def run_baselines(graph: TemporalHypergraph, cutoffs: list[int], levels: int = 2) -> dict:
    """Run fixed structural/semantic mixes plus the recurrent full model.

    This is a registered protocol, not evidence that RL has been trained.
    """
    configs = {"structural_only": 1.0, "semantic_structural": 0.65, "semantic_emphasis": 0.2}
    report = {}
    for name, alpha in configs.items():
        previous = {}; snapshots = []
        memory = RecurrentNodeMemory() if name == "semantic_structural" else None
        for cutoff in cutoffs:
            snapshot = graph.snapshot(cutoff)
            states = memory.update(snapshot) if memory else None
            hierarchy = build_hierarchy(snapshot, levels=levels, alpha=alpha, recurrent_states=states)
            current, events = match_levels(cutoff, previous, hierarchy[-1])
            snapshots.append({"cutoff": cutoff, "hyperedge_retention": hyperedge_retention(snapshot, hierarchy[-1].clusters), "clusters": len(hierarchy[-1].clusters), "stability": temporal_membership_agreement(previous, current), "events": len(events)})
            previous = current
        report[name] = snapshots
    return report
