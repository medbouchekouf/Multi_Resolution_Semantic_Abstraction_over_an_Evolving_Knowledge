from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_hypergraph
from .coarsen import build_hierarchy
from .temporal import match_levels
from .temporal import RecurrentNodeMemory
from .experiments import run_baselines
from .neo4j_store import Neo4jStore
from .visualize import write_level_dot
from .rl import LinearMergePolicy


def _write(path: str, value: dict) -> None:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRAIL-Hyper experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("describe", "run", "experiment"):
        p = sub.add_parser(name)
        p.add_argument("--hypergraph", required=True)
        p.add_argument("--cutoffs", required=True, nargs="+", type=int)
        p.add_argument("--output", required=True)
        if name in ("run", "experiment"):
            p.add_argument("--levels", type=int, default=2)
        if name == "run": p.add_argument("--rl", action="store_true", help="Use REINFORCE bounded merge policy")
    seed = sub.add_parser("neo4j-seed", help="Load native incidence model into Neo4j")
    seed.add_argument("--hypergraph", required=True)
    seed.add_argument("--uri", default="bolt://localhost:7687")
    seed.add_argument("--username", default="neo4j")
    seed.add_argument("--password", required=True)
    args = parser.parse_args()
    graph = load_hypergraph(args.hypergraph)
    if args.command == "neo4j-seed":
        store = Neo4jStore(args.uri, args.username, args.password)
        try:
            store.seed(graph)
        finally:
            store.close()
        return
    snapshots = [graph.snapshot(cutoff) for cutoff in args.cutoffs]
    if args.command == "describe":
        report = [s.describe() for s in snapshots]
        for prior, now in zip(report, report[1:]):
            now["growth_since_previous"] = {"nodes": now["nodes"] - prior["nodes"], "hyperedges": now["hyperedges"] - prior["hyperedges"]}
        _write(args.output, {"snapshots": report})
        return
    if args.command == "experiment":
        _write(args.output, {"cutoffs": args.cutoffs, "baselines": run_baselines(graph, args.cutoffs, args.levels), "metrics": ["hyperedge_retention", "temporal_membership_agreement", "label_support_rate", "Recall@k", "MRR"]})
        return
    previous = {}; all_events = []; output = []; memory = RecurrentNodeMemory(); policy = LinearMergePolicy(seed=11) if args.rl else None
    for snapshot in snapshots:
        states = memory.update(snapshot)
        hierarchy = build_hierarchy(snapshot, levels=args.levels, recurrent_states=states, policy=policy)
        if policy is not None:
            # The immediate native-hyperedge retention is the reproducible
            # episode reward for this lightweight REINFORCE starter.
            from .metrics import hyperedge_retention
            policy.reinforce(hyperedge_retention(snapshot, hierarchy[-1].clusters))
        current, events = match_levels(snapshot.cutoff, previous, hierarchy[-1])
        Path("artifacts").mkdir(exist_ok=True)
        write_level_dot(hierarchy[-1], Path("artifacts") / f"coarse_{snapshot.cutoff}.dot", f"TRAIL-Hyper {snapshot.cutoff}")
        output.append({"cutoff": snapshot.cutoff, "levels": [{"clusters": len(level.clusters), "coarse_edges": len(level.coarse_edges), "internal_edges": level.internal_edges} for level in hierarchy]})
        previous = current; all_events.extend(event.__dict__ for event in events)
    _write(args.output, {"runs": output, "events": all_events})


if __name__ == "__main__":
    main()
