from trail_hyper.data import Node, Hyperedge, Snapshot
from trail_hyper.coarsen import collapse_hyperedges, build_hierarchy
from trail_hyper.temporal import RecurrentNodeMemory


def sample() -> Snapshot:
    nodes = {name: Node(name, "method", name, 2024, {}) for name in "abcd"}
    edges = (Hyperedge("e1", "evaluated_on", ("a", "b", "c"), 2024, {}), Hyperedge("e2", "claims", ("c", "d"), 2024, {}))
    return Snapshot(2024, nodes, edges)


def test_collapse_preserves_three_way_edge_and_multiplicity():
    coarse, internal = collapse_hyperedges(sample(), {"a": 0, "b": 0, "c": 1, "d": 2})
    target = next(edge for edge in coarse if edge.relation_type == "evaluated_on")
    assert target.endpoints == (0, 1)
    assert target.multiplicities == (2, 1)
    assert target.original_arity == 3
    assert internal == 0


def test_hierarchy_is_laminar():
    hierarchy = build_hierarchy(sample(), levels=2, shrink=0.5)
    for earlier, later in zip(hierarchy, hierarchy[1:]):
        assert all(any(child <= parent for parent in later.clusters) for child in earlier.clusters)


def test_recurrent_memory_is_deterministic_and_complete():
    first = RecurrentNodeMemory(seed=4).update(sample())
    second = RecurrentNodeMemory(seed=4).update(sample())
    assert set(first) == set(sample().nodes)
    assert all((first[node] == second[node]).all() for node in first)
