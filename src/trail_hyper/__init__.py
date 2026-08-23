"""TRAIL-Hyper: native temporal-hypergraph abstraction."""

from .data import TemporalHypergraph, load_hypergraph
from .coarsen import build_hierarchy

__all__ = ["TemporalHypergraph", "load_hypergraph", "build_hierarchy"]

