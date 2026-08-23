"""Neo4j storage that represents every hyperedge as a first-class node.

(:Entity)<-[:HAS_MEMBER]-(h:Hyperedge) is native incidence, unlike clique
projection. The same model supports original and coarse hyperedges.
"""
from __future__ import annotations

from typing import Iterable
from .data import TemporalHypergraph


SCHEMA = (
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT hyperedge_id IF NOT EXISTS FOR (h:Hyperedge) REQUIRE h.id IS UNIQUE",
)


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def seed(self, graph: TemporalHypergraph) -> None:
        with self.driver.session() as session:
            for query in SCHEMA:
                session.run(query).consume()
            session.run("UNWIND $rows AS row MERGE (n:Entity {id:row.id}) SET n += row", rows=[
                {"id": n.id, "type": n.type, "surface_form": n.surface_form, "first_seen_year": n.first_seen_year}
                for n in graph.nodes.values()
            ]).consume()
            session.run("UNWIND $rows AS row MERGE (h:Hyperedge {id:row.id}) SET h += row", rows=[
                {"id": e.id, "relation_type": e.relation_type, "year": e.year, "arity": e.arity}
                for e in graph.hyperedges
            ]).consume()
            rows = []
            for edge in graph.hyperedges:
                rows.extend({"edge_id": edge.id, "node_id": node, "position": pos} for pos, node in enumerate(edge.members))
            session.run("UNWIND $rows AS row MATCH (h:Hyperedge {id:row.edge_id}) MATCH (n:Entity {id:row.node_id}) MERGE (h)-[r:HAS_MEMBER {position:row.position}]->(n)", rows=rows).consume()

    @staticmethod
    def snapshot_query(cutoff: int) -> tuple[str, dict]:
        return ("MATCH (h:Hyperedge)-[:HAS_MEMBER]->(n:Entity) WHERE h.year <= $cutoff AND n.first_seen_year <= $cutoff RETURN h,n", {"cutoff": cutoff})
