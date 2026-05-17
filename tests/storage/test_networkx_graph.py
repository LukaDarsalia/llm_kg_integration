import pytest

from llm_kg.storage import GRAPH_REGISTRY
from llm_kg.storage.networkx_graph import NetworkXGraphStore


def test_registered_as_networkx() -> None:
    assert "networkx" in GRAPH_REGISTRY
    assert GRAPH_REGISTRY.get("networkx") is NetworkXGraphStore


def test_add_node_with_attrs() -> None:
    g = NetworkXGraphStore()
    g.add_node("alice", type="PERSON", description="a person")
    g.add_node("bob", type="PERSON")
    assert "alice" in g
    assert g.get_node("alice")["type"] == "PERSON"
    assert g.get_node("alice")["description"] == "a person"


def test_get_missing_node_returns_none() -> None:
    g = NetworkXGraphStore()
    assert g.get_node("nobody") is None


def test_add_edge_creates_endpoints_if_missing() -> None:
    g = NetworkXGraphStore()
    g.add_edge("alice", "bob", predicate="knows")
    assert "alice" in g and "bob" in g
    assert g.get_edge("alice", "bob")["predicate"] == "knows"


def test_neighbors_is_undirected() -> None:
    """LightRAG semantics: relations are undirected."""
    g = NetworkXGraphStore()
    g.add_edge("alice", "bob", predicate="knows")
    assert "bob" in g.neighbors("alice")
    assert "alice" in g.neighbors("bob")


def test_edge_lookup_works_both_directions() -> None:
    g = NetworkXGraphStore()
    g.add_edge("alice", "bob", weight=5.0)
    assert g.get_edge("alice", "bob")["weight"] == 5.0
    assert g.get_edge("bob", "alice")["weight"] == 5.0


def test_node_degree() -> None:
    g = NetworkXGraphStore()
    g.add_edge("alice", "bob")
    g.add_edge("alice", "carol")
    assert g.node_degree("alice") == 2
    assert g.node_degree("bob") == 1
    assert g.node_degree("missing") == 0


def test_personalized_pagerank_biases_toward_seeds() -> None:
    g = NetworkXGraphStore()
    # chain: a - b - c - d
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    ppr = g.personalized_pagerank({"a": 1.0})
    # Nodes closer to the seed should have higher PPR than nodes farther away.
    # (In an undirected chain `b` may slightly exceed `a` due to the teleport+edge balance.)
    assert ppr["a"] > ppr["c"] > ppr["d"]
    assert ppr["b"] > ppr["d"]


def test_personalized_pagerank_empty_seeds_uses_uniform() -> None:
    """Empty seeds → standard PageRank (uniform)."""
    g = NetworkXGraphStore()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    ppr = g.personalized_pagerank({})
    assert set(ppr.keys()) == {"a", "b", "c"}
    assert all(v > 0 for v in ppr.values())


def test_personalized_pagerank_unknown_seed_ignored() -> None:
    g = NetworkXGraphStore()
    g.add_edge("a", "b")
    ppr = g.personalized_pagerank({"ghost": 1.0, "a": 1.0})
    assert "a" in ppr
    assert "ghost" not in ppr


def test_re_adding_edge_updates_attrs() -> None:
    g = NetworkXGraphStore()
    g.add_edge("a", "b", weight=1.0, description="first")
    g.add_edge("a", "b", weight=2.0, description="second")
    e = g.get_edge("a", "b")
    assert e["weight"] == 2.0
    assert e["description"] == "second"


def test_iter_nodes() -> None:
    g = NetworkXGraphStore()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("c", "d")
    assert set(g.nodes()) == {"a", "b", "c", "d"}
