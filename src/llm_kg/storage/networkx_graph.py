"""NetworkX-backed in-memory graph store with personalized PageRank.

Uses `networkx.Graph` (undirected, simple) — exactly what LightRAG and HippoRAG
assume. Re-adding an edge between the same pair last-writes-wins per attribute,
so methods that want merge semantics (sum weights, union keywords) need to
`get_edge` first, compute the merge, then re-`add_edge`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx

from llm_kg.storage import GRAPH_REGISTRY
from llm_kg.storage.base import GraphStore


@GRAPH_REGISTRY.register("networkx")
class NetworkXGraphStore(GraphStore):
    """Undirected attributed graph backed by `networkx.Graph`."""

    def __init__(self) -> None:
        self._g: nx.Graph = nx.Graph()

    def add_node(self, id: str, **attrs: Any) -> None:
        if id in self._g:
            self._g.nodes[id].update(attrs)
        else:
            self._g.add_node(id, **attrs)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        if self._g.has_edge(src, dst):
            self._g[src][dst].update(attrs)
        else:
            self._g.add_edge(src, dst, **attrs)

    def get_node(self, id: str) -> dict[str, Any] | None:
        if id not in self._g:
            return None
        return dict(self._g.nodes[id])

    def get_edge(self, src: str, dst: str) -> dict[str, Any] | None:
        if not self._g.has_edge(src, dst):
            return None
        return dict(self._g[src][dst])

    def neighbors(self, id: str) -> list[str]:
        if id not in self._g:
            return []
        return list(self._g.neighbors(id))

    def node_edges(self, id: str) -> list[tuple[str, dict[str, Any]]]:
        if id not in self._g:
            return []
        return [(other, dict(self._g[id][other])) for other in self._g.neighbors(id)]

    def node_degree(self, id: str) -> int:
        if id not in self._g:
            return 0
        return int(self._g.degree[id])

    def nodes(self) -> Iterable[str]:
        return list(self._g.nodes)

    def __contains__(self, id: str) -> bool:
        return id in self._g

    def personalized_pagerank(
        self,
        seeds: dict[str, float],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1.0e-6,
        **kwargs: Any,
    ) -> dict[str, float]:
        if len(self._g) == 0:
            return {}
        # Drop seeds for nodes not in the graph so networkx doesn't choke.
        present = {k: v for k, v in seeds.items() if k in self._g}
        personalization = present if present else None
        return dict(
            nx.pagerank(
                self._g,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol,
            )
        )
