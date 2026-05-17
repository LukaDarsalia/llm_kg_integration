"""Abstract base classes for the three storage backends.

Concrete implementations live alongside (e.g. `numpy_vector_store.py`,
`networkx_graph_store.py`) and self-register on the relevant registry. Day 1
ships zero concrete implementations — only contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from llm_kg.data.types import ScoredHit

if TYPE_CHECKING:
    import numpy as np


class VectorStore(ABC):
    """A dense-vector store. Items are arbitrary string ids with metadata."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        meta: list[dict[str, Any]],
    ) -> None:
        """Add or overwrite `len(ids)` items. `vectors.shape == (len(ids), dim)`."""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredHit]:
        """Return top-`k` by descending similarity. `query.shape == (dim,)`."""


class GraphStore(ABC):
    """An undirected labelled graph store with PPR support.

    Semantics match LightRAG / HippoRAG: relations are treated as undirected
    even when extraction produces a directed (src, dst) pair. `add_edge(a, b)`
    and `add_edge(b, a)` should behave identically; `get_edge` and `neighbors`
    return the same result regardless of direction.

    When `add_edge` is called with an existing pair, attrs MUST be updated
    (last-write-wins per attribute). Methods that want merge semantics
    (sum weights, union keywords) should read the current attrs via `get_edge`,
    compute the merged values, and re-call `add_edge`.
    """

    @abstractmethod
    def add_node(self, id: str, **attrs: Any) -> None:
        """Add or update a node with the given attributes (last-write-wins per attr)."""

    @abstractmethod
    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        """Add or update the edge between `src` and `dst` (undirected). Endpoints
        are created if missing.
        """

    @abstractmethod
    def get_node(self, id: str) -> dict[str, Any] | None:
        """Return the node's attribute dict, or None if absent."""

    @abstractmethod
    def get_edge(self, src: str, dst: str) -> dict[str, Any] | None:
        """Return the edge's attribute dict, or None if absent."""

    @abstractmethod
    def neighbors(self, id: str) -> list[str]:
        """Return all neighbours of `id` (undirected). Empty list if `id` is absent."""

    @abstractmethod
    def node_edges(self, id: str) -> list[tuple[str, dict[str, Any]]]:
        """Return `(neighbour_id, edge_attrs)` for every edge incident to `id`."""

    @abstractmethod
    def node_degree(self, id: str) -> int:
        """Number of edges incident to `id`. Zero if `id` is absent."""

    @abstractmethod
    def nodes(self) -> Iterable[str]:
        """Iterate over all node ids."""

    @abstractmethod
    def __contains__(self, id: str) -> bool: ...

    @abstractmethod
    def personalized_pagerank(
        self,
        seeds: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """PPR over the graph with bias toward `seeds` (id → weight). Returns score per node."""


class KVStore(ABC):
    """A simple typed key-value store. Used for chunk/document text and method-specific blobs."""

    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def put(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def __contains__(self, key: str) -> bool: ...
