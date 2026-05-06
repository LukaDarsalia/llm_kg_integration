"""Abstract base classes for the three storage backends.

Concrete implementations live alongside (e.g. `numpy_vector_store.py`,
`networkx_graph_store.py`) and self-register on the relevant registry. Day 1
ships zero concrete implementations — only contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        vectors: "np.ndarray",
        meta: list[dict[str, Any]],
    ) -> None:
        """Add or overwrite `len(ids)` items. `vectors.shape == (len(ids), dim)`."""

    @abstractmethod
    def search(
        self,
        query: "np.ndarray",
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredHit]:
        """Return top-`k` by descending similarity. `query.shape == (dim,)`."""


class GraphStore(ABC):
    """A directed labelled multigraph store with PPR support."""

    @abstractmethod
    def add_node(self, id: str, **attrs: Any) -> None: ...

    @abstractmethod
    def add_edge(self, src: str, dst: str, **attrs: Any) -> None: ...

    @abstractmethod
    def neighbors(self, id: str) -> list[str]: ...

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
