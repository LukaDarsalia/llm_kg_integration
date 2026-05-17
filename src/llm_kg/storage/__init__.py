"""Storage backend abstractions (vector, graph, KV)."""

from llm_kg.registry import Registry
from llm_kg.storage.base import GraphStore, KVStore, VectorStore

VECTOR_REGISTRY: Registry[VectorStore] = Registry("vector_store")
GRAPH_REGISTRY: Registry[GraphStore] = Registry("graph_store")
KV_REGISTRY: Registry[KVStore] = Registry("kv_store")

# Import concrete implementations so their @register decorators run.
# (Adding a new backend? Add an import line below and a @register on the class.)
from llm_kg.storage import jsonfile_kv, networkx_graph, numpy_vector  # noqa: E402, F401

__all__ = [
    "GRAPH_REGISTRY",
    "GraphStore",
    "KV_REGISTRY",
    "KVStore",
    "VECTOR_REGISTRY",
    "VectorStore",
]
