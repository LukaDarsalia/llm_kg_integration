import pytest

from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY
from llm_kg.storage.base import GraphStore, KVStore, VectorStore


def test_vector_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_graph_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        GraphStore()  # type: ignore[abstract]


def test_kv_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        KVStore()  # type: ignore[abstract]


def test_registries_are_distinct() -> None:
    assert VECTOR_REGISTRY is not GRAPH_REGISTRY
    assert GRAPH_REGISTRY is not KV_REGISTRY
