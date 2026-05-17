import numpy as np
import pytest

from llm_kg.storage import VECTOR_REGISTRY
from llm_kg.storage.numpy_vector import NumpyVectorStore


def test_registered_as_numpy() -> None:
    assert "numpy" in VECTOR_REGISTRY
    assert VECTOR_REGISTRY.get("numpy") is NumpyVectorStore


def test_empty_store_returns_no_hits() -> None:
    vs = NumpyVectorStore()
    assert vs.search(np.array([1.0, 0.0, 0.0]), k=5) == []


def test_upsert_then_search_returns_nearest_first() -> None:
    vs = NumpyVectorStore()
    vs.upsert(
        ids=["a", "b", "c"],
        vectors=np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32),
        meta=[{}, {}, {}],
    )
    hits = vs.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [h.id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score == pytest.approx(0.7 / (0.7**2 + 0.7**2) ** 0.5, abs=1e-6)


def test_top_k_truncates_results() -> None:
    vs = NumpyVectorStore()
    vs.upsert(
        ids=[f"v{i}" for i in range(10)],
        vectors=np.random.rand(10, 4).astype(np.float32),
        meta=[{} for _ in range(10)],
    )
    assert len(vs.search(np.random.rand(4).astype(np.float32), k=3)) == 3


def test_filter_restricts_results_by_meta() -> None:
    vs = NumpyVectorStore()
    vs.upsert(
        ids=["e1", "e2", "r1", "c1"],
        vectors=np.eye(4, 4, dtype=np.float32),
        meta=[
            {"kind": "entity"},
            {"kind": "entity"},
            {"kind": "relation"},
            {"kind": "chunk"},
        ],
    )
    q = np.array([1, 1, 1, 1], dtype=np.float32)
    ent_hits = vs.search(q, k=10, filter={"kind": "entity"})
    assert {h.id for h in ent_hits} == {"e1", "e2"}

    rel_hits = vs.search(q, k=10, filter={"kind": "relation"})
    assert {h.id for h in rel_hits} == {"r1"}


def test_filter_returns_empty_when_no_match() -> None:
    vs = NumpyVectorStore()
    vs.upsert(
        ids=["a"], vectors=np.eye(1, 3, dtype=np.float32), meta=[{"kind": "entity"}]
    )
    assert vs.search(np.eye(1, 3)[0], k=5, filter={"kind": "missing"}) == []


def test_reupsert_updates_in_place() -> None:
    vs = NumpyVectorStore()
    vs.upsert(ids=["x"], vectors=np.array([[1.0, 0.0]], dtype=np.float32), meta=[{"v": 1}])
    vs.upsert(ids=["x"], vectors=np.array([[0.0, 1.0]], dtype=np.float32), meta=[{"v": 2}])

    hits_x_axis = vs.search(np.array([1.0, 0.0], dtype=np.float32), k=1)
    hits_y_axis = vs.search(np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert hits_x_axis[0].score == pytest.approx(0.0, abs=1e-6)  # x updated to y-axis
    assert hits_y_axis[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits_y_axis[0].meta == {"v": 2}


def test_batched_upsert_then_more() -> None:
    """Two upsert calls; second batch appends. Result merges both."""
    vs = NumpyVectorStore()
    vs.upsert(ids=["a", "b"], vectors=np.eye(2, 3, dtype=np.float32), meta=[{}, {}])
    vs.upsert(ids=["c"], vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float32), meta=[{}])
    hits = vs.search(np.array([0.0, 0.0, 1.0], dtype=np.float32), k=3)
    assert hits[0].id == "c"


def test_empty_upsert_is_noop() -> None:
    vs = NumpyVectorStore()
    vs.upsert(ids=[], vectors=np.zeros((0, 4), dtype=np.float32), meta=[])
    assert vs.search(np.zeros(4, dtype=np.float32), k=5) == []


def test_zero_vector_query_returns_empty() -> None:
    vs = NumpyVectorStore()
    vs.upsert(ids=["a"], vectors=np.array([[1.0, 0.0]], dtype=np.float32), meta=[{}])
    assert vs.search(np.zeros(2, dtype=np.float32), k=1) == []


def test_length_mismatch_raises() -> None:
    vs = NumpyVectorStore()
    with pytest.raises(ValueError, match="length mismatch"):
        vs.upsert(ids=["a", "b"], vectors=np.eye(1, 2, dtype=np.float32), meta=[{}, {}])
