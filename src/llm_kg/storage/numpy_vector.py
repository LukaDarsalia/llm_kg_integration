"""NumPy-backed in-memory vector store with cosine similarity.

Vectors are L2-normalized on insert so search reduces to a single matrix-vector
dot product. Suitable for tens to low-hundreds of thousands of vectors against
a brute-force search; beyond that, swap in a real ANN backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_kg.data.types import ScoredHit
from llm_kg.storage import VECTOR_REGISTRY
from llm_kg.storage.base import VectorStore


@VECTOR_REGISTRY.register("numpy")
class NumpyVectorStore(VectorStore):
    """In-memory cosine vector store with optional metadata filtering."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._vectors: np.ndarray | None = None  # shape (N, dim), unit-norm
        self._meta: list[dict[str, Any]] = []

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        meta: list[dict[str, Any]],
    ) -> None:
        if len(ids) == 0:
            return
        if vectors.shape[0] != len(ids) or len(meta) != len(ids):
            raise ValueError(
                f"length mismatch: ids={len(ids)}, vectors={vectors.shape[0]}, meta={len(meta)}"
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = (vectors / norms).astype(np.float32)

        # Updates happen in place; collect inserts for one batched concat at the end.
        insert_idx: list[int] = []
        for i, item_id in enumerate(ids):
            if item_id in self._id_to_idx:
                idx = self._id_to_idx[item_id]
                self._vectors[idx] = normalized[i]  # type: ignore[index]
                self._meta[idx] = meta[i]
            else:
                insert_idx.append(i)

        if insert_idx:
            new_vecs = normalized[insert_idx]
            start = len(self._ids)
            for j, src_i in enumerate(insert_idx):
                self._id_to_idx[ids[src_i]] = start + j
                self._ids.append(ids[src_i])
                self._meta.append(meta[src_i])
            if self._vectors is None:
                self._vectors = new_vecs.copy()
            else:
                self._vectors = np.concatenate([self._vectors, new_vecs], axis=0)

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredHit]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        q_norm = float(np.linalg.norm(query))
        if q_norm == 0:
            return []
        q = (query / q_norm).astype(np.float32)

        if filter is None:
            indices = np.arange(len(self._ids), dtype=np.int64)
        else:
            indices = np.array(
                [
                    i
                    for i, m in enumerate(self._meta)
                    if all(m.get(fk) == fv for fk, fv in filter.items())
                ],
                dtype=np.int64,
            )
            if indices.size == 0:
                return []

        scores = self._vectors[indices] @ q  # cosine since both are unit-norm
        top = np.argsort(-scores)[: max(k, 0)]
        return [
            ScoredHit(
                id=self._ids[int(indices[i])],
                score=float(scores[i]),
                meta=self._meta[int(indices[i])],
            )
            for i in top
        ]

    def __len__(self) -> int:
        return len(self._ids)
