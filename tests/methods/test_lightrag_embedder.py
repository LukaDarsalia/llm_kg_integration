import numpy as np

from llm_kg.data.types import Chunk, Entity, Relation
from llm_kg.methods.lightrag.embedder import (
    LightRAGEmbedder,
    _chunk_vec_id,
    _entity_id,
    _relation_id,
)
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.extractor import ExtractionResult
from llm_kg.providers.base import EmbeddingProvider
from llm_kg.storage.jsonfile_kv import JsonFileKVStore
from llm_kg.storage.numpy_vector import NumpyVectorStore


class _DeterministicEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")[: self._dim]):
                out[i, j] = float(ch) / 255.0
        return out


def _ctx(emb, vs, kv) -> PipelineContext:
    return PipelineContext(
        llm=None, embedder=emb,
        vector_store=vs, graph_store=None, kv_store=kv,
        logger=None, trace={},
    )


async def test_embedder_writes_three_kinds_to_vector_store() -> None:
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    emb = _DeterministicEmbedder()

    er = ExtractionResult(
        chunks=[Chunk(id="c1", text="Alice met Bob.", doc_id="d1", position=0)],
        entities=[
            Entity(id="Alice", name="Alice", type="person",
                   metadata={"description": "founder of X"}),
            Entity(id="Bob", name="Bob", type="person",
                   metadata={"description": "engineer"}),
        ],
        relations=[
            Relation(src="Alice", dst="Bob", predicate="",
                     metadata={"description": "co-workers", "keywords": ["work"], "weight": 1.0}),
        ],
    )
    await LightRAGEmbedder().run(er, _ctx(emb, vs, kv))

    # 1 chunk + 2 entities + 1 relation = 4 vectors total
    assert len(vs) == 4

    # filter by kind
    q = np.ones(4, dtype=np.float32)
    chunk_hits = vs.search(q, k=10, filter={"kind": "chunk"})
    ent_hits = vs.search(q, k=10, filter={"kind": "entity"})
    rel_hits = vs.search(q, k=10, filter={"kind": "relation"})
    assert {h.id for h in chunk_hits} == {_chunk_vec_id("c1")}
    assert {h.id for h in ent_hits} == {_entity_id("Alice"), _entity_id("Bob")}
    assert {h.id for h in rel_hits} == {_relation_id("Alice", "Bob")}


async def test_embedder_persists_chunk_text_to_kv() -> None:
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    er = ExtractionResult(
        chunks=[Chunk(id="c1", text="hello", doc_id="d", position=0)],
        entities=[], relations=[],
    )
    await LightRAGEmbedder().run(er, _ctx(_DeterministicEmbedder(), vs, kv))
    assert kv.get("c1") == "hello"


async def test_embedder_persists_entity_text_to_kv() -> None:
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    er = ExtractionResult(
        chunks=[],
        entities=[Entity(id="X", name="X", type="t",
                         metadata={"description": "desc"})],
        relations=[],
    )
    await LightRAGEmbedder().run(er, _ctx(_DeterministicEmbedder(), vs, kv))
    assert kv.get(_entity_id("X")) == "X\ndesc"


async def test_embedder_dedupes_entities_across_extractions() -> None:
    """Same entity name appearing twice should be embedded once."""
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    er = ExtractionResult(
        chunks=[],
        entities=[
            Entity(id="Alice", name="Alice", type="person",
                   metadata={"description": "d1"}),
            Entity(id="Alice", name="Alice", type="person",
                   metadata={"description": "d2"}),
        ],
        relations=[],
    )
    await LightRAGEmbedder().run(er, _ctx(_DeterministicEmbedder(), vs, kv))
    ent_hits = vs.search(np.ones(4, dtype=np.float32), k=10, filter={"kind": "entity"})
    assert len(ent_hits) == 1
    # KV concatenates descriptions
    stored = kv.get(_entity_id("Alice"))
    assert "d1" in stored and "d2" in stored


async def test_embedder_dedupes_relations_by_undirected_pair() -> None:
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    er = ExtractionResult(
        chunks=[], entities=[],
        relations=[
            Relation(src="A", dst="B", predicate="",
                     metadata={"description": "d1", "keywords": ["k1"], "weight": 1.0}),
            Relation(src="B", dst="A", predicate="",  # reversed
                     metadata={"description": "d2", "keywords": ["k2"], "weight": 1.0}),
        ],
    )
    await LightRAGEmbedder().run(er, _ctx(_DeterministicEmbedder(), vs, kv))
    rel_hits = vs.search(np.ones(4, dtype=np.float32), k=10, filter={"kind": "relation"})
    assert len(rel_hits) == 1
    stored = kv.get(_relation_id("A", "B"))
    assert "k1" in stored and "k2" in stored


async def test_embedder_handles_empty_extraction_result() -> None:
    vs = NumpyVectorStore()
    kv = JsonFileKVStore(path=None)
    er = ExtractionResult(chunks=[], entities=[], relations=[])
    out = await LightRAGEmbedder().run(er, _ctx(_DeterministicEmbedder(), vs, kv))
    assert out is er
    assert len(vs) == 0
