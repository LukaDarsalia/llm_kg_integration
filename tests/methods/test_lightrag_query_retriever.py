"""Tests for LightRAGQueryProcessor + LightRAGRetriever."""

import numpy as np

from llm_kg.data.types import Chunk, Entity, Relation
from llm_kg.methods.lightrag.embedder import (
    LightRAGEmbedder,
    _entity_id,
    _relation_id,
)
from llm_kg.methods.lightrag.graph_builder import LightRAGGraphBuilder
from llm_kg.methods.lightrag.query_processor import (
    LightRAGQueryProcessor,
    _parse_keywords_json,
)
from llm_kg.methods.lightrag.retriever import LightRAGRetriever
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.extractor import ExtractionResult
from llm_kg.pipeline.stages.query_processor import ProcessedQuery
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.storage.jsonfile_kv import JsonFileKVStore
from llm_kg.storage.networkx_graph import NetworkXGraphStore
from llm_kg.storage.numpy_vector import NumpyVectorStore


# ---------- _parse_keywords_json ----------

def test_parse_keywords_strict_json() -> None:
    raw = '{"high_level_keywords": ["a"], "low_level_keywords": ["b", "c"]}'
    hl, ll = _parse_keywords_json(raw)
    assert hl == ["a"]
    assert ll == ["b", "c"]


def test_parse_keywords_handles_code_fence() -> None:
    raw = '```json\n{"high_level_keywords": ["x"], "low_level_keywords": []}\n```'
    hl, ll = _parse_keywords_json(raw)
    assert hl == ["x"]
    assert ll == []


def test_parse_keywords_extracts_json_blob_from_chatter() -> None:
    raw = 'sure thing! {"high_level_keywords": ["a"], "low_level_keywords": ["b"]} hope this helps'
    hl, ll = _parse_keywords_json(raw)
    assert hl == ["a"]
    assert ll == ["b"]


def test_parse_keywords_invalid_returns_empty() -> None:
    hl, ll = _parse_keywords_json("nothing useful here")
    assert hl == [] and ll == []


# ---------- QueryProcessor ----------

async def test_query_processor_extracts_dual_levels() -> None:
    canned = '{"high_level_keywords": ["geography"], "low_level_keywords": ["france", "paris"]}'

    class _LLM(LLMProvider):
        async def generate(self, prompt, **kwargs):
            return LLMResponse(text=canned, prompt_tokens=10, completion_tokens=5)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    qp = LightRAGQueryProcessor()
    ctx = PipelineContext(
        llm=_LLM(), embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    pq = await qp.run("what is the capital of france?", ctx)
    assert isinstance(pq, ProcessedQuery)
    assert pq.entities == ["france", "paris"]       # low-level
    assert pq.meta["high_level"] == ["geography"]    # high-level
    assert set(pq.keywords) == {"geography", "france", "paris"}


async def test_query_processor_short_empty_keywords_falls_back_to_query() -> None:
    class _LLM(LLMProvider):
        async def generate(self, prompt, **kwargs):
            return LLMResponse(
                text='{"high_level_keywords": [], "low_level_keywords": []}',
                prompt_tokens=1, completion_tokens=1,
            )

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    qp = LightRAGQueryProcessor()
    ctx = PipelineContext(
        llm=_LLM(), embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    pq = await qp.run("hi", ctx)
    assert pq.entities == ["hi"]


# ---------- Retriever ----------

class _Embedder(EmbeddingProvider):
    """Deterministic per-text embedding: token-character based, normalized."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")):
                out[i, j % self._dim] += float(ch) / 255.0
        return out


async def _build_indexed_state() -> PipelineContext:
    """Run extractor outputs through GraphBuilder + Embedder to simulate a tiny
    indexed corpus: 2 chunks, 2 entities, 1 relation."""
    vs = NumpyVectorStore()
    gs = NetworkXGraphStore()
    kv = JsonFileKVStore(path=None)
    emb = _Embedder(dim=8)
    ctx = PipelineContext(
        llm=None, embedder=emb,
        vector_store=vs, graph_store=gs, kv_store=kv,
        logger=None, trace={},
    )

    er = ExtractionResult(
        chunks=[
            Chunk(id="c1", text="Paris is the capital of France.", doc_id="d1", position=0),
            Chunk(id="c2", text="Berlin is the capital of Germany.", doc_id="d2", position=0),
        ],
        entities=[
            Entity(id="Paris", name="Paris", type="location",
                   metadata={"description": "Capital of France.", "source_chunks": ["c1"]}),
            Entity(id="France", name="France", type="location",
                   metadata={"description": "European country.", "source_chunks": ["c1"]}),
            Entity(id="Berlin", name="Berlin", type="location",
                   metadata={"description": "Capital of Germany.", "source_chunks": ["c2"]}),
        ],
        relations=[
            Relation(src="Paris", dst="France", predicate="capital of",
                     metadata={"description": "Paris is the capital.",
                               "keywords": ["capital"], "weight": 1.0,
                               "source_chunks": ["c1"]}),
        ],
    )
    await LightRAGGraphBuilder().run(er, ctx)
    await LightRAGEmbedder().run(er, ctx)
    return ctx


async def test_retriever_returns_chunk_hits() -> None:
    ctx = await _build_indexed_state()
    pq = ProcessedQuery(text="capital of France", entities=["France", "Paris"],
                        meta={"high_level": ["geography"]})
    retriever = LightRAGRetriever(top_k_entities=10, top_k_chunks=5, cosine_threshold=0.0)
    hits = await retriever.run(pq, ctx)
    # Should retrieve at least c1; both c1 and c2 are valid candidates.
    assert len(hits) >= 1
    ids = {h.id for h in hits}
    assert "c1" in ids


async def test_retriever_returns_empty_when_index_is_empty() -> None:
    vs = NumpyVectorStore()
    gs = NetworkXGraphStore()
    kv = JsonFileKVStore(path=None)
    ctx = PipelineContext(
        llm=None, embedder=_Embedder(),
        vector_store=vs, graph_store=gs, kv_store=kv,
        logger=None, trace={},
    )
    pq = ProcessedQuery(text="anything", entities=[], meta={"high_level": []})
    assert await LightRAGRetriever().run(pq, ctx) == []


async def test_retriever_respects_top_k_chunks() -> None:
    ctx = await _build_indexed_state()
    pq = ProcessedQuery(text="capital", entities=["France"], meta={"high_level": []})
    hits = await LightRAGRetriever(top_k_chunks=1, cosine_threshold=0.0).run(pq, ctx)
    assert len(hits) <= 1


async def test_retriever_entity_lookup_uses_entity_name_meta() -> None:
    """Smoke check that the entity hit -> graph node -> source_id path works."""
    ctx = await _build_indexed_state()
    # Verify scaffolding: entity_id mapping resolves correctly
    ent_id = _entity_id("Paris")
    assert ent_id in ctx.kv_store
    # Verify the graph has the chunk pointer back through source_id
    node = ctx.graph_store.get_node("Paris")
    assert node is not None
    assert "c1" in node["source_id"]


async def test_retriever_relation_lookup_uses_undirected_pair() -> None:
    ctx = await _build_indexed_state()
    rel_id = _relation_id("Paris", "France")
    assert rel_id in ctx.kv_store  # KV was populated by the embedder
    # Edge should exist regardless of direction
    assert ctx.graph_store.get_edge("Paris", "France") is not None
    assert ctx.graph_store.get_edge("France", "Paris") is not None
