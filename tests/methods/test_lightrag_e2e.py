"""End-to-end smoke test: real LightRAG pipelines, stubbed providers.

Runs the full indexing + query pipeline against a fake LLM that emits
deterministic LightRAG-formatted extraction and a fake embedder. Asserts
the whole graph composes and produces a non-empty answer.
"""

import numpy as np

from llm_kg.data.types import Document, QAExample
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import MethodConfig
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.storage.jsonfile_kv import JsonFileKVStore
from llm_kg.storage.networkx_graph import NetworkXGraphStore
from llm_kg.storage.numpy_vector import NumpyVectorStore

TD = "<|#|>"
CD = "<|COMPLETE|>"


class _ScriptedLLM(LLMProvider):
    """LLM that returns extraction output if the prompt looks like extraction,
    keyword JSON if it looks like keyword extraction, and `paris` otherwise."""

    EXTRACTION_RESPONSE = (
        f"entity{TD}Paris{TD}location{TD}Capital of France.\n"
        f"entity{TD}France{TD}location{TD}European country whose capital is Paris.\n"
        f"relation{TD}Paris{TD}France{TD}capital, geography{TD}Paris is the capital of France.\n"
        f"{CD}\n"
    )
    KEYWORDS_RESPONSE = '{"high_level_keywords": ["geography"], "low_level_keywords": ["france", "paris"]}'

    async def generate(self, prompt: str, **kwargs):
        p = prompt.lower()
        if "knowledge graph specialist" in p or "extract entities" in p:
            text = self.EXTRACTION_RESPONSE
        elif "keyword extractor" in p or "low_level_keywords" in p:
            text = self.KEYWORDS_RESPONSE
        else:
            text = "Paris"  # final answer
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=5)

    async def generate_structured(self, *a, **k):
        raise NotImplementedError


class _DeterministicEmbedder(EmbeddingProvider):
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


async def test_lightrag_end_to_end() -> None:
    """Index a 1-document corpus, query it, get an answer."""
    docs = [
        Document(id="d1", text="Paris is the capital of France. France is in Europe.",
                 metadata={"title": "Paris"}),
    ]

    ctx = PipelineContext(
        llm=_ScriptedLLM(),
        embedder=_DeterministicEmbedder(dim=8),
        vector_store=NumpyVectorStore(),
        graph_store=NetworkXGraphStore(),
        kv_store=JsonFileKVStore(path=None),
        logger=None,
        trace={},
    )
    method = METHOD_REGISTRY.get("lightrag")()
    indexing, query = method.build(
        MethodConfig(
            name="lightrag",
            params={
                "chunk_size": 50,    # force the test doc into 1 chunk
                "chunk_overlap": 0,
                "max_gleaning": 0,
                "top_k_entities": 5,
                "top_k_chunks": 3,
                "cosine_threshold": 0.0,
            },
        ),
        ctx,
    )

    # --- indexing ---
    await indexing.run(docs, ctx)

    # Sanity: graph should have Paris + France nodes and the edge between them
    assert "Paris" in ctx.graph_store
    assert "France" in ctx.graph_store
    assert ctx.graph_store.get_edge("Paris", "France") is not None

    # --- query ---
    ex = QAExample(id="q1", question="What is the capital of France?", answers=["paris"])
    ctx.query = ex.question  # runner normally does this
    answer = await query.run(ex.question, ctx)
    assert answer == "Paris"


async def test_lightrag_indexing_produces_chunks_entities_relations() -> None:
    """Verify all three "kinds" land in the vector store after indexing."""
    docs = [Document(id="d1", text="Alice met Bob in Paris.", metadata={"title": "X"})]

    vs = NumpyVectorStore()
    ctx = PipelineContext(
        llm=_ScriptedLLM(),
        embedder=_DeterministicEmbedder(dim=8),
        vector_store=vs,
        graph_store=NetworkXGraphStore(),
        kv_store=JsonFileKVStore(path=None),
        logger=None,
        trace={},
    )
    method = METHOD_REGISTRY.get("lightrag")()
    indexing, _ = method.build(
        MethodConfig(name="lightrag", params={"chunk_size": 50, "chunk_overlap": 0, "max_gleaning": 0}),
        ctx,
    )
    await indexing.run(docs, ctx)

    q = np.ones(8, dtype=np.float32)
    assert len(vs.search(q, k=10, filter={"kind": "chunk"})) >= 1
    assert len(vs.search(q, k=10, filter={"kind": "entity"})) >= 2
    assert len(vs.search(q, k=10, filter={"kind": "relation"})) >= 1
