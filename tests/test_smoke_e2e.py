"""End-to-end smoke test: stub everything that needs a network or disk-heavy
backend, then run a real Experiment through the real runner."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample, ScoredHit
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stages.chunker import DefaultChunker
from llm_kg.pipeline.stages.context_builder import DefaultContextBuilder
from llm_kg.pipeline.stages.embedder import DefaultEmbedder
from llm_kg.pipeline.stages.extractor import NoOpExtractor
from llm_kg.pipeline.stages.generator import DefaultGenerator
from llm_kg.pipeline.stages.graph_builder import NoOpGraphBuilder
from llm_kg.pipeline.stages.query_processor import IdentityQueryProcessor
from llm_kg.pipeline.stages.retriever import Retriever
from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.runner.experiment import Experiment
from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY
from llm_kg.storage.base import GraphStore, KVStore, VectorStore

# ---------- fake providers ----------

@LLM_REGISTRY.register("fake_llm")
class FakeLLM(LLMProvider):
    def __init__(self, model: str = "fake") -> None:
        self.model = model

    async def generate(self, prompt: str, **kwargs):
        return LLMResponse(text="paris", prompt_tokens=1, completion_tokens=1)

    async def generate_structured(self, prompt, schema, **kwargs):
        raise NotImplementedError


@EMBEDDING_REGISTRY.register("fake_embed")
class FakeEmbedder(EmbeddingProvider):
    def __init__(self, model: str = "fake", dim: int = 4) -> None:
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts):
        # deterministic hash-based vectors
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")[: self._dim]):
                out[i, j] = float(ch) / 255.0
        return out


# ---------- fake storage ----------

@VECTOR_REGISTRY.register("fake_vec")
class FakeVecStore(VectorStore):
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vecs: np.ndarray | None = None
        self.meta: list[dict] = []

    def upsert(self, ids, vectors, meta):
        self.ids.extend(ids)
        self.meta.extend(meta)
        self.vecs = vectors if self.vecs is None else np.concatenate([self.vecs, vectors], axis=0)

    def search(self, query, k, filter=None):
        if self.vecs is None or len(self.ids) == 0:
            return []
        # cosine-ish: dot product with normalized
        scores = self.vecs @ query
        top = np.argsort(-scores)[:k]
        return [ScoredHit(id=self.ids[i], score=float(scores[i]), meta=self.meta[i]) for i in top]


@GRAPH_REGISTRY.register("fake_graph")
class FakeGraphStore(GraphStore):
    def __init__(self) -> None:
        self._adj: dict[str, list[str]] = {}

    def add_node(self, id, **attrs):
        self._adj.setdefault(id, [])

    def add_edge(self, src, dst, **attrs):
        self._adj.setdefault(src, []).append(dst)

    def neighbors(self, id):
        return list(self._adj.get(id, []))

    def personalized_pagerank(self, seeds, **kwargs):
        return dict.fromkeys(seeds, 1.0)


@KV_REGISTRY.register("fake_kv")
class FakeKV(KVStore):
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key):
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value

    def __contains__(self, key):
        return key in self._d


# ---------- fake retriever, fake method, fake dataset ----------

class FakeRetriever(Retriever):
    name = "FakeRetriever"

    async def run(self, inp, ctx):
        if ctx.embedder is None or ctx.vector_store is None:
            return []
        qvec = (await ctx.embedder.embed([inp.text]))[0]
        return ctx.vector_store.search(qvec, k=3)


class _ChunkPersistingEmbedder(DefaultEmbedder):
    """Embedder that also stuffs each chunk's text into the KV store."""

    name = "_ChunkPersistingEmbedder"

    async def run(self, inp, ctx):
        for c in inp.chunks:
            ctx.kv_store.put(c.id, c.text)
        return await super().run(inp, ctx)


@METHOD_REGISTRY.register("fake_method")
class FakeMethod(Method):
    name = "fake_method"

    def build(self, cfg, ctx):
        indexing = IndexingPipeline(
            stages=[
                DefaultChunker(chunk_size=4, chunk_overlap=0),
                NoOpExtractor(),
                NoOpGraphBuilder(),
                _ChunkPersistingEmbedder(),
            ]
        )
        query = QueryPipeline(
            stages=[
                IdentityQueryProcessor(),
                FakeRetriever(),
                DefaultContextBuilder(),
                DefaultGenerator(),
            ]
        )
        return indexing, query


@DATASET_REGISTRY.register("fake_ds")
class FakeDataset(QADataset):
    def __init__(self) -> None:
        self._docs = [
            Document(id="d1", text="The capital of France is Paris."),
            Document(id="d2", text="Berlin is the capital of Germany."),
        ]
        self._ex = [QAExample(id="q1", question="capital of france?", answers=["paris"])]

    def examples(self) -> Iterable[QAExample]:
        return iter(self._ex)

    def corpus(self) -> Corpus:
        docs = self._docs

        class _C(Corpus):
            def documents(self):
                return iter(docs)

            def __len__(self):
                return len(docs)

        return _C()

    def __len__(self):
        return len(self._ex)


# ---------- the test ----------

async def test_smoke_e2e_pipeline_runs(tmp_path: Path) -> None:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
method: {name: fake_method, params: {}}
llm: {name: fake_llm, params: {model: fake}, cache: false}
embedder: {name: fake_embed, params: {model: fake, dim: 4}, cache: false}
storage:
  vector: {name: fake_vec, params: {}}
  graph: {name: fake_graph, params: {}}
  kv: {name: fake_kv, params: {}}
dataset: {name: fake_ds, params: {}}
evaluator: {name: extractive, params: {}}
logger: {name: "null"}
cache_dir: """ + str(tmp_path / ".cache") + """
seed: 0
"""
    )
    metrics = await Experiment(cfg).run()
    # FakeLLM always answers "paris" → EM = 1 for the one question.
    assert metrics["em"] == 1.0
    assert metrics["n"] == 1.0
