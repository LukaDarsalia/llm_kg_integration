import numpy as np
import pytest

from llm_kg.data.types import Chunk, Document, Entity, Relation
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.chunker import Chunker, DefaultChunker
from llm_kg.pipeline.stages.embedder import DefaultEmbedder, Embedder
from llm_kg.pipeline.stages.extractor import (
    ExtractionResult,
    InformationExtractor,
    NoOpExtractor,
)
from llm_kg.pipeline.stages.graph_builder import GraphBuilder, NoOpGraphBuilder


def _empty_ctx(**overrides) -> PipelineContext:
    base = dict(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    base.update(overrides)
    return PipelineContext(**base)


def test_chunker_is_abstract() -> None:
    with pytest.raises(TypeError):
        Chunker()  # type: ignore[abstract]


async def test_default_chunker_splits_by_word_count() -> None:
    chunker = DefaultChunker(chunk_size=5, chunk_overlap=0)
    docs = [Document(id="d1", text=" ".join(f"w{i}" for i in range(12)))]
    chunks = await chunker.run(docs, _empty_ctx())
    assert len(chunks) == 3
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    assert chunks[0].text.split() == ["w0", "w1", "w2", "w3", "w4"]


async def test_default_chunker_overlap() -> None:
    chunker = DefaultChunker(chunk_size=4, chunk_overlap=1)
    docs = [Document(id="d1", text=" ".join(f"w{i}" for i in range(7)))]
    chunks = await chunker.run(docs, _empty_ctx())
    # window 4, stride 3 (4 - 1) over 7 words → starts at 0, 3 → 2 chunks
    assert len(chunks) == 2
    # last word of chunk 0 == first word of chunk 1 (the overlap)
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]
    assert chunks[0].text.split()[-1] == "w3"


async def test_default_chunker_short_doc_one_chunk() -> None:
    chunker = DefaultChunker(chunk_size=100, chunk_overlap=0)
    docs = [Document(id="d1", text="only a few words")]
    chunks = await chunker.run(docs, _empty_ctx())
    assert len(chunks) == 1
    assert chunks[0].text == "only a few words"


def test_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        InformationExtractor()  # type: ignore[abstract]


async def test_noop_extractor_returns_empty_extraction() -> None:
    ext = NoOpExtractor()
    chunks = [Chunk(id="c1", text="x", doc_id="d1", position=0)]
    out = await ext.run(chunks, _empty_ctx())
    assert isinstance(out, ExtractionResult)
    assert out.chunks == chunks
    assert out.entities == []
    assert out.relations == []


def test_graph_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        GraphBuilder()  # type: ignore[abstract]


async def test_noop_graph_builder_passes_through() -> None:
    gb = NoOpGraphBuilder()
    er = ExtractionResult(
        chunks=[Chunk(id="c1", text="x", doc_id="d1", position=0)],
        entities=[Entity(id="alice", name="Alice")],
        relations=[Relation(src="alice", dst="bob", predicate="knows")],
    )
    out = await gb.run(er, _empty_ctx())
    assert out is er


def test_embedder_is_abstract() -> None:
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


async def test_default_embedder_writes_chunk_vectors_to_store() -> None:
    class FakeEmbedder:
        @property
        def dim(self) -> int:
            return 3

        async def embed(self, texts):
            return np.array([[float(len(t))] * 3 for t in texts])

    upserts: list[tuple[list[str], np.ndarray, list[dict]]] = []

    class FakeVecStore:
        def upsert(self, ids, vectors, meta):
            upserts.append((ids, vectors, meta))

        def search(self, q, k, filter=None):
            return []

    ctx = _empty_ctx(embedder=FakeEmbedder(), vector_store=FakeVecStore())
    er = ExtractionResult(
        chunks=[
            Chunk(id="c1", text="ab", doc_id="d1", position=0),
            Chunk(id="c2", text="xyz", doc_id="d1", position=1),
        ],
        entities=[],
        relations=[],
    )
    out = await DefaultEmbedder().run(er, ctx)
    assert out is er  # passes through
    assert len(upserts) == 1
    ids, vecs, metas = upserts[0]
    assert ids == ["c1", "c2"]
    assert vecs.shape == (2, 3)
    assert metas[0]["doc_id"] == "d1"
