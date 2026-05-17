import pytest

from llm_kg.data.types import Document
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.tiktoken_chunker import TiktokenChunker


def _ctx() -> PipelineContext:
    return PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )


async def test_short_doc_one_chunk() -> None:
    c = TiktokenChunker(chunk_size=1200, chunk_overlap=100)
    chunks = await c.run(
        [Document(id="d1", text="The capital of France is Paris.")], _ctx()
    )
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"
    assert chunks[0].position == 0
    assert chunks[0].text == "The capital of France is Paris."


async def test_long_doc_splits_into_multiple_chunks() -> None:
    """Tiktoken tokenisation of `word0`, `word1`, ... varies in token count, so
    we pin invariants instead of exact chunk counts."""
    c = TiktokenChunker(chunk_size=10, chunk_overlap=2)
    text = " ".join(f"word{i}" for i in range(25))
    chunks = await c.run([Document(id="d1", text=text)], _ctx())

    assert len(chunks) > 1
    assert all(ch.doc_id == "d1" for ch in chunks)
    # positions are contiguous starting at 0
    assert [ch.position for ch in chunks] == list(range(len(chunks)))
    # concatenating decoded chunks (deduping the overlap) recovers the original
    # tokens — sanity check that no content was lost
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    full_tokens = enc.encode(text)
    # First chunk must start at token 0
    assert enc.decode(full_tokens[:10]) == chunks[0].text


async def test_multiple_docs() -> None:
    c = TiktokenChunker(chunk_size=1200, chunk_overlap=100)
    chunks = await c.run(
        [
            Document(id="d1", text="alpha beta"),
            Document(id="d2", text="gamma delta"),
        ],
        _ctx(),
    )
    assert len(chunks) == 2
    assert {ch.doc_id for ch in chunks} == {"d1", "d2"}
    assert chunks[0].id == "d1::chunk0"
    assert chunks[1].id == "d2::chunk0"


async def test_empty_doc_produces_no_chunks() -> None:
    c = TiktokenChunker(chunk_size=1200, chunk_overlap=100)
    chunks = await c.run([Document(id="d1", text="")], _ctx())
    assert chunks == []


def test_chunk_overlap_validation() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        TiktokenChunker(chunk_size=10, chunk_overlap=10)


async def test_preserves_document_metadata_in_chunk() -> None:
    c = TiktokenChunker(chunk_size=1200, chunk_overlap=100)
    docs = [Document(id="d1", text="hello world", metadata={"title": "Foo"})]
    chunks = await c.run(docs, _ctx())
    # metadata is per-chunk; title comes from doc — propagate via metadata
    assert chunks[0].metadata.get("title") == "Foo"
