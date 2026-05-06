import pytest

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.context_builder import ContextBuilder, DefaultContextBuilder
from llm_kg.pipeline.stages.generator import DefaultGenerator, Generator
from llm_kg.pipeline.stages.query_processor import (
    IdentityQueryProcessor,
    ProcessedQuery,
    QueryProcessor,
)
from llm_kg.pipeline.stages.retriever import Retriever
from llm_kg.providers.base import LLMResponse


def _empty_ctx(**overrides) -> PipelineContext:
    base = dict(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    base.update(overrides)
    return PipelineContext(**base)


def test_query_processor_is_abstract() -> None:
    with pytest.raises(TypeError):
        QueryProcessor()  # type: ignore[abstract]


async def test_identity_query_processor_wraps_string() -> None:
    qp = IdentityQueryProcessor()
    out = await qp.run("who is alice?", _empty_ctx())
    assert isinstance(out, ProcessedQuery)
    assert out.text == "who is alice?"
    assert out.keywords == []
    assert out.entities == []


def test_retriever_is_abstract() -> None:
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


def test_retriever_has_no_default() -> None:
    """No DefaultRetriever — every method must define its own."""
    import llm_kg.pipeline.stages.retriever as r

    assert not hasattr(r, "DefaultRetriever")


def test_context_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        ContextBuilder()  # type: ignore[abstract]


async def test_default_context_builder_concatenates_chunks_via_kv() -> None:
    class FakeKV:
        def __init__(self) -> None:
            self.data = {"c1": "alpha text", "c2": "beta text"}

        def get(self, key):
            return self.data[key]

        def put(self, key, value):
            self.data[key] = value

        def __contains__(self, key):
            return key in self.data

    ctx = _empty_ctx(kv_store=FakeKV())
    hits = [ScoredHit(id="c1", score=0.9, meta={}), ScoredHit(id="c2", score=0.8, meta={})]
    ctx_str = await DefaultContextBuilder().run(hits, ctx)
    assert "alpha text" in ctx_str
    assert "beta text" in ctx_str
    # higher score first
    assert ctx_str.find("alpha text") < ctx_str.find("beta text")


async def test_default_context_builder_empty_hits() -> None:
    ctx = _empty_ctx()
    out = await DefaultContextBuilder().run([], ctx)
    assert out == ""


def test_generator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


async def test_default_generator_reads_question_from_ctx_and_calls_llm() -> None:
    seen = {}

    class FakeLLM:
        async def generate(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return LLMResponse(text="42", prompt_tokens=10, completion_tokens=1)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    ctx = _empty_ctx(llm=FakeLLM(), query="what?")
    answer = await DefaultGenerator().run("ctx text", ctx)
    assert answer == "42"
    assert "what?" in seen["prompt"]
    assert "ctx text" in seen["prompt"]


async def test_default_generator_raises_without_query() -> None:
    class FakeLLM:
        async def generate(self, prompt, **kwargs):
            return LLMResponse(text="x", prompt_tokens=1, completion_tokens=1)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    ctx = _empty_ctx(llm=FakeLLM(), query=None)
    with pytest.raises(RuntimeError, match="ctx.query"):
        await DefaultGenerator().run("ctx text", ctx)
