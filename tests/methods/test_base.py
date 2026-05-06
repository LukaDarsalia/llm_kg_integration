import pytest

from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method, MethodConfig
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext


def test_method_is_abstract() -> None:
    with pytest.raises(TypeError):
        Method()  # type: ignore[abstract]


def test_method_config_holds_arbitrary_params() -> None:
    cfg = MethodConfig(name="x", params={"chunk_size": 256, "top_k": 5})
    assert cfg.params["chunk_size"] == 256
    assert cfg.name == "x"


def test_concrete_method_returns_two_pipelines() -> None:
    class FakeMethod(Method):
        name = "fake"

        def build(self, cfg, ctx):
            return IndexingPipeline(stages=[]), QueryPipeline(stages=[])

    m = FakeMethod()
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    indexing, query = m.build(MethodConfig(name="fake"), ctx)
    assert isinstance(indexing, IndexingPipeline)
    assert isinstance(query, QueryPipeline)


def test_method_registry_constructed() -> None:
    # Day 1 ships no method implementations.
    assert "naive_rag" not in METHOD_REGISTRY
    assert "hipporag" not in METHOD_REGISTRY
    assert "lightrag" not in METHOD_REGISTRY
