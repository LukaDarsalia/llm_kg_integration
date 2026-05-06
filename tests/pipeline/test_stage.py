import pytest

from llm_kg.pipeline.stage import PipelineContext, Stage


class AddOne(Stage[int, int]):
    name = "AddOne"

    async def run(self, inp: int, ctx: PipelineContext) -> int:
        return inp + 1


def test_stage_is_abstract() -> None:
    with pytest.raises(TypeError):
        Stage()  # type: ignore[abstract]


async def test_concrete_stage_runs() -> None:
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    s = AddOne()
    assert await s.run(1, ctx) == 2


def test_pipeline_context_trace_is_mutable() -> None:
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    ctx.trace["x"] = 1
    assert ctx.trace["x"] == 1
