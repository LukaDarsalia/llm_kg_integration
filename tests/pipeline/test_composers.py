from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext, Stage


class AddOne(Stage[int, int]):
    name = "AddOne"

    async def run(self, inp, ctx):
        return inp + 1


class Double(Stage[int, int]):
    name = "Double"

    async def run(self, inp, ctx):
        return inp * 2


def _empty_ctx() -> PipelineContext:
    return PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )


async def test_indexing_pipeline_runs_stages_in_order() -> None:
    p = IndexingPipeline(stages=[AddOne(), Double()])
    out = await p.run(1, _empty_ctx())
    assert out == 4  # (1+1)*2


async def test_query_pipeline_runs_stages_in_order() -> None:
    p = QueryPipeline(stages=[Double(), AddOne()])
    out = await p.run(1, _empty_ctx())
    assert out == 3  # (1*2)+1


async def test_pipeline_logs_each_stage_when_logger_present() -> None:
    logged: list[tuple[str, float]] = []

    class FakeLogger:
        def log_stage(self, stage, duration_s, **extras):
            logged.append((stage, duration_s))

        def init(self, *a, **k): ...
        def log_metrics(self, *a, **k): ...
        def log_predictions(self, *a, **k): ...
        def log_artifact(self, *a, **k): ...
        def finish(self): ...

    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=FakeLogger(), trace={},
    )
    p = IndexingPipeline(stages=[AddOne(), Double()])
    await p.run(0, ctx)
    assert [name for name, _ in logged] == ["AddOne", "Double"]
    assert all(d >= 0 for _, d in logged)


async def test_empty_pipeline_returns_input_unchanged() -> None:
    p = IndexingPipeline(stages=[])
    assert await p.run(42, _empty_ctx()) == 42
