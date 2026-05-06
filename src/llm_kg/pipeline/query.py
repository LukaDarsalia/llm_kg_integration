"""Sequential query pipeline composer. See `indexing.py` for design notes."""

from __future__ import annotations

import time
from typing import Any

from llm_kg.pipeline.stage import PipelineContext, Stage


class QueryPipeline:
    """Runs a list of `Stage`s in order for one query."""

    def __init__(self, stages: list[Stage[Any, Any]]) -> None:
        self.stages = stages

    async def run(self, inp: Any, ctx: PipelineContext) -> Any:
        current: Any = inp
        for stage in self.stages:
            t0 = time.perf_counter()
            current = await stage.run(current, ctx)
            elapsed = time.perf_counter() - t0
            if ctx.logger is not None:
                ctx.logger.log_stage(stage.name, elapsed)
        return current
