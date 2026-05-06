"""Sequential indexing pipeline composer.

The composer is intentionally type-loose: it threads the output of each stage
into the next without static checks. The Method that assembles the pipeline is
responsible for stage compatibility — see the design doc, §4.4.
"""

from __future__ import annotations

import time
from typing import Any

from llm_kg.pipeline.stage import PipelineContext, Stage


class IndexingPipeline:
    """Runs a list of `Stage`s in order, threading values through.

    Times each stage and reports via `ctx.logger.log_stage(name, duration)` if
    a logger is present.
    """

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
