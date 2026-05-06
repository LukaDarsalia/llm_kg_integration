"""GraphBuilder stage: writes entities/relations from an ExtractionResult into the graph store."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.extractor import ExtractionResult


class GraphBuilder(Stage[ExtractionResult, ExtractionResult], ABC):
    """Abstract: persist the extracted graph into `ctx.graph_store`."""

    name = "GraphBuilder"

    @abstractmethod
    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult: ...


class NoOpGraphBuilder(GraphBuilder):
    """Default for methods that don't construct a graph."""

    name = "NoOpGraphBuilder"

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        return inp
