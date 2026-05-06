"""Retriever stage: ProcessedQuery → ranked list of ScoredHit.

There is no default retriever — every method must implement its own (this is
the most distinguishing stage between methods).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.query_processor import ProcessedQuery


class Retriever(Stage[ProcessedQuery, list[ScoredHit]], ABC):
    """Abstract: return ranked chunk hits for a processed query."""

    name = "Retriever"

    @abstractmethod
    async def run(self, inp: ProcessedQuery, ctx: PipelineContext) -> list[ScoredHit]: ...
