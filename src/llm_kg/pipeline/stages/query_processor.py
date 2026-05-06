"""QueryProcessor stage: raw question string → structured ProcessedQuery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_kg.pipeline.stage import PipelineContext, Stage


@dataclass
class ProcessedQuery:
    """A query enriched with extracted keywords / entities for graph retrieval."""

    text: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


class QueryProcessor(Stage[str, ProcessedQuery], ABC):
    """Abstract: turn a raw question into a ProcessedQuery."""

    name = "QueryProcessor"

    @abstractmethod
    async def run(self, inp: str, ctx: PipelineContext) -> ProcessedQuery: ...


class IdentityQueryProcessor(QueryProcessor):
    """Default: wrap the question with no keyword/entity extraction.

    Naive RAG uses this; HippoRAG and LightRAG implement their own.
    """

    name = "IdentityQueryProcessor"

    async def run(self, inp: str, ctx: PipelineContext) -> ProcessedQuery:
        return ProcessedQuery(text=inp)
