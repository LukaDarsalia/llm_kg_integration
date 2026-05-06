"""InformationExtractor stage: chunks → (entities, relations)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_kg.data.types import Chunk, Entity, Relation
from llm_kg.pipeline.stage import PipelineContext, Stage


@dataclass
class ExtractionResult:
    """The artifact passed from extractor → graph_builder → embedder.

    `chunks` is always preserved end-to-end; `entities` / `relations` are added by
    the extractor. Methods may attach extra fields via `meta`.
    """

    chunks: list[Chunk]
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


class InformationExtractor(Stage[list[Chunk], ExtractionResult], ABC):
    """Abstract: pull entities and relations out of chunks."""

    name = "InformationExtractor"

    @abstractmethod
    async def run(self, inp: list[Chunk], ctx: PipelineContext) -> ExtractionResult: ...


class NoOpExtractor(InformationExtractor):
    """Default for methods that don't build a knowledge graph (e.g. naive RAG)."""

    name = "NoOpExtractor"

    async def run(self, inp: list[Chunk], ctx: PipelineContext) -> ExtractionResult:
        return ExtractionResult(chunks=inp, entities=[], relations=[])
