"""LightRAGExtractor — per-chunk entity + relation extraction via LLM.

Mirrors `extract_entities` in upstream `lightrag/operate.py` but stripped of
LightRAG's internal storage layer. One LLM call per chunk (the extraction
prompt), optionally followed by `max_gleaning` continue-extraction calls that
catch missed / malformed records.

**Cost note:** the upstream default is `max_gleaning=1`. We default to **0** to
keep budget safe — every gleaning pass roughly doubles the indexing spend. Set
`max_gleaning=1` in YAML to match upstream behavior exactly.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from llm_kg.data.types import Chunk, Entity, Relation
from llm_kg.methods.lightrag.parser import parse_extraction
from llm_kg.methods.lightrag.prompts import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_LANGUAGE,
    PROMPTS,
)
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.extractor import ExtractionResult, InformationExtractor


class LightRAGExtractor(InformationExtractor):
    name: ClassVar[str] = "LightRAGExtractor"

    def __init__(
        self,
        entity_types: list[str] | None = None,
        language: str = DEFAULT_LANGUAGE,
        max_gleaning: int = 0,
        max_concurrent: int = 16,
    ) -> None:
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.language = language
        self.max_gleaning = max_gleaning
        self.max_concurrent = max_concurrent

    async def run(self, inp: list[Chunk], ctx: PipelineContext) -> ExtractionResult:
        if not inp:
            return ExtractionResult(chunks=[], entities=[], relations=[])
        if ctx.llm is None:
            raise RuntimeError("LightRAGExtractor requires ctx.llm to be set")

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _extract_one(chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
            async with sem:
                return await self._extract_chunk(chunk, ctx)

        results = await asyncio.gather(*[_extract_one(c) for c in inp])

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        for ents, rels in results:
            all_entities.extend(ents)
            all_relations.extend(rels)

        return ExtractionResult(chunks=inp, entities=all_entities, relations=all_relations)

    async def _extract_chunk(
        self, chunk: Chunk, ctx: PipelineContext
    ) -> tuple[list[Entity], list[Relation]]:
        td = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        cd = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        entity_types_str = ", ".join(self.entity_types)
        examples_str = "\n".join(
            ex.format(tuple_delimiter=td, completion_delimiter=cd)
            for ex in PROMPTS["entity_extraction_examples"]
        )

        system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
            entity_types=entity_types_str,
            tuple_delimiter=td,
            completion_delimiter=cd,
            language=self.language,
            examples=examples_str,
        )
        user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            entity_types=entity_types_str,
            tuple_delimiter=td,
            completion_delimiter=cd,
            language=self.language,
            input_text=chunk.text,
        )
        # Single combined prompt — providers that don't carry a `system` field
        # still get the same content. Concatenating preserves cache hits across
        # providers that wrap a single-prompt LLMProvider.
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await ctx.llm.generate(prompt)
        raw = response.text

        # Optional gleaning passes
        for _ in range(self.max_gleaning):
            gleaning_user = PROMPTS["entity_continue_extraction_user_prompt"].format(
                tuple_delimiter=td,
                completion_delimiter=cd,
                language=self.language,
            )
            gleaning_prompt = (
                f"{system_prompt}\n\n{user_prompt}\n\n---Previous Extraction---\n{raw}\n\n"
                f"{gleaning_user}"
            )
            extra_resp = await ctx.llm.generate(gleaning_prompt)
            raw = f"{raw}\n{extra_resp.text}"

        parsed = parse_extraction(raw, tuple_delimiter=td, completion_delimiter=cd)

        # Build framework entities/relations, tagging each with the source chunk id.
        entities = [
            Entity(
                id=pe.name,                      # name IS the id in LightRAG
                name=pe.name,
                type=pe.type,
                metadata={"description": pe.description, "source_chunks": [chunk.id]},
            )
            for pe in parsed.entities
        ]
        relations = [
            Relation(
                src=pr.src,
                dst=pr.dst,
                predicate=", ".join(pr.keywords) if pr.keywords else "",
                metadata={
                    "description": pr.description,
                    "keywords": pr.keywords,
                    "weight": pr.weight,
                    "source_chunks": [chunk.id],
                },
            )
            for pr in parsed.relations
        ]
        return entities, relations
