"""ContextBuilder stage: ranked hits → a single context string for the LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext, Stage


class ContextBuilder(Stage[list[ScoredHit], str], ABC):
    """Abstract: format retrieved hits into the prompt-ready context."""

    name = "ContextBuilder"

    @abstractmethod
    async def run(self, inp: list[ScoredHit], ctx: PipelineContext) -> str: ...


class DefaultContextBuilder(ContextBuilder):
    """Default: pull each hit's text from `ctx.kv_store` and concatenate, highest-score first."""

    name = "DefaultContextBuilder"

    def __init__(self, separator: str = "\n\n---\n\n") -> None:
        self.separator = separator

    async def run(self, inp: list[ScoredHit], ctx: PipelineContext) -> str:
        if not inp:
            return ""
        if ctx.kv_store is None:
            raise RuntimeError("DefaultContextBuilder requires ctx.kv_store to be set")
        ordered = sorted(inp, key=lambda h: h.score, reverse=True)
        parts = [str(ctx.kv_store.get(h.id)) for h in ordered]
        return self.separator.join(parts)
