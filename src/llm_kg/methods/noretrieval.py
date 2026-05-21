"""Closed-book baseline: the LLM answers from memory only, no retrieval.

Indexing is a no-op (the corpus is loaded but ignored). The query pipeline
calls the LLM with the question and an empty context — a memorization probe
for comparing against retrieval-based methods.
"""

from __future__ import annotations

from typing import ClassVar

from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method, MethodConfig
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.generator import DefaultGenerator

CLOSED_BOOK_PROMPT = """Answer the question. Respond with the shortest correct answer — \
typically one entity name or a short noun phrase. Do not explain, do not use markdown — \
output only the answer text.

Question: {question}

Context: {context}

Answer:"""


class _NoOpStage(Stage[list, list]):
    """Pass-through stage used to keep IndexingPipeline non-empty."""

    name = "NoOpIndexing"

    async def run(self, inp, ctx):
        return inp


class _EmptyContextStage(Stage[str, str]):
    """Query-side: take the raw question (str) and emit empty context (str)."""

    name = "EmptyContext"

    async def run(self, inp, ctx):
        return ""


@METHOD_REGISTRY.register("noretrieval")
class NoRetrieval(Method):
    """Closed-book LLM baseline. Provides the question with an empty context."""

    name: ClassVar[str] = "noretrieval"

    def build(
        self, cfg: MethodConfig, ctx: PipelineContext
    ) -> tuple[IndexingPipeline, QueryPipeline]:
        return (
            IndexingPipeline(stages=[_NoOpStage()]),
            QueryPipeline(stages=[
                _EmptyContextStage(),
                DefaultGenerator(prompt_template=CLOSED_BOOK_PROMPT),
            ]),
        )
