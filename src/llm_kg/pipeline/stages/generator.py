"""Generator stage: context (str) → answer (str). Reads the question from `ctx.query`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage


class Generator(Stage[str, str], ABC):
    """Abstract: produce the final answer given the assembled context.

    The question itself is read from `ctx.query` (set by the runner before each
    query), not threaded through the pipeline. This lets every intermediate
    stage stay focused on its own data type.
    """

    name = "Generator"

    @abstractmethod
    async def run(self, inp: str, ctx: PipelineContext) -> str: ...


_DEFAULT_PROMPT = """Answer the question using only the context below. If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class DefaultGenerator(Generator):
    """Default: stuff the context into a single prompt and call `ctx.llm.generate`."""

    name = "DefaultGenerator"

    def __init__(self, prompt_template: str = _DEFAULT_PROMPT) -> None:
        self.prompt_template = prompt_template

    async def run(self, inp: str, ctx: PipelineContext) -> str:
        if ctx.llm is None:
            raise RuntimeError("DefaultGenerator requires ctx.llm to be set")
        if ctx.query is None:
            raise RuntimeError(
                "DefaultGenerator requires ctx.query to be set "
                "(the runner should set it before query.run)"
            )
        prompt = self.prompt_template.format(context=inp, question=ctx.query)
        response = await ctx.llm.generate(prompt)
        return response.text
