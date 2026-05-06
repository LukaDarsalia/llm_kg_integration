"""LLM-as-judge abstraction (LightRAG-style head-to-head)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from llm_kg.providers.base import LLMProvider


class LLMJudge(ABC):
    """Abstract: judge a candidate vs baseline answer on multiple axes.

    Returns a dict mapping each axis to either "candidate" or "baseline" — the
    winner. Day 1 ships only a stub; a real prompt-based judge lands later.
    """

    @abstractmethod
    async def judge(
        self,
        question: str,
        candidate: str,
        baseline: str,
        axes: Iterable[str],
    ) -> dict[str, str]: ...


_JUDGE_PROMPT_STUB = """[STUB JUDGE PROMPT — replace before running real generative evals]

Question: {question}

Answer A (candidate): {candidate}
Answer B (baseline):  {baseline}

For each axis below, decide whether A or B is better and respond with only "A" or "B" on its own line.
Axes: {axes}
"""


class PromptedLLMJudge(LLMJudge):
    """A LightRAG-style judge backed by an LLMProvider.

    Day 1 the prompt is intentionally a placeholder; the real prompt and parsing
    land in the PR that adds the GenerativeEvaluator workflow.
    """

    def __init__(self, llm: LLMProvider, prompt_template: str = _JUDGE_PROMPT_STUB) -> None:
        self._llm = llm
        self._prompt = prompt_template

    async def judge(
        self,
        question: str,
        candidate: str,
        baseline: str,
        axes: Iterable[str],
    ) -> dict[str, str]:
        axes_list = list(axes)
        prompt = self._prompt.format(
            question=question, candidate=candidate, baseline=baseline, axes=", ".join(axes_list)
        )
        response = await self._llm.generate(prompt)
        # PLACEHOLDER parser — real version lands later.
        verdicts: dict[str, str] = {}
        lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
        for axis, line in zip(axes_list, lines, strict=False):
            verdicts[axis] = "candidate" if line.upper().startswith("A") else "baseline"
        for axis in axes_list:
            verdicts.setdefault(axis, "baseline")
        return verdicts
