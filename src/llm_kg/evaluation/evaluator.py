"""Evaluator abstract + extractive (EM/F1/recall@k) and generative (LLM judge) impls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from llm_kg.data.types import ScoredHit
from llm_kg.evaluation.judge import LLMJudge
from llm_kg.evaluation.metrics import exact_match, f1_score, recall_at_k


@dataclass
class Prediction:
    """One model prediction with everything an evaluator might need to score it."""

    qid: str
    question: str
    answer: str
    retrieved: list[ScoredHit] = field(default_factory=list)
    gold: list[str] | str | None = None
    relevant_ids: list[str] | None = None  # for retrieval recall@k
    metadata: dict = field(default_factory=dict)

    def gold_list(self) -> list[str]:
        if self.gold is None:
            return []
        if isinstance(self.gold, str):
            return [self.gold]
        return list(self.gold)


class Evaluator(ABC):
    @abstractmethod
    async def score(self, predictions: list[Prediction]) -> dict[str, float]: ...


class ExtractiveEvaluator(Evaluator):
    """Scores short-answer QA: aggregate EM, F1, and recall@k for each k in `recall_ks`."""

    def __init__(self, recall_ks: tuple[int, ...] = (1, 5, 10)) -> None:
        self.recall_ks = recall_ks

    async def score(self, predictions: list[Prediction]) -> dict[str, float]:
        if not predictions:
            return {"n": 0.0}
        em_total = 0.0
        f1_total = 0.0
        recall_totals: dict[int, float] = {k: 0.0 for k in self.recall_ks}
        recall_counts: dict[int, int] = {k: 0 for k in self.recall_ks}
        n = len(predictions)
        for p in predictions:
            golds = p.gold_list()
            em_total += exact_match(p.answer, golds)
            f1_total += f1_score(p.answer, golds)
            if p.relevant_ids is not None:
                retrieved_ids = [h.id for h in p.retrieved]
                for k in self.recall_ks:
                    recall_totals[k] += recall_at_k(retrieved_ids, p.relevant_ids, k)
                    recall_counts[k] += 1
        out: dict[str, float] = {
            "em": em_total / n,
            "f1": f1_total / n,
            "n": float(n),
        }
        for k in self.recall_ks:
            if recall_counts[k]:
                out[f"recall@{k}"] = recall_totals[k] / recall_counts[k]
        return out


class GenerativeEvaluator(Evaluator):
    """LightRAG-style head-to-head: each prediction vs a baseline on `axes`.

    `baseline_predictions` maps qid → baseline answer string. Predictions whose
    qid is not in the baseline map are skipped.
    """

    def __init__(
        self,
        judge: LLMJudge,
        baseline_predictions: dict[str, str],
        axes: Iterable[str] = (
            "comprehensiveness",
            "diversity",
            "empowerment",
            "overall",
        ),
    ) -> None:
        self._judge = judge
        self._baseline = baseline_predictions
        self._axes = list(axes)

    async def score(self, predictions: list[Prediction]) -> dict[str, float]:
        wins = {axis: 0 for axis in self._axes}
        n = 0
        for p in predictions:
            baseline = self._baseline.get(p.qid)
            if baseline is None:
                continue
            verdicts = await self._judge.judge(p.question, p.answer, baseline, self._axes)
            for axis, winner in verdicts.items():
                if winner == "candidate":
                    wins[axis] += 1
            n += 1
        out: dict[str, float] = {"n": float(n)}
        if n:
            for axis in self._axes:
                out[f"winrate_{axis}"] = wins[axis] / n
        return out
