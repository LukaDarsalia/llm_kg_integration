import pytest

from llm_kg.data.types import ScoredHit
from llm_kg.evaluation import EVALUATOR_REGISTRY, Prediction
from llm_kg.evaluation.evaluator import (
    Evaluator,
    ExtractiveEvaluator,
    GenerativeEvaluator,
)
from llm_kg.evaluation.judge import LLMJudge


def test_evaluator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]


def test_evaluator_registry_lists_extractive_and_generative() -> None:
    assert "extractive" in EVALUATOR_REGISTRY
    assert "generative" in EVALUATOR_REGISTRY


async def test_extractive_evaluator_aggregates_em_and_f1() -> None:
    preds = [
        Prediction(qid="q1", question="?", answer="Paris", retrieved=[], gold=["paris"]),
        Prediction(qid="q2", question="?", answer="London", retrieved=[], gold=["paris"]),
    ]
    ev = ExtractiveEvaluator()
    metrics = await ev.score(preds)
    assert metrics["em"] == 0.5
    assert 0.4 < metrics["f1"] <= 0.5
    assert metrics["n"] == 2


async def test_extractive_evaluator_recall_at_k() -> None:
    preds = [
        Prediction(
            qid="q1",
            question="?",
            answer="x",
            retrieved=[ScoredHit(id="c1", score=1, meta={}), ScoredHit(id="c3", score=0.5, meta={})],
            gold=["x"],
            relevant_ids=["c1"],
        ),
    ]
    ev = ExtractiveEvaluator(recall_ks=(1, 2))
    m = await ev.score(preds)
    assert m["recall@1"] == 1.0
    assert m["recall@2"] == 1.0


async def test_generative_evaluator_calls_judge_for_each_pair() -> None:
    """The shell judge always returns the configured baseline winner; we just verify wiring."""

    class StubJudge(LLMJudge):
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, question, candidate, baseline, axes):
            self.calls += 1
            return {axis: "candidate" for axis in axes}

    judge = StubJudge()
    ev = GenerativeEvaluator(
        judge=judge,
        baseline_predictions={
            "q1": "baseline answer 1",
            "q2": "baseline answer 2",
        },
        axes=("comprehensiveness", "overall"),
    )
    preds = [
        Prediction(qid="q1", question="?", answer="cand 1", retrieved=[], gold=None),
        Prediction(qid="q2", question="?", answer="cand 2", retrieved=[], gold=None),
    ]
    m = await ev.score(preds)
    assert judge.calls == 2
    assert m["winrate_comprehensiveness"] == 1.0
    assert m["winrate_overall"] == 1.0
    assert m["n"] == 2
