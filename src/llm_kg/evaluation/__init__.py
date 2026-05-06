"""Evaluation: metrics, judges, evaluators."""

from llm_kg.evaluation.evaluator import (
    Evaluator,
    ExtractiveEvaluator,
    GenerativeEvaluator,
    Prediction,
)
from llm_kg.registry import Registry

EVALUATOR_REGISTRY: Registry[Evaluator] = Registry("evaluator")
EVALUATOR_REGISTRY.register("extractive")(ExtractiveEvaluator)
EVALUATOR_REGISTRY.register("generative")(GenerativeEvaluator)

__all__ = [
    "EVALUATOR_REGISTRY",
    "Evaluator",
    "ExtractiveEvaluator",
    "GenerativeEvaluator",
    "Prediction",
]
