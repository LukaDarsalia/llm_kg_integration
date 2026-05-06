import pytest

from llm_kg.evaluation import Prediction
from llm_kg.logging_ import LOGGER_REGISTRY
from llm_kg.logging_.base import ExperimentLogger, NullLogger


def test_logger_is_abstract() -> None:
    with pytest.raises(TypeError):
        ExperimentLogger()  # type: ignore[abstract]


def test_logger_registry_has_null_and_wandb() -> None:
    assert "null" in LOGGER_REGISTRY
    assert "wandb" in LOGGER_REGISTRY


def test_null_logger_swallows_everything() -> None:
    log = NullLogger()
    log.init({"x": 1}, run_name="test")
    log.log_metrics({"em": 0.5})
    log.log_stage("Chunker", 0.01)
    log.log_predictions([Prediction(qid="q1", question="?", answer="a")])
    log.log_artifact("anything", {"k": "v"})
    log.finish()
