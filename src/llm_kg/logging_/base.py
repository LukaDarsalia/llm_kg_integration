"""Abstract experiment logger and a no-op implementation for tests/offline runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_kg.evaluation import Prediction


class ExperimentLogger(ABC):
    """Abstract: tracks one experiment run (config, per-stage timing, predictions, metrics)."""

    @abstractmethod
    def init(self, config: dict[str, Any], run_name: str) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None: ...

    @abstractmethod
    def log_predictions(self, predictions: list[Prediction]) -> None: ...

    @abstractmethod
    def log_artifact(self, name: str, payload: Any) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class NullLogger(ExperimentLogger):
    """Swallows every call. Used by tests and `--no-log` runs."""

    def init(self, config: dict[str, Any], run_name: str) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None: ...
    def log_predictions(self, predictions: list[Prediction]) -> None: ...
    def log_artifact(self, name: str, payload: Any) -> None: ...
    def finish(self) -> None: ...
