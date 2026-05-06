"""Weights & Biases logger.

Imports `wandb` lazily so tests that don't use it (and CI without WANDB_API_KEY)
don't pay the import cost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm_kg.evaluation import Prediction
from llm_kg.logging_.base import ExperimentLogger


class WandbLogger(ExperimentLogger):
    """W&B implementation of ExperimentLogger.

    Stage durations are accumulated and flushed at `finish()` as a histogram
    rather than logged per-call (per-call would spam the run).
    """

    def __init__(self, project: str, run_name: str | None = None) -> None:
        self._project = project
        self._run_name = run_name
        self._run: Any = None
        self._stage_totals: dict[str, float] = defaultdict(float)
        self._stage_counts: dict[str, int] = defaultdict(int)

    def init(self, config: dict[str, Any], run_name: str) -> None:
        import wandb

        self._run = wandb.init(
            project=self._project,
            name=self._run_name or run_name,
            config=config,
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        if step is None:
            self._run.log(metrics)
        else:
            self._run.log(metrics, step=step)

    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None:
        self._stage_totals[stage] += duration_s
        self._stage_counts[stage] += 1

    def log_predictions(self, predictions: list[Prediction]) -> None:
        if self._run is None:
            return
        import wandb

        rows = [
            [
                p.qid,
                p.question,
                p.answer,
                "; ".join(p.gold_list()),
                len(p.retrieved),
            ]
            for p in predictions
        ]
        table = wandb.Table(
            columns=["qid", "question", "answer", "gold", "n_retrieved"],
            data=rows,
        )
        self._run.log({"predictions": table})

    def log_artifact(self, name: str, payload: Any) -> None:
        if self._run is None:
            return
        self._run.log({name: payload})

    def finish(self) -> None:
        if self._run is None:
            return
        stage_metrics = {
            f"stage_time/{k}_total_s": v for k, v in self._stage_totals.items()
        }
        stage_metrics.update(
            {
                f"stage_time/{k}_avg_s": v / max(self._stage_counts[k], 1)
                for k, v in self._stage_totals.items()
            }
        )
        if stage_metrics:
            self._run.log(stage_metrics)
        self._run.finish()
        self._run = None
