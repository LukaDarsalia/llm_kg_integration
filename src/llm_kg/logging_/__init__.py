"""Experiment logging (W&B + null)."""

from llm_kg.logging_.base import ExperimentLogger, NullLogger
from llm_kg.logging_.wandb_logger import WandbLogger
from llm_kg.registry import Registry

LOGGER_REGISTRY: Registry[ExperimentLogger] = Registry("logger")
LOGGER_REGISTRY.register("null")(NullLogger)
LOGGER_REGISTRY.register("wandb")(WandbLogger)

__all__ = ["LOGGER_REGISTRY", "ExperimentLogger", "NullLogger", "WandbLogger"]
