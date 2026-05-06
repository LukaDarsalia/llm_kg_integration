"""Pydantic schema for experiment YAML configs.

The schema is intentionally permissive about per-plugin `params` blobs (just a
free-form dict) — each plug-in validates its own params on construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Names a provider (LLM, embedding, vector store, etc.) by registry key."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    cache: bool = True


class StorageConfig(BaseModel):
    vector: ProviderConfig
    graph: ProviderConfig
    kv: ProviderConfig


class MethodConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class LoggerConfig(BaseModel):
    name: str = "wandb"
    project: str | None = None
    run_name: str | None = None


class RootConfig(BaseModel):
    method: MethodConfig
    llm: ProviderConfig
    embedder: ProviderConfig
    storage: StorageConfig
    dataset: DatasetConfig
    evaluator: EvaluatorConfig
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    cache_dir: Path = Path(".cache")
    seed: int = 0
