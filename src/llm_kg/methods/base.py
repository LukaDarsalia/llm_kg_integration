"""The Method abstract: bundles a (IndexingPipeline, QueryPipeline) pair.

A Method picks which stage implementations go into each slot and how they're
parameterized. Day 1 ships zero concrete methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext


class MethodConfig(BaseModel):
    """Method-specific config block from YAML."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Method(ABC):
    """Assembles the indexing + query pipelines for one approach."""

    name: ClassVar[str] = "UnnamedMethod"

    @abstractmethod
    def build(
        self, cfg: MethodConfig, ctx: PipelineContext
    ) -> tuple[IndexingPipeline, QueryPipeline]:
        """Return the two pipelines fully wired for this method."""
