"""The Stage abstract base and the PipelineContext passed to every stage.

Stages are the unit of composition. Each stage takes one input value, has access
to all providers/storages/loggers via `PipelineContext`, and returns one output
value. Concrete stage families (Chunker, Retriever, …) further restrict
`InT`/`OutT` to specific data types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


@dataclass
class PipelineContext:
    """Carries everything a stage might need: providers, storage, logger, trace.

    `trace` is a free-form per-run dict where stages can record intermediate
    artifacts for debugging. Day-1 contract is loose — see the design doc's
    "Open questions" section.

    `query` carries the original question through the query pipeline so stages
    that don't see it directly (e.g. Generator, which receives only the assembled
    context) can still reference it. The runner sets it before each query.

    Storage and provider fields are typed as `Any` here to keep the import graph
    shallow; they are typed precisely on construction in the runner.
    """

    llm: Any  # LLMProvider | None
    embedder: Any  # EmbeddingProvider | None
    vector_store: Any  # VectorStore | None
    graph_store: Any  # GraphStore | None
    kv_store: Any  # KVStore | None
    logger: Any  # ExperimentLogger | None
    trace: dict[str, Any]
    query: str | None = None


class Stage(ABC, Generic[InT, OutT]):
    """One step of an indexing or query pipeline.

    Subclasses set a `name` (for logging) and implement `run`. They MAY also
    write side-effects to `ctx.vector_store`, `ctx.graph_store`, or
    `ctx.kv_store`; they SHOULD log timings via `ctx.logger.log_stage(...)`
    only if they do extra internal sub-steps (the pipeline already times the
    top-level call).
    """

    name: ClassVar[str] = "UnnamedStage"

    @abstractmethod
    async def run(self, inp: InT, ctx: PipelineContext) -> OutT: ...
