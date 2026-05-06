"""Embedder stage: vectorizes chunks (and optionally entities) and writes them to the vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.extractor import ExtractionResult


class Embedder(Stage[ExtractionResult, ExtractionResult], ABC):
    """Abstract: produce dense vectors and persist them in `ctx.vector_store`."""

    name = "Embedder"

    @abstractmethod
    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult: ...


class DefaultEmbedder(Embedder):
    """Embeds every chunk's text via `ctx.embedder` and upserts to `ctx.vector_store`.

    Does NOT embed entities — methods that need entity embeddings (HippoRAG,
    LightRAG) implement their own Embedder.
    """

    name = "DefaultEmbedder"

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        if not inp.chunks:
            return inp
        if ctx.embedder is None:
            raise RuntimeError("DefaultEmbedder requires ctx.embedder to be set")
        if ctx.vector_store is None:
            raise RuntimeError("DefaultEmbedder requires ctx.vector_store to be set")

        texts = [c.text for c in inp.chunks]
        ids = [c.id for c in inp.chunks]
        meta = [{"doc_id": c.doc_id, "position": c.position, **c.metadata} for c in inp.chunks]

        vectors = await ctx.embedder.embed(texts)
        ctx.vector_store.upsert(ids, vectors, meta)
        return inp
