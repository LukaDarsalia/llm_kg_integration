"""Chunker stage: splits a list of Documents into a flat list of Chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import Chunk, Document
from llm_kg.pipeline.stage import PipelineContext, Stage


class Chunker(Stage[list[Document], list[Chunk]], ABC):
    """Abstract: split documents into chunks."""

    name = "Chunker"

    @abstractmethod
    async def run(self, inp: list[Document], ctx: PipelineContext) -> list[Chunk]: ...


class DefaultChunker(Chunker):
    """Word-window chunker. Splits each document by whitespace, then groups
    `chunk_size` words at a time with `chunk_overlap` words of overlap.

    Cheap and language-agnostic. Methods that need sentence- or token-aware
    splitting should implement their own Chunker.
    """

    name = "DefaultChunker"

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def run(self, inp: list[Document], ctx: PipelineContext) -> list[Chunk]:
        out: list[Chunk] = []
        stride = self.chunk_size - self.chunk_overlap
        for doc in inp:
            words = doc.text.split()
            if not words:
                continue
            position = 0
            i = 0
            while i < len(words):
                window = words[i : i + self.chunk_size]
                out.append(
                    Chunk(
                        id=f"{doc.id}::chunk{position}",
                        text=" ".join(window),
                        doc_id=doc.id,
                        position=position,
                    )
                )
                position += 1
                if i + self.chunk_size >= len(words):
                    break
                i += stride
        return out
