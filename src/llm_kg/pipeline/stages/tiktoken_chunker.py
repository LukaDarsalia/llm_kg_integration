"""Token-based chunker using tiktoken — matches LightRAG's chunking exactly.

Defaults (chunk_size=1200 tokens, chunk_overlap=100 tokens, gpt-4o-mini
encoding) reproduce LightRAG's `chunking_by_token_size` from
`lightrag/operate.py`. Decode is done with `errors="ignore"` so a chunk
boundary that lands inside a multi-byte sequence yields safe output.
"""

from __future__ import annotations

import tiktoken

from llm_kg.data.types import Chunk, Document
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.chunker import Chunker


class TiktokenChunker(Chunker):
    """Token-window chunker. Token boundaries via tiktoken's `encoding_for_model`."""

    name = "TiktokenChunker"

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        model: str = "gpt-4o-mini",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self._enc = tiktoken.encoding_for_model(model)

    async def run(self, inp: list[Document], ctx: PipelineContext) -> list[Chunk]:
        out: list[Chunk] = []
        stride = self.chunk_size - self.chunk_overlap
        for doc in inp:
            if not doc.text:
                continue
            tokens = self._enc.encode(doc.text, disallowed_special=())
            if not tokens:
                continue
            position = 0
            i = 0
            while i < len(tokens):
                window = tokens[i : i + self.chunk_size]
                text = self._enc.decode(window, errors="ignore")
                out.append(
                    Chunk(
                        id=f"{doc.id}::chunk{position}",
                        text=text,
                        doc_id=doc.id,
                        position=position,
                        metadata=dict(doc.metadata),
                    )
                )
                position += 1
                if i + self.chunk_size >= len(tokens):
                    break
                i += stride
        return out
