"""CustomRAG retriever — TEMPLATE. Copy this folder to start a new GraphRAG method.

Implement ``initialize`` (reload the index from ``self.working_dir``) and ``answer``
(return ``(generated_answer, retrieved_context_chunks)``). See
``src/methods/lightrag/data_retriever.py`` for a worked reference, and CONTRIBUTING.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.pipeline.shared.contracts import BaseRetriever


class CustomRAGRetriever(BaseRetriever):
    async def initialize(self) -> None:
        raise NotImplementedError(
            "CustomRAG is a template. Implement initialize(): reload your index from "
            "self.working_dir (downloaded from S3) into memory."
        )

    async def answer(self, question: str, query_params: Dict[str, Any]) -> Tuple[str, List[str]]:
        raise NotImplementedError(
            "CustomRAG is a template. Implement answer(): return (generated_answer, "
            "list_of_retrieved_context_chunks) for the given question."
        )
