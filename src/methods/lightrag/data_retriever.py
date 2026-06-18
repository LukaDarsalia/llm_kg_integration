"""LightRAG retriever: reload a persisted index and answer questions.

Runs in the evaluator stage, in a fresh process, after the index has been downloaded
from S3. It must reconstruct LightRAG with the SAME embedding config used at index time
(``initialize_storages`` reloads the vectors/graph from disk — no re-insert).

For each question it returns both the generated answer (``aquery``) and the list of
retrieved chunk texts (``aquery_data`` -> data.chunks[*].content), which together form
the GraphRAG-Bench unified record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lightrag import LightRAG, QueryParam

from src.pipeline.shared.contracts import BaseRetriever
from .models import build_lightrag_embedding_func, build_lightrag_llm_func


class LightRAGRetriever(BaseRetriever):
    async def initialize(self) -> None:
        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=build_lightrag_llm_func(self.providers["llm"]),
            embedding_func=build_lightrag_embedding_func(self.providers["embedding"]),
        )
        await self._rag.initialize_storages()  # reloads the index from disk

    def _query_param(self, query_params: Dict[str, Any]) -> QueryParam:
        merged = {**(self.params.get("query") or {}), **(query_params or {})}
        kwargs: Dict[str, Any] = {"mode": merged.get("mode", "hybrid")}
        if merged.get("top_k") is not None:
            kwargs["top_k"] = int(merged["top_k"])
        if merged.get("chunk_top_k") is not None:
            kwargs["chunk_top_k"] = int(merged["chunk_top_k"])
        return QueryParam(**kwargs)

    async def answer(self, question: str, query_params: Dict[str, Any]) -> Tuple[str, List[str]]:
        generated = await self._rag.aquery(question, param=self._query_param(query_params))
        data = await self._rag.aquery_data(question, param=self._query_param(query_params))

        contexts: List[str] = []
        if isinstance(data, dict):
            chunks = (data.get("data") or {}).get("chunks") or []
            contexts = [c.get("content", "") for c in chunks if isinstance(c, dict)]

        answer = generated if isinstance(generated, str) else str(generated)
        return answer, contexts

    async def close(self) -> None:
        rag = getattr(self, "_rag", None)
        if rag is not None:
            await rag.finalize_storages()
