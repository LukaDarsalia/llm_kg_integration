"""LightRAG indexer: build a knowledge-graph index for one corpus and persist it.

Everything is written under ``working_dir`` (kv_store_*.json, vdb_*.json,
graph_chunk_entity_relation.graphml). The indexer stage uploads that directory to S3 as
the index artifact; the retriever later reloads it verbatim.
"""

from __future__ import annotations

from typing import List

from lightrag import LightRAG

from src.pipeline.shared.contracts import BaseIndexer, CorpusDoc
from .models import build_lightrag_embedding_func, build_lightrag_llm_func


class LightRAGIndexer(BaseIndexer):
    async def _make_rag(self) -> LightRAG:
        index_cfg = self.params.get("index", {}) or {}
        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=build_lightrag_llm_func(self.providers["llm"]),
            embedding_func=build_lightrag_embedding_func(self.providers["embedding"]),
            chunk_token_size=int(index_cfg.get("chunk_token_size", 1200)),
            chunk_overlap_token_size=int(index_cfg.get("chunk_overlap_token_size", 100)),
        )
        # Required before insert; in 1.5.x this also initializes the pipeline status.
        await rag.initialize_storages()
        return rag

    async def build(self, corpus: List[CorpusDoc]) -> None:
        self._rag = await self._make_rag()
        contexts = [doc.context for doc in corpus]
        ids = [f"{doc.corpus_name}#{i}" for i, doc in enumerate(corpus)]
        file_paths = [doc.corpus_name for doc in corpus]
        await self._rag.ainsert(contexts, ids=ids, file_paths=file_paths)

    async def close(self) -> None:
        rag = getattr(self, "_rag", None)
        if rag is not None:
            # Flush all storages to disk so the working_dir is a complete, reloadable index.
            await rag.finalize_storages()
