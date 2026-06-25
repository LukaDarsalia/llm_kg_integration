"""CustomRAG indexer — TEMPLATE. Copy this folder to start a new GraphRAG method.

Implement ``build`` to index ``corpus`` into ``self.working_dir`` and persist it there so
the retriever can reload it in a separate process. See
``src/methods/lightrag/data_indexer.py`` for a worked reference, and CONTRIBUTING.md.
"""

from __future__ import annotations

from typing import List

from src.pipeline.shared.contracts import BaseIndexer, CorpusDoc


class CustomRAGIndexer(BaseIndexer):
    async def build(self, corpus: List[CorpusDoc]) -> None:
        raise NotImplementedError(
            "CustomRAG is a template. Implement build(): index `corpus` (a list of "
            "CorpusDoc) into self.working_dir and persist it. You have access to "
            "self.providers (llm + embedding config) and self.params (your method config)."
        )
