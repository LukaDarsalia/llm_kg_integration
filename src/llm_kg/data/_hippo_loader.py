"""Shared loader for HippoRAG-format multi-hop QA datasets.

HippoRAG publishes pre-sampled 1000-query JSONs for MuSiQue, HotpotQA, and
2WikiMultiHopQA in their `reproduce/dataset/` directory. MuSiQue uses a
`paragraphs` field with a different shape; HotpotQA and 2Wiki share a common
`context: list[[title, list[sentences]]]` shape with `supporting_facts` as
sentence-level `[title, sent_idx]` pairs.

This module provides the shared loader/corpus builder for the HotpotQA/2Wiki
shape. MuSiQue has its own loader in `musique.py`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample

_HIPPO_BASE = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"


def _doc_id(prefix: str, title: str, text: str) -> str:
    h = hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()
    return f"{prefix}-{h[:16]}"


def _paragraph_text(sentences: list[str]) -> str:
    """HotpotQA / 2Wiki sentences carry their own leading whitespace."""
    return "".join(sentences)


class HippoMultiHopDataset(QADataset):
    """Loads any HippoRAG-format multi-hop QA dataset with the HotpotQA shape.

    Subclasses set `name` (dataset slug used in the HippoRAG repo and as the
    doc-id prefix). On first construction the file is downloaded from
    `_HIPPO_BASE/{name}.json` and cached locally.
    """

    name: str = ""  # override in subclass

    def __init__(
        self,
        local_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        n_queries: int | None = None,
        download: bool = True,
    ) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set `name`")
        cache_path = Path(cache_dir or f".cache/{self.name}")
        if local_dir is not None:
            queries_path = Path(local_dir) / f"{self.name}.json"
        else:
            queries_path = cache_path / f"{self.name}.json"
            if not queries_path.exists():
                if not download:
                    raise FileNotFoundError(f"{queries_path} not present and download=False")
                cache_path.mkdir(parents=True, exist_ok=True)
                url = f"{_HIPPO_BASE}/{self.name}.json"
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                queries_path.write_bytes(resp.content)

        self._records: list[dict[str, Any]] = json.loads(queries_path.read_text())
        if n_queries is not None:
            self._records = self._records[:n_queries]

        # Build corpus dedup map: (title, text) → doc_id, ordered.
        self._corpus_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for rec in self._records:
            for ctx_item in rec.get("context", []):
                title = ctx_item[0]
                text = _paragraph_text(ctx_item[1])
                pair = (title, text)
                if pair not in seen:
                    seen.add(pair)
                    self._corpus_pairs.append(pair)

    def examples(self) -> Iterable[QAExample]:
        for rec in self._records:
            supporting_titles = {sf[0] for sf in rec.get("supporting_facts", [])}
            supporting_doc_ids: list[str] = []
            for ctx_item in rec.get("context", []):
                title = ctx_item[0]
                if title in supporting_titles:
                    text = _paragraph_text(ctx_item[1])
                    supporting_doc_ids.append(_doc_id(self.name, title, text))

            answer = rec["answer"]
            # HotpotQA / 2Wiki don't ship `answer_aliases` like MuSiQue does.
            golds = [answer]

            yield QAExample(
                id=rec["_id"],
                question=rec["question"],
                answers=golds,
                long_answer=answer,
                metadata={
                    "supporting": supporting_doc_ids,
                    "type": rec.get("type"),
                    "level": rec.get("level"),
                },
            )

    def corpus(self) -> Corpus:
        pairs = self._corpus_pairs
        prefix = self.name

        class _C(Corpus):
            def documents(self) -> Iterable[Document]:
                for title, text in pairs:
                    yield Document(
                        id=_doc_id(prefix, title, text),
                        text=text,
                        metadata={"title": title},
                    )

            def __len__(self) -> int:
                return len(pairs)

        return _C()

    def __len__(self) -> int:
        return len(self._records)
