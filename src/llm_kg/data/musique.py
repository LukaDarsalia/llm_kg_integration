"""MuSiQue dataset loader (HippoRAG-compatible).

Loads the 1000-query MuSiQue-Ans subset that HippoRAG curates at
`reproduce/dataset/musique.json` — same set every other paper compares against.

Skips HuggingFace entirely; loads from a local path or downloads on first use
from the HippoRAG GitHub raw URL.

The corpus is built from the paragraphs embedded in each query record,
deduped by (title, paragraph_text). For the full 1000-query file this yields
exactly the 11,656 unique passages HippoRAG publishes as `musique_corpus.json`.

`metadata["supporting"]` on each QAExample carries the doc ids (same ids the
corpus produces) of the paragraphs marked `is_supporting=True`. Pass these
into a `Prediction.relevant_ids` field to compute recall@k.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample

_HIPPORAG_MUSIQUE_URL = (
    "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/musique.json"
)


def _doc_id(title: str, text: str) -> str:
    """Stable, deterministic doc id derived from the (title, text) pair."""
    h = hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()
    return f"musique-{h[:16]}"


@DATASET_REGISTRY.register("musique")
class MuSiQueDataset(QADataset):
    """MuSiQue-Ans (answerable) loaded from HippoRAG's pre-sampled 1000-query JSON."""

    def __init__(
        self,
        local_dir: str | Path | None = None,
        cache_dir: str | Path = ".cache/musique",
        n_queries: int | None = None,
        download: bool = True,
    ) -> None:
        cache_path = Path(cache_dir)
        if local_dir is not None:
            queries_path = Path(local_dir) / "musique.json"
        else:
            queries_path = cache_path / "musique.json"
            if not queries_path.exists():
                if not download:
                    raise FileNotFoundError(
                        f"{queries_path} not present and download=False"
                    )
                cache_path.mkdir(parents=True, exist_ok=True)
                resp = requests.get(_HIPPORAG_MUSIQUE_URL, timeout=60)
                resp.raise_for_status()
                queries_path.write_bytes(resp.content)

        self._records: list[dict[str, Any]] = json.loads(queries_path.read_text())
        if n_queries is not None:
            self._records = self._records[:n_queries]

        # Pre-build corpus dedup map: (title, text) → doc_id, ordered.
        self._corpus_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for rec in self._records:
            for p in rec.get("paragraphs", []):
                pair = (p["title"], p["paragraph_text"])
                if pair not in seen:
                    seen.add(pair)
                    self._corpus_pairs.append(pair)

    def examples(self) -> Iterable[QAExample]:
        for rec in self._records:
            golds = [rec["answer"]] + list(rec.get("answer_aliases", []))
            supporting_ids = [
                _doc_id(p["title"], p["paragraph_text"])
                for p in rec.get("paragraphs", [])
                if p.get("is_supporting")
            ]
            yield QAExample(
                id=rec["id"],
                question=rec["question"],
                answers=golds,
                long_answer=rec["answer"],
                metadata={
                    "supporting": supporting_ids,
                    "answerable": rec.get("answerable", True),
                    "decomposition": rec.get("question_decomposition", []),
                },
            )

    def corpus(self) -> Corpus:
        pairs = self._corpus_pairs

        class _C(Corpus):
            def documents(self) -> Iterable[Document]:
                for title, text in pairs:
                    yield Document(
                        id=_doc_id(title, text),
                        text=text,
                        metadata={"title": title},
                    )

            def __len__(self) -> int:
                return len(pairs)

        return _C()

    def __len__(self) -> int:
        return len(self._records)
