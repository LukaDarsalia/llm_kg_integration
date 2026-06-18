"""Shared data contracts for the GraphRAG pipeline.

These types and abstract base classes are the *contract* every stage and every method
agrees on. Read this file (and the per-stage READMEs) before adding a loader, a method,
or an evaluator.

The two key contracts:

* The **dataset contract** — what a loader must produce: a `corpus.parquet`
  (``corpus_name, context``) and a `qa.parquet` (``id, source, question, answer,
  question_type, evidence``) per subset. Enforced by
  ``src/pipeline/loader/registry.py::validate_dataset_output``.

* The **method contract** — what a GraphRAG method must implement: a
  :class:`BaseIndexer` (build + persist an index for one corpus) and a
  :class:`BaseRetriever` (reload that index and answer questions). The retriever's
  output, joined with the gold QA fields, becomes a :class:`Prediction`, which matches
  the GraphRAG-Bench unified record the evaluator scores.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

# --- dataset contract: column requirements ---------------------------------------------
CORPUS_REQUIRED_COLUMNS = ["corpus_name", "context"]
QA_REQUIRED_COLUMNS = ["id", "source", "question", "answer", "question_type", "evidence"]

# --- prediction contract: the GraphRAG-Bench unified record ----------------------------
# Exactly the fields Evaluation/generation_eval.py + retrieval_eval.py consume.
PREDICTION_FIELDS = [
    "id",
    "question",
    "source",
    "context",
    "evidence",
    "question_type",
    "generated_answer",
    "ground_truth",
]


def corpus_dir_name(corpus_name: str) -> str:
    """Filesystem-safe directory name for one corpus's index.

    The indexer writes each corpus to ``<index>/<subset>/<corpus_dir_name(name)>`` and the
    evaluator maps a question's ``source`` back to the same dir — so this mapping MUST be
    identical on both sides. Keep it deterministic.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(corpus_name))


@dataclass
class CorpusDoc:
    """A single document to be indexed. One corpus (``corpus_name``) may have >= 1 doc."""

    corpus_name: str
    context: str


@dataclass
class Prediction:
    """A method's answer for one question, in the GraphRAG-Bench unified schema.

    ``context`` is the list of retrieved text chunks (what retrieval/faithfulness score);
    ``generated_answer`` is the method's answer; ``ground_truth`` is the gold answer.
    """

    id: str
    question: str
    source: str
    context: List[str]
    evidence: str
    question_type: str
    generated_answer: str
    ground_truth: str

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


def validate_predictions(records: List[Dict[str, Any]]) -> None:
    """Fail loudly if any prediction record is missing a required field."""
    for i, rec in enumerate(records):
        missing = [f for f in PREDICTION_FIELDS if f not in rec]
        if missing:
            raise ValueError(
                f"Prediction record #{i} (id={rec.get('id')}) missing fields {missing}. "
                f"Required: {PREDICTION_FIELDS}"
            )
        if not isinstance(rec["context"], list):
            raise ValueError(
                f"Prediction record #{i} (id={rec.get('id')}): `context` must be a "
                f"List[str], got {type(rec['context']).__name__}."
            )


class BaseIndexer(ABC):
    """Builds and persists a method's index for ONE corpus into ``working_dir``.

    The indexer stage instantiates one indexer per ``corpus_name`` with a dedicated
    ``working_dir`` (so each corpus is an independently reloadable index). Everything
    written under ``working_dir`` is uploaded to S3 and tracked as the index artifact.
    """

    def __init__(self, working_dir: str, providers: Dict[str, Any], params: Dict[str, Any]):
        self.working_dir = working_dir
        self.providers = providers  # resolved {"llm": {...}, "embedding": {...}} from providers.yaml
        self.params = params  # method param dict (e.g. lightrag_params.yaml)

    @abstractmethod
    async def build(self, corpus: List[CorpusDoc]) -> None:
        """Index the given documents and flush the index to ``working_dir``."""

    async def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources / final flush. Override if needed."""
        return None


class BaseRetriever(ABC):
    """Reloads a persisted index from ``working_dir`` and answers questions.

    Runs in a *separate* process from the indexer (the evaluator downloads the index
    artifact from S3 first), so it must reconstruct everything from disk + config — never
    assume in-memory state from indexing. The embedding configuration MUST match the one
    used at index time, or the vector space will be inconsistent.
    """

    def __init__(self, working_dir: str, providers: Dict[str, Any], params: Dict[str, Any]):
        self.working_dir = working_dir
        self.providers = providers
        self.params = params

    @abstractmethod
    async def initialize(self) -> None:
        """Load the index from ``working_dir`` into memory (no re-indexing)."""

    @abstractmethod
    async def answer(self, question: str, query_params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Return ``(generated_answer, retrieved_context_chunks)`` for one question."""

    async def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources. Override if needed."""
        return None
