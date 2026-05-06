"""Plain dataclasses used throughout the pipeline.

These are intentionally minimal; methods may carry additional info via the
`metadata` dict rather than subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of corpus input (e.g., one Wikipedia page, one PDF, one note)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A contiguous piece of a Document, as produced by a Chunker."""

    id: str
    text: str
    doc_id: str
    position: int  # ordinal within the parent document, 0-indexed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    name: str
    type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    src: str
    dst: str
    predicate: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    """One question-answer record from a dataset.

    For extractive datasets, `answers` holds gold short-form answers (often
    multiple acceptable forms). For long-form datasets, `long_answer` holds the
    reference response and `answers` may be empty.
    """

    id: str
    question: str
    answers: list[str] = field(default_factory=list)
    long_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredHit:
    """A retriever's output: an item id with a score and arbitrary metadata."""

    id: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)
