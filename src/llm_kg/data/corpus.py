"""The Corpus abstract: a stream of Documents to index."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from llm_kg.data.types import Document


class Corpus(ABC):
    """Abstract: a (potentially large) collection of Documents.

    Returned as an Iterable so streaming corpora don't need to fit in memory.
    Implementations that can cheaply provide a count should override `__len__`.
    """

    @abstractmethod
    def documents(self) -> Iterable[Document]: ...

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} does not support len()")
