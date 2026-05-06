"""The QADataset abstract: a corpus + paired QAExamples for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from llm_kg.data.corpus import Corpus
from llm_kg.data.types import QAExample


class QADataset(ABC):
    """Abstract: question-answer dataset paired with its source corpus.

    Default `corpus()` raises — datasets that have an associated corpus should
    override it. Datasets that work against a separately-loaded corpus (e.g.
    open-domain HotpotQA) can leave it unimplemented.
    """

    @abstractmethod
    def examples(self) -> Iterable[QAExample]: ...

    def corpus(self) -> Corpus:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a corpus; "
            "load one separately and pass it to the runner"
        )

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} does not support len()")
