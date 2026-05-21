from collections.abc import Iterable

import pytest

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample


def test_corpus_is_abstract() -> None:
    with pytest.raises(TypeError):
        Corpus()  # type: ignore[abstract]


def test_qa_dataset_is_abstract() -> None:
    with pytest.raises(TypeError):
        QADataset()  # type: ignore[abstract]


def test_dataset_registry_constructed() -> None:
    assert "hotpotqa" in DATASET_REGISTRY
    assert "musique" in DATASET_REGISTRY
    assert "2wikimultihopqa" in DATASET_REGISTRY


def test_concrete_corpus_iterable() -> None:
    class InMemCorpus(Corpus):
        def __init__(self, docs: list[Document]) -> None:
            self._docs = docs

        def documents(self) -> Iterable[Document]:
            return iter(self._docs)

        def __len__(self) -> int:
            return len(self._docs)

    c = InMemCorpus([Document(id="a", text="x"), Document(id="b", text="y")])
    docs = list(c.documents())
    assert len(c) == 2
    assert [d.id for d in docs] == ["a", "b"]


def test_concrete_qa_dataset() -> None:
    class InMemQA(QADataset):
        def __init__(self, examples: list[QAExample]) -> None:
            self._ex = examples

        def examples(self) -> Iterable[QAExample]:
            return iter(self._ex)

        def __len__(self) -> int:
            return len(self._ex)

    ds = InMemQA([QAExample(id="q1", question="?", answers=["a"])])
    assert len(ds) == 1
    assert next(iter(ds.examples())).question == "?"
