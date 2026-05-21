"""Tests for the shared HippoRAG-style loader + HotpotQA + 2Wiki datasets."""

import json
from pathlib import Path

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data._hippo_loader import HippoMultiHopDataset, _doc_id
from llm_kg.data.hotpotqa import HotpotQADataset
from llm_kg.data.wiki2 import MultiHopWiki2Dataset


def _write_fake(tmp_path: Path, name: str) -> Path:
    ds_dir = tmp_path / name
    ds_dir.mkdir()
    queries = [
        {
            "_id": "q1",
            "question": "What is X?",
            "answer": "Foo",
            "supporting_facts": [["Title A", 0], ["Title A", 1], ["Title B", 0]],
            "context": [
                ["Title A", ["Sentence one.", " Sentence two."]],
                ["Title B", ["B sentence."]],
                ["Distractor", ["Random text."]],
            ],
            "type": "comparison",
            "level": "hard",
        },
        {
            "_id": "q2",
            "question": "Where is Y?",
            "answer": "Bar",
            "supporting_facts": [["Title C", 0]],
            "context": [
                ["Title C", ["C sentence."]],
                # Reuse Distractor exactly — corpus should dedupe.
                ["Distractor", ["Random text."]],
            ],
        },
    ]
    (ds_dir / f"{name}.json").write_text(json.dumps(queries))
    return ds_dir


def test_hotpotqa_registered() -> None:
    assert "hotpotqa" in DATASET_REGISTRY
    assert DATASET_REGISTRY.get("hotpotqa") is HotpotQADataset


def test_2wiki_registered() -> None:
    assert "2wikimultihopqa" in DATASET_REGISTRY
    assert DATASET_REGISTRY.get("2wikimultihopqa") is MultiHopWiki2Dataset


def test_loads_examples_and_corpus(tmp_path: Path) -> None:
    ds_dir = _write_fake(tmp_path, "hotpotqa")
    ds = HotpotQADataset(local_dir=str(ds_dir))

    examples = list(ds.examples())
    assert len(examples) == 2
    assert examples[0].id == "q1"
    assert examples[0].question == "What is X?"
    assert examples[0].answers == ["Foo"]
    assert examples[0].long_answer == "Foo"

    # Supporting passages: Title A + Title B (unique titles from supporting_facts).
    sup = examples[0].metadata["supporting"]
    assert len(sup) == 2
    expected_a = _doc_id("hotpotqa", "Title A", "Sentence one. Sentence two.")
    expected_b = _doc_id("hotpotqa", "Title B", "B sentence.")
    assert expected_a in sup
    assert expected_b in sup

    corpus = ds.corpus()
    docs = list(corpus.documents())
    # 4 unique (title, text) pairs: A, B, C, Distractor (deduped across queries)
    assert len(docs) == 4
    titles = sorted({d.metadata["title"] for d in docs})
    assert titles == ["Distractor", "Title A", "Title B", "Title C"]


def test_2wiki_uses_2wikimultihopqa_filename(tmp_path: Path) -> None:
    ds_dir = _write_fake(tmp_path, "2wikimultihopqa")
    ds = MultiHopWiki2Dataset(local_dir=str(ds_dir))
    assert len(list(ds.examples())) == 2


def test_n_queries_caps_records(tmp_path: Path) -> None:
    ds_dir = _write_fake(tmp_path, "hotpotqa")
    ds = HotpotQADataset(local_dir=str(ds_dir), n_queries=1)
    assert len(list(ds.examples())) == 1


def test_corpus_dedupes_across_queries(tmp_path: Path) -> None:
    ds_dir = _write_fake(tmp_path, "hotpotqa")
    ds = HotpotQADataset(local_dir=str(ds_dir))
    docs = list(ds.corpus().documents())
    distractors = [d for d in docs if d.metadata["title"] == "Distractor"]
    assert len(distractors) == 1


def test_supporting_ids_match_corpus_doc_ids(tmp_path: Path) -> None:
    """Critical for recall@k — supporting ids must match corpus doc ids."""
    ds_dir = _write_fake(tmp_path, "hotpotqa")
    ds = HotpotQADataset(local_dir=str(ds_dir))
    corpus_ids = {d.id for d in ds.corpus().documents()}
    for ex in ds.examples():
        sup = set(ex.metadata["supporting"])
        assert sup.issubset(corpus_ids), f"orphan supporting ids: {sup - corpus_ids}"


def test_doc_id_prefix_distinguishes_datasets(tmp_path: Path) -> None:
    """A passage shared in name+text between datasets should NOT collide in doc id."""
    ds_dir_hp = _write_fake(tmp_path, "hotpotqa")
    ds_dir_w2 = _write_fake(tmp_path, "2wikimultihopqa")
    hp_ids = {d.id for d in HotpotQADataset(local_dir=str(ds_dir_hp)).corpus().documents()}
    w2_ids = {d.id for d in MultiHopWiki2Dataset(local_dir=str(ds_dir_w2)).corpus().documents()}
    assert hp_ids.isdisjoint(w2_ids)


def test_abstract_requires_name() -> None:
    """Instantiating HippoMultiHopDataset directly (without setting name) raises."""
    import pytest

    with pytest.raises(ValueError, match="name"):
        HippoMultiHopDataset(local_dir=".")
