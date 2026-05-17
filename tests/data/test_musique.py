import json
from pathlib import Path

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.musique import MuSiQueDataset


def test_registered() -> None:
    assert "musique" in DATASET_REGISTRY
    assert DATASET_REGISTRY.get("musique") is MuSiQueDataset


def _write_fake_dataset(tmp_path: Path) -> Path:
    """Write a small fake HippoRAG-format JSON pair to tmp_path/dataset/."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()

    queries = [
        {
            "id": "2hop__1",
            "question": "Where was the author of X born?",
            "answer": "Paris",
            "answer_aliases": ["paris france"],
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "Book X", "paragraph_text": "X was written by A.", "is_supporting": True},
                {"idx": 1, "title": "Author A", "paragraph_text": "A was born in Paris.", "is_supporting": True},
                {"idx": 2, "title": "Distractor", "paragraph_text": "Unrelated stuff.", "is_supporting": False},
            ],
            "question_decomposition": [],
        },
        {
            "id": "2hop__2",
            "question": "What country borders Y?",
            "answer": "Germany",
            "answer_aliases": [],
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "Country Y", "paragraph_text": "Y borders Germany.", "is_supporting": True},
                # Reuse the same (title, text) as in q1 — corpus dedup should collapse.
                {"idx": 1, "title": "Distractor", "paragraph_text": "Unrelated stuff.", "is_supporting": False},
            ],
            "question_decomposition": [],
        },
    ]
    (ds_dir / "musique.json").write_text(json.dumps(queries))
    return ds_dir


def test_loads_queries_and_corpus_from_local_paths(tmp_path: Path) -> None:
    ds_dir = _write_fake_dataset(tmp_path)
    ds = MuSiQueDataset(local_dir=str(ds_dir))

    examples = list(ds.examples())
    assert len(examples) == 2
    assert examples[0].id == "2hop__1"
    assert examples[0].question.startswith("Where")
    # Gold = answer + aliases
    assert examples[0].answers == ["Paris", "paris france"]
    # Supporting paragraphs are stashed in metadata for recall@k
    assert len(examples[0].metadata["supporting"]) == 2

    corpus = ds.corpus()
    docs = list(corpus.documents())
    # 3 unique paragraphs across the two queries (the "Distractor" appears twice but dedupes)
    assert len(docs) == 4  # X, A, Distractor, Y — wait, 4 paragraphs minus 1 dup = 4
    titles = sorted({d.metadata["title"] for d in docs})
    assert titles == ["Author A", "Book X", "Country Y", "Distractor"]


def test_n_queries_param_caps_examples(tmp_path: Path) -> None:
    ds_dir = _write_fake_dataset(tmp_path)
    ds = MuSiQueDataset(local_dir=str(ds_dir), n_queries=1)
    examples = list(ds.examples())
    assert len(examples) == 1


def test_corpus_dedupes_by_title_and_text(tmp_path: Path) -> None:
    ds_dir = _write_fake_dataset(tmp_path)
    ds = MuSiQueDataset(local_dir=str(ds_dir))
    docs = list(ds.corpus().documents())

    # The "Distractor" / "Unrelated stuff." pair appears in both queries — should appear once.
    distractors = [d for d in docs if d.metadata["title"] == "Distractor"]
    assert len(distractors) == 1


def test_doc_id_is_stable_hash(tmp_path: Path) -> None:
    """Same (title, text) → same doc id across instances; deterministic."""
    ds_dir = _write_fake_dataset(tmp_path)
    docs_a = list(MuSiQueDataset(local_dir=str(ds_dir)).corpus().documents())
    docs_b = list(MuSiQueDataset(local_dir=str(ds_dir)).corpus().documents())
    assert [d.id for d in docs_a] == [d.id for d in docs_b]


def test_supporting_metadata_uses_corpus_doc_ids(tmp_path: Path) -> None:
    """metadata['supporting'] should reference the same doc ids the corpus produces,
    so recall@k can compare them directly."""
    ds_dir = _write_fake_dataset(tmp_path)
    ds = MuSiQueDataset(local_dir=str(ds_dir))
    corpus_ids = {d.id for d in ds.corpus().documents()}
    for ex in ds.examples():
        supporting_ids = set(ex.metadata["supporting"])
        assert supporting_ids.issubset(corpus_ids)


def test_long_answer_is_canonical_answer(tmp_path: Path) -> None:
    ds_dir = _write_fake_dataset(tmp_path)
    ex = next(iter(MuSiQueDataset(local_dir=str(ds_dir)).examples()))
    assert ex.long_answer == "Paris"  # canonical, no aliases


def test_len(tmp_path: Path) -> None:
    ds_dir = _write_fake_dataset(tmp_path)
    ds = MuSiQueDataset(local_dir=str(ds_dir))
    assert len(ds) == 2
