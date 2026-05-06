from llm_kg.data.types import Chunk, Document, Entity, QAExample, Relation, ScoredHit


def test_document_minimal() -> None:
    doc = Document(id="d1", text="hello world")
    assert doc.id == "d1"
    assert doc.text == "hello world"
    assert doc.metadata == {}


def test_document_with_metadata() -> None:
    doc = Document(id="d1", text="x", metadata={"src": "wiki"})
    assert doc.metadata["src"] == "wiki"


def test_chunk_links_to_document() -> None:
    c = Chunk(id="c1", text="part of doc", doc_id="d1", position=0)
    assert c.doc_id == "d1"
    assert c.position == 0
    assert c.metadata == {}


def test_qa_example_short_answer() -> None:
    ex = QAExample(id="q1", question="who?", answers=["Alice"])
    assert ex.answers == ["Alice"]
    assert ex.long_answer is None


def test_qa_example_long_form() -> None:
    ex = QAExample(id="q1", question="explain X", answers=[], long_answer="It is …")
    assert ex.long_answer == "It is …"


def test_entity_dataclass() -> None:
    e = Entity(id="alice", name="Alice", type="PERSON")
    assert e.name == "Alice"
    assert e.type == "PERSON"


def test_relation_dataclass() -> None:
    r = Relation(src="alice", dst="bob", predicate="knows")
    assert r.predicate == "knows"


def test_scored_hit() -> None:
    h = ScoredHit(id="c1", score=0.91, meta={"src": "doc1"})
    assert h.score == 0.91
    assert h.meta["src"] == "doc1"
