"""Extractor + GraphBuilder integration tests with a stubbed LLM."""

from llm_kg.data.types import Chunk
from llm_kg.methods.lightrag.extractor import LightRAGExtractor
from llm_kg.methods.lightrag.graph_builder import LightRAGGraphBuilder
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.providers.base import LLMProvider, LLMResponse
from llm_kg.storage.networkx_graph import NetworkXGraphStore

TD = "<|#|>"
CD = "<|COMPLETE|>"


def _llm(canned_response: str) -> LLMProvider:
    class _StubLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, prompt, **kwargs):
            self.calls += 1
            return LLMResponse(text=canned_response, prompt_tokens=10, completion_tokens=5)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    return _StubLLM()


def _ctx(llm=None, graph=None) -> PipelineContext:
    return PipelineContext(
        llm=llm, embedder=None,
        vector_store=None, graph_store=graph, kv_store=None,
        logger=None, trace={},
    )


async def test_extractor_returns_entities_and_relations() -> None:
    canned = (
        f"entity{TD}Alice{TD}person{TD}A person.\n"
        f"entity{TD}Bob{TD}person{TD}Another.\n"
        f"relation{TD}Alice{TD}Bob{TD}friendship{TD}They are friends.\n"
        f"{CD}\n"
    )
    extractor = LightRAGExtractor(max_gleaning=0)
    chunks = [Chunk(id="c1", text="Alice knows Bob.", doc_id="d1", position=0)]
    er = await extractor.run(chunks, _ctx(llm=_llm(canned)))

    assert len(er.entities) == 2
    assert {e.name for e in er.entities} == {"Alice", "Bob"}
    assert len(er.relations) == 1
    rel = er.relations[0]
    assert rel.metadata["keywords"] == ["friendship"]
    # source chunk tagging
    assert er.entities[0].metadata["source_chunks"] == ["c1"]


async def test_extractor_gleaning_doubles_llm_calls() -> None:
    canned = f"entity{TD}A{TD}person{TD}d\n{CD}\n"
    llm = _llm(canned)
    extractor = LightRAGExtractor(max_gleaning=2)
    chunks = [Chunk(id="c1", text="x", doc_id="d", position=0)]
    await extractor.run(chunks, _ctx(llm=llm))
    assert llm.calls == 3  # 1 initial + 2 gleaning


async def test_graph_builder_writes_nodes_and_edges() -> None:
    from llm_kg.data.types import Entity, Relation
    from llm_kg.pipeline.stages.extractor import ExtractionResult

    gs = NetworkXGraphStore()
    er = ExtractionResult(
        chunks=[Chunk(id="c1", text="x", doc_id="d", position=0)],
        entities=[
            Entity(id="Alice", name="Alice", type="person",
                   metadata={"description": "A person.", "source_chunks": ["c1"]}),
            Entity(id="Bob", name="Bob", type="person",
                   metadata={"description": "Another.", "source_chunks": ["c1"]}),
        ],
        relations=[
            Relation(src="Alice", dst="Bob", predicate="friendship",
                     metadata={"description": "They are friends.",
                               "keywords": ["friendship"], "weight": 1.0,
                               "source_chunks": ["c1"]}),
        ],
    )
    await LightRAGGraphBuilder().run(er, _ctx(graph=gs))

    assert "Alice" in gs
    assert gs.get_node("Alice")["entity_type"] == "person"
    assert gs.get_node("Alice")["description"] == "A person."
    edge = gs.get_edge("Alice", "Bob")
    assert edge["weight"] == 1.0
    assert "friendship" in edge["keywords"]


async def test_graph_builder_merges_duplicates() -> None:
    from llm_kg.data.types import Entity
    from llm_kg.pipeline.stages.extractor import ExtractionResult

    gs = NetworkXGraphStore()

    # First batch
    er1 = ExtractionResult(
        chunks=[],
        entities=[Entity(id="Alice", name="Alice", type="person",
                         metadata={"description": "A founder.", "source_chunks": ["c1"]})],
        relations=[],
    )
    await LightRAGGraphBuilder().run(er1, _ctx(graph=gs))

    # Second batch: same name, different chunk, different description
    er2 = ExtractionResult(
        chunks=[],
        entities=[Entity(id="Alice", name="Alice", type="person",
                         metadata={"description": "A CEO.", "source_chunks": ["c2"]})],
        relations=[],
    )
    await LightRAGGraphBuilder().run(er2, _ctx(graph=gs))

    node = gs.get_node("Alice")
    assert "A founder." in node["description"]
    assert "A CEO." in node["description"]
    assert "c1" in node["source_id"]
    assert "c2" in node["source_id"]


async def test_graph_builder_sums_edge_weight_and_unions_keywords() -> None:
    from llm_kg.data.types import Relation
    from llm_kg.pipeline.stages.extractor import ExtractionResult

    gs = NetworkXGraphStore()
    er1 = ExtractionResult(
        chunks=[],
        entities=[],
        relations=[
            Relation(src="A", dst="B", predicate="",
                     metadata={"description": "d1", "keywords": ["k1"],
                               "weight": 1.0, "source_chunks": ["c1"]}),
        ],
    )
    er2 = ExtractionResult(
        chunks=[],
        entities=[],
        relations=[
            Relation(src="B", dst="A", predicate="",  # reversed direction
                     metadata={"description": "d2", "keywords": ["k2"],
                               "weight": 1.0, "source_chunks": ["c2"]}),
        ],
    )
    await LightRAGGraphBuilder().run(er1, _ctx(graph=gs))
    await LightRAGGraphBuilder().run(er2, _ctx(graph=gs))

    e = gs.get_edge("A", "B")
    assert e["weight"] == 2.0
    assert "k1" in e["keywords"] and "k2" in e["keywords"]
    assert "d1" in e["description"] and "d2" in e["description"]


async def test_graph_builder_creates_stub_for_unknown_endpoints() -> None:
    from llm_kg.data.types import Relation
    from llm_kg.pipeline.stages.extractor import ExtractionResult

    gs = NetworkXGraphStore()
    er = ExtractionResult(
        chunks=[],
        entities=[],
        relations=[
            Relation(src="X", dst="Y", predicate="",
                     metadata={"description": "", "keywords": [],
                               "weight": 1.0, "source_chunks": []}),
        ],
    )
    await LightRAGGraphBuilder().run(er, _ctx(graph=gs))

    assert gs.get_node("X")["entity_type"] == "UNKNOWN"
    assert gs.get_node("Y")["entity_type"] == "UNKNOWN"
