"""The LightRAG method — assembles the indexing + query pipelines."""

from __future__ import annotations

from typing import ClassVar

from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method, MethodConfig
from llm_kg.methods.lightrag.embedder import LightRAGEmbedder
from llm_kg.methods.lightrag.extractor import LightRAGExtractor
from llm_kg.methods.lightrag.graph_builder import LightRAGGraphBuilder
from llm_kg.methods.lightrag.prompts import DEFAULT_ENTITY_TYPES, DEFAULT_LANGUAGE, PROMPTS
from llm_kg.methods.lightrag.query_processor import LightRAGQueryProcessor
from llm_kg.methods.lightrag.retriever import LightRAGRetriever
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.context_builder import DefaultContextBuilder
from llm_kg.pipeline.stages.generator import DefaultGenerator
from llm_kg.pipeline.stages.tiktoken_chunker import TiktokenChunker

# A short-answer-focused generation prompt for MuSiQue-style extractive QA.
# Uses LightRAG's grounding language but with a strict "answer-only" instruction
# at the end (LightRAG's stock rag_response targets long-form markdown).
SHORT_ANSWER_PROMPT = """Answer the question using ONLY the information in the context below.
If the context does not contain the answer, say "I don't know."
Respond with the shortest correct answer — typically one entity name or a short noun phrase.
Do not explain, do not cite sources, do not use markdown — output only the answer text.

Context:
{context}

Question: {question}

Answer:"""


@METHOD_REGISTRY.register("lightrag")
class LightRAG(Method):
    """LightRAG method as described in arXiv:2410.05779 (HKUDS/LightRAG).

    Configurable params (under `method.params` in YAML):
      chunk_size        : tokens per chunk (default 1200)
      chunk_overlap     : token overlap between chunks (default 100)
      chunker_model     : tiktoken model name (default gpt-4o-mini)
      entity_types      : list[str] of allowed entity types
      language          : extraction / keyword language (default English)
      max_gleaning      : extra LLM extraction passes (default 0; upstream 1)
      max_concurrent_extract : asyncio.Semaphore size for extraction (default 16)
      top_k_entities    : entities/relations per retrieval pass (default 40)
      top_k_chunks      : final chunks returned by the retriever (default 20)
      cosine_threshold  : drop VDB hits below this score (default 0.2)
      embed_batch_size  : batch size for embedding calls (default 64)
      prompt_template   : generation prompt format with {context} and {question}
    """

    name: ClassVar[str] = "lightrag"

    def build(
        self, cfg: MethodConfig, ctx: PipelineContext
    ) -> tuple[IndexingPipeline, QueryPipeline]:
        p = cfg.params

        indexing = IndexingPipeline(stages=[
            TiktokenChunker(
                chunk_size=p.get("chunk_size", 1200),
                chunk_overlap=p.get("chunk_overlap", 100),
                model=p.get("chunker_model", "gpt-4o-mini"),
            ),
            LightRAGExtractor(
                entity_types=p.get("entity_types") or DEFAULT_ENTITY_TYPES,
                language=p.get("language", DEFAULT_LANGUAGE),
                max_gleaning=p.get("max_gleaning", 0),
                max_concurrent=p.get("max_concurrent_extract", 16),
            ),
            LightRAGGraphBuilder(),
            LightRAGEmbedder(
                embed_batch_size=p.get("embed_batch_size", 64),
            ),
        ])

        query = QueryPipeline(stages=[
            LightRAGQueryProcessor(language=p.get("language", DEFAULT_LANGUAGE)),
            LightRAGRetriever(
                top_k_entities=p.get("top_k_entities", 40),
                top_k_chunks=p.get("top_k_chunks", 20),
                cosine_threshold=p.get("cosine_threshold", 0.2),
            ),
            DefaultContextBuilder(),
            DefaultGenerator(prompt_template=p.get("prompt_template", SHORT_ANSWER_PROMPT)),
        ])

        return indexing, query


# Touch the upstream PROMPTS dict to make sure prompts are importable
# (catches the case where someone edits prompts.py and breaks an import).
_ = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
