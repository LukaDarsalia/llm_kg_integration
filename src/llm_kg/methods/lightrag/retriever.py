"""LightRAGRetriever — dual-pass keyword retrieval + graph walks → chunk hits.

Simplified port of LightRAG's `_perform_kg_search` (mode='mix'):

1. Embed three texts in one batch call: original query, ll_keywords joined,
   hl_keywords joined.
2. **Low-level pass:** search entities VDB with ll_embedding → top_k_entities
   entities → walk graph to their source chunks via the entity node's
   `source_id` (chunks the entity was extracted from).
3. **High-level pass:** search relations VDB with hl_embedding → top_k_entities
   relations → walk graph to their source chunks via the edge's `source_id`.
4. **Vector pass:** search chunks VDB with the query embedding → top_k_chunks
   chunks directly (gives a hard floor of relevance for queries the KG doesn't
   cover well).
5. Round-robin merge the three chunk pools, dedup by chunk id, rank by the
   query-embedding similarity, return top `top_k_chunks`.

Differences from upstream:
- We skip the token-budget truncation steps — for MuSiQue's short-answer eval,
  chunk count matters more than packing context efficiently.
- We skip the cross-encoder reranker (would be a separate component).
- We don't surface entity/relation descriptions as JSON context — the
  DefaultContextBuilder just stitches chunk text. Adding entity/relation
  context lifts long-form quality but adds noise for short-answer F1.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from llm_kg.data.types import ScoredHit
from llm_kg.methods.lightrag.prompts import GRAPH_FIELD_SEP
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.query_processor import ProcessedQuery
from llm_kg.pipeline.stages.retriever import Retriever


def _parse_source_chunks(source_id: str | None) -> list[str]:
    if not source_id:
        return []
    return [s for s in source_id.split(GRAPH_FIELD_SEP) if s]


def _round_robin_merge(*lists: list[str]) -> list[str]:
    """Interleave lists, dedup preserving first-occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    iters = [iter(lst) for lst in lists]
    while True:
        progressed = False
        for it in iters:
            try:
                item = next(it)
                progressed = True
                if item not in seen:
                    out.append(item)
                    seen.add(item)
            except StopIteration:
                pass
        if not progressed:
            break
    return out


class LightRAGRetriever(Retriever):
    name: ClassVar[str] = "LightRAGRetriever"

    def __init__(
        self,
        top_k_entities: int = 40,
        top_k_chunks: int = 20,
        cosine_threshold: float = 0.2,
    ) -> None:
        self.top_k_entities = top_k_entities
        self.top_k_chunks = top_k_chunks
        self.cosine_threshold = cosine_threshold

    async def run(self, inp: ProcessedQuery, ctx: PipelineContext) -> list[ScoredHit]:
        if ctx.embedder is None or ctx.vector_store is None or ctx.graph_store is None:
            raise RuntimeError(
                "LightRAGRetriever requires ctx.embedder, ctx.vector_store, ctx.graph_store"
            )

        # ---------- batch-embed query + keyword joins ----------
        ll = inp.entities
        hl = list(inp.meta.get("high_level", []))
        ll_str = ", ".join(ll) if ll else inp.text
        hl_str = ", ".join(hl) if hl else inp.text
        to_embed = [inp.text, ll_str, hl_str]
        vecs = await ctx.embedder.embed(to_embed)
        q_vec, ll_vec, hl_vec = vecs[0], vecs[1], vecs[2]

        # ---------- low-level pass: entities ----------
        ent_hits = ctx.vector_store.search(
            ll_vec, k=self.top_k_entities, filter={"kind": "entity"}
        )
        ent_hits = [h for h in ent_hits if h.score >= self.cosine_threshold]
        ent_chunk_ids: list[str] = []
        for h in ent_hits:
            ent_name = h.meta.get("entity_name")
            if not ent_name:
                continue
            node = ctx.graph_store.get_node(ent_name)
            if node is None:
                continue
            ent_chunk_ids.extend(_parse_source_chunks(node.get("source_id")))

        # ---------- high-level pass: relations ----------
        rel_hits = ctx.vector_store.search(
            hl_vec, k=self.top_k_entities, filter={"kind": "relation"}
        )
        rel_hits = [h for h in rel_hits if h.score >= self.cosine_threshold]
        rel_chunk_ids: list[str] = []
        for h in rel_hits:
            src = h.meta.get("src")
            dst = h.meta.get("dst")
            if not src or not dst:
                continue
            edge = ctx.graph_store.get_edge(src, dst)
            if edge is None:
                continue
            rel_chunk_ids.extend(_parse_source_chunks(edge.get("source_id")))

        # ---------- vector pass: direct chunk search ----------
        vec_hits = ctx.vector_store.search(
            q_vec, k=self.top_k_chunks, filter={"kind": "chunk"}
        )
        # Convert each vec-store chunk id ('chunk-<md5>') back to the source
        # chunk id we stored in the chunk hit's meta.
        vec_chunk_ids = [h.meta["chunk_id"] for h in vec_hits if "chunk_id" in h.meta]
        # Build score map (chunk_id -> cosine to query) for final ranking.
        scores: dict[str, float] = {
            h.meta["chunk_id"]: h.score for h in vec_hits if "chunk_id" in h.meta
        }

        # ---------- round-robin merge candidates ----------
        candidates = _round_robin_merge(vec_chunk_ids, ent_chunk_ids, rel_chunk_ids)
        if not candidates:
            return []

        # ---------- score any candidates we don't have a score for yet by
        # re-fetching their vectors (cheap: filter to {kind: chunk} again and pick)
        unscored = [c for c in candidates if c not in scores]
        if unscored:
            # We didn't fetch these in the vector pass; do a wider search to score them.
            wider = ctx.vector_store.search(
                q_vec,
                k=max(self.top_k_chunks * 5, len(candidates) * 2),
                filter={"kind": "chunk"},
            )
            wider_by_chunk = {h.meta["chunk_id"]: h.score for h in wider if "chunk_id" in h.meta}
            for cid in unscored:
                scores.setdefault(cid, wider_by_chunk.get(cid, 0.0))

        # Final: rank candidates by score desc, take top_k_chunks
        ranked = sorted(candidates, key=lambda c: scores.get(c, 0.0), reverse=True)
        ranked = ranked[: self.top_k_chunks]
        return [ScoredHit(id=cid, score=scores.get(cid, 0.0), meta={}) for cid in ranked]
