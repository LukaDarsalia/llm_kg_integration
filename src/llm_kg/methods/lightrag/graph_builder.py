"""LightRAGGraphBuilder — merges extracted entities + relations into the graph store.

Merge rules (mirrors `_merge_nodes_then_upsert` and `_merge_edges_then_upsert`
in upstream `lightrag/operate.py`):

- **Entity merge key = exact entity name** (post-normalization in the parser).
- **Entity type**: majority vote across all extractions of the same name.
- **Entity description**: unique fragments joined by `<SEP>` (LightRAG also runs
  an LLM summarizer once ≥8 fragments accumulate — we skip that to save spend).
- **Entity source_chunks**: union, FIFO-capped at 300.
- **Edges undirected**: store the alphabetically-sorted (src, dst) pair.
- **Edge weight**: sum across extractions.
- **Edge keywords**: union (preserving order).
- **Edge description**: unique fragments joined by `<SEP>`.
- **Stub entities**: if a relation references a name not yet in the graph,
  create a stub node with type "UNKNOWN".
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

from llm_kg.data.types import Entity, Relation
from llm_kg.methods.lightrag.prompts import GRAPH_FIELD_SEP
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.extractor import ExtractionResult
from llm_kg.pipeline.stages.graph_builder import GraphBuilder

_MAX_SOURCE_IDS = 300


def _join_unique(*lists: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lst in lists:
        for item in lst:
            if item and item not in seen:
                out.append(item)
                seen.add(item)
    return out


def _join_descriptions(*descs: str) -> str:
    parts = _join_unique([d for d in descs if d])
    return GRAPH_FIELD_SEP.join(parts)


def _undirected_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class LightRAGGraphBuilder(GraphBuilder):
    name: ClassVar[str] = "LightRAGGraphBuilder"

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        if ctx.graph_store is None:
            raise RuntimeError("LightRAGGraphBuilder requires ctx.graph_store to be set")
        gs = ctx.graph_store

        # --- bucket extractions by entity name / edge pair so we apply merges once ---
        ents_by_name: dict[str, list[Entity]] = {}
        for e in inp.entities:
            ents_by_name.setdefault(e.name, []).append(e)

        rels_by_pair: dict[tuple[str, str], list[Relation]] = {}
        for r in inp.relations:
            pair = _undirected_pair(r.src, r.dst)
            rels_by_pair.setdefault(pair, []).append(r)

        # --- merge entities ---
        merged_entity_attrs: dict[str, dict] = {}
        for name, extractions in ents_by_name.items():
            existing = gs.get_node(name) or {}
            # Type vote: existing + all extractions
            type_votes = Counter()
            if existing.get("entity_type"):
                type_votes[existing["entity_type"]] += 1
            for e in extractions:
                type_votes[e.type] += 1
            best_type = type_votes.most_common(1)[0][0] if type_votes else ""

            # Descriptions: existing + all extractions (deduped, SEP-joined)
            existing_desc_parts = (existing.get("description") or "").split(GRAPH_FIELD_SEP)
            existing_desc_parts = [p for p in existing_desc_parts if p]
            new_descs = [e.metadata.get("description", "") for e in extractions]
            merged_desc = _join_descriptions(*existing_desc_parts, *new_descs)

            # Source chunks: existing + all extractions (FIFO cap)
            existing_src = (existing.get("source_id") or "").split(GRAPH_FIELD_SEP)
            existing_src = [s for s in existing_src if s]
            new_srcs: list[str] = []
            for e in extractions:
                new_srcs.extend(e.metadata.get("source_chunks", []))
            merged_src = _join_unique(existing_src, new_srcs)[-_MAX_SOURCE_IDS:]

            attrs = {
                "entity_id": name,
                "entity_type": best_type,
                "description": merged_desc,
                "source_id": GRAPH_FIELD_SEP.join(merged_src),
            }
            gs.add_node(name, **attrs)
            merged_entity_attrs[name] = attrs

        # --- merge relations ---
        for (src, dst), extractions in rels_by_pair.items():
            # Auto-create stub endpoints if missing (LightRAG behavior)
            for endpoint in (src, dst):
                if endpoint not in gs and endpoint not in merged_entity_attrs:
                    gs.add_node(
                        endpoint,
                        entity_id=endpoint,
                        entity_type="UNKNOWN",
                        description="",
                        source_id="",
                    )

            existing = gs.get_edge(src, dst) or {}
            existing_weight = float(existing.get("weight", 0.0))
            existing_desc_parts = (existing.get("description") or "").split(GRAPH_FIELD_SEP)
            existing_desc_parts = [p for p in existing_desc_parts if p]
            existing_kws = (existing.get("keywords") or "").split(",")
            existing_kws = [k.strip() for k in existing_kws if k.strip()]
            existing_src = (existing.get("source_id") or "").split(GRAPH_FIELD_SEP)
            existing_src = [s for s in existing_src if s]

            new_weight = sum(r.metadata.get("weight", 1.0) for r in extractions)
            new_descs = [r.metadata.get("description", "") for r in extractions]
            new_kws: list[str] = []
            for r in extractions:
                new_kws.extend(r.metadata.get("keywords", []))
            new_srcs: list[str] = []
            for r in extractions:
                new_srcs.extend(r.metadata.get("source_chunks", []))

            merged_kws = _join_unique(existing_kws, new_kws)
            merged_desc = _join_descriptions(*existing_desc_parts, *new_descs)
            merged_src = _join_unique(existing_src, new_srcs)[-_MAX_SOURCE_IDS:]

            gs.add_edge(
                src,
                dst,
                weight=existing_weight + new_weight,
                description=merged_desc,
                keywords=", ".join(merged_kws),
                source_id=GRAPH_FIELD_SEP.join(merged_src),
            )

        return inp
