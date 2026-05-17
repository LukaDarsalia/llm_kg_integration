"""LightRAGEmbedder — embeds chunks + entities + relations into one vector store.

Mirrors LightRAG's three-VDB design (`entities_vdb`, `relationships_vdb`,
`chunks_vdb`) but collapses them into a single `VectorStore` with a `kind`
metadata field so the retriever can filter. The content sent to the embedder
per kind matches upstream exactly:

| kind     | id prefix | embedded text                                      |
|----------|-----------|----------------------------------------------------|
| chunk    | chunk-    | the chunk's raw text                               |
| entity   | ent-      | f"{entity_name}\\n{description}"                   |
| relation | rel-      | f"{keywords}\\t{src}\\n{dst}\\n{description}"      |

Also persists each chunk's text and each entity/relation's description into
`ctx.kv_store` keyed by the same prefixed id (so the retriever can look up
context strings after a vector hit).
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from llm_kg.methods.lightrag.prompts import GRAPH_FIELD_SEP
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.embedder import Embedder
from llm_kg.pipeline.stages.extractor import ExtractionResult


def _mdhash(text: str, prefix: str) -> str:
    return f"{prefix}{hashlib.md5(text.encode('utf-8')).hexdigest()}"


def _entity_id(name: str) -> str:
    return _mdhash(name, "ent-")


def _relation_id(src: str, dst: str) -> str:
    a, b = sorted([src, dst])
    return _mdhash(a + b, "rel-")


def _chunk_vec_id(chunk_id: str) -> str:
    return _mdhash(chunk_id, "chunk-")


class LightRAGEmbedder(Embedder):
    name: ClassVar[str] = "LightRAGEmbedder"

    def __init__(self, embed_batch_size: int = 64) -> None:
        self.embed_batch_size = embed_batch_size

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        if ctx.embedder is None or ctx.vector_store is None or ctx.kv_store is None:
            raise RuntimeError(
                "LightRAGEmbedder requires ctx.embedder, ctx.vector_store, ctx.kv_store"
            )

        # ---------- chunks ----------
        if inp.chunks:
            chunk_ids = [_chunk_vec_id(c.id) for c in inp.chunks]
            chunk_texts = [c.text for c in inp.chunks]
            chunk_meta: list[dict[str, Any]] = [
                {"kind": "chunk", "chunk_id": c.id, "doc_id": c.doc_id, **c.metadata}
                for c in inp.chunks
            ]
            chunk_vecs = await self._embed_batched(ctx, chunk_texts)
            ctx.vector_store.upsert(chunk_ids, chunk_vecs, chunk_meta)
            # KV: chunk_id (NOT prefixed) -> raw text, for retriever / context builder
            ctx.kv_store.put_many({c.id: c.text for c in inp.chunks})

        # ---------- entities (dedupe by name to avoid embedding the same name twice) ----------
        ents_by_name: dict[str, tuple[str, str]] = {}  # name -> (name, joined_descriptions)
        for e in inp.entities:
            name = e.name
            desc = e.metadata.get("description", "") or ""
            if name in ents_by_name:
                _, prev = ents_by_name[name]
                if desc and desc not in prev.split(GRAPH_FIELD_SEP):
                    ents_by_name[name] = (name, f"{prev}{GRAPH_FIELD_SEP}{desc}" if prev else desc)
            else:
                ents_by_name[name] = (name, desc)

        if ents_by_name:
            ent_ids = [_entity_id(name) for name in ents_by_name]
            ent_texts = [f"{name}\n{desc}" for name, desc in ents_by_name.values()]
            ent_meta = [
                {"kind": "entity", "entity_name": name}
                for name in ents_by_name
            ]
            ent_vecs = await self._embed_batched(ctx, ent_texts)
            ctx.vector_store.upsert(ent_ids, ent_vecs, ent_meta)
            # KV: prefixed id -> the embedded text (for the context builder)
            ctx.kv_store.put_many(dict(zip(ent_ids, ent_texts, strict=True)))

        # ---------- relations (dedupe by undirected pair) ----------
        rels_by_pair: dict[tuple[str, str], list] = {}
        for r in inp.relations:
            a, b = sorted([r.src, r.dst])
            rels_by_pair.setdefault((a, b), []).append(r)

        if rels_by_pair:
            rel_ids: list[str] = []
            rel_texts: list[str] = []
            rel_meta: list[dict[str, Any]] = []
            rel_kv: dict[str, str] = {}
            for (a, b), extractions in rels_by_pair.items():
                rel_id = _relation_id(a, b)
                # Merge keywords + descriptions across extractions
                all_keywords: list[str] = []
                all_descs: list[str] = []
                for r in extractions:
                    all_keywords.extend(r.metadata.get("keywords", []))
                    desc = r.metadata.get("description", "") or ""
                    if desc:
                        all_descs.append(desc)
                kw_str = ", ".join(dict.fromkeys(all_keywords))  # dedup preserving order
                desc_str = GRAPH_FIELD_SEP.join(dict.fromkeys(all_descs))
                content = f"{kw_str}\t{a}\n{b}\n{desc_str}"
                rel_ids.append(rel_id)
                rel_texts.append(content)
                rel_meta.append({"kind": "relation", "src": a, "dst": b})
                rel_kv[rel_id] = content
            rel_vecs = await self._embed_batched(ctx, rel_texts)
            ctx.vector_store.upsert(rel_ids, rel_vecs, rel_meta)
            ctx.kv_store.put_many(rel_kv)

        return inp

    async def _embed_batched(self, ctx: PipelineContext, texts: list[str]):
        """Call ctx.embedder in batches of `embed_batch_size`."""
        import numpy as np

        if not texts:
            return np.zeros((0, ctx.embedder.dim), dtype=np.float32)
        out_batches = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            vecs = await ctx.embedder.embed(batch)
            out_batches.append(vecs)
        return np.concatenate(out_batches, axis=0)
