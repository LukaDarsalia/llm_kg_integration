"""Translate provider config (providers.yaml) into the callables LightRAG expects.

LightRAG takes an ``llm_model_func`` and an ``embedding_func``. We build both from the
shared provider config so the indexer and retriever stay byte-identical in how they call
models — which is what makes the S3 cache-and-reload safe (the query embedding lands in
the same vector space as the indexed one).

Mirrors GraphRAG-Bench's Examples/run_lightrag.py: the local embedding path uses
``lightrag.llm.hf.hf_embed`` over BAAI/bge-large-en-v1.5 (symmetric, no query prefix).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from src.pipeline.shared.providers import resolve_api_key

# Loaded HF embedding models are cached per-config so the evaluator does not reload the
# (heavy) bge model once per corpus.
_EMBED_CACHE: Dict[tuple, EmbeddingFunc] = {}


def build_lightrag_llm_func(llm_cfg: Dict[str, Any]):
    """Return an async ``llm_model_func`` calling an OpenAI-compatible chat endpoint."""
    api_key = resolve_api_key(llm_cfg)
    base_url = llm_cfg.get("base_url")
    model = llm_cfg["model"]
    params = dict(llm_cfg.get("params") or {})
    extra_body = llm_cfg.get("extra_body")

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Dict[str, Any]] | None = None,
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
        call_kwargs: Dict[str, Any] = dict(params)
        call_kwargs.update(kwargs)  # caller kwargs win over static params
        if extra_body is not None and "extra_body" not in call_kwargs:
            call_kwargs["extra_body"] = extra_body
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=base_url,
            api_key=api_key,
            keyword_extraction=keyword_extraction,
            **call_kwargs,
        )

    return llm_model_func


def build_lightrag_embedding_func(emb_cfg: Dict[str, Any]) -> EmbeddingFunc:
    """Return an ``EmbeddingFunc`` for the configured embedding provider (cached)."""
    dim = int(emb_cfg.get("dim", 1024))
    max_tok = int(emb_cfg.get("max_seq_len", 512))
    provider = emb_cfg.get("provider", "local")
    cache_key = (provider, emb_cfg.get("model"), dim, max_tok, emb_cfg.get("base_url"))
    if cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[cache_key]

    if provider == "local":
        # Lazy heavy imports so they only load when the local path is actually used.
        from lightrag.llm.hf import hf_embed
        from transformers import AutoModel, AutoTokenizer

        model_name = emb_cfg["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embed_model = AutoModel.from_pretrained(model_name)

        async def _embed(texts: List[str]):
            return await hf_embed(texts, tokenizer=tokenizer, embed_model=embed_model)

        embedding_func = EmbeddingFunc(embedding_dim=dim, max_token_size=max_tok, func=_embed)
    else:
        # Hosted, OpenAI-compatible /embeddings endpoint. Use openai_embed.func (unwrapped)
        # to avoid double EmbeddingFunc wrapping.
        api_key = resolve_api_key(emb_cfg)
        embedding_func = EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=max_tok,
            func=partial(
                openai_embed.func,
                model=emb_cfg["model"],
                base_url=emb_cfg.get("base_url"),
                api_key=api_key,
            ),
        )

    _EMBED_CACHE[cache_key] = embedding_func
    return embedding_func
