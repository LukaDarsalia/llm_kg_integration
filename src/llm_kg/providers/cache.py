"""Transparent on-disk caching for LLM and embedding providers.

Cache keys are SHA-256 of (provider_name, model, prompt-or-text, sorted_kwargs).
Cache values are JSON for LLM responses and `.npy` for embedding vectors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


def cache_key(provider_name: str, model: str, payload: str, kwargs: dict[str, Any]) -> str:
    """SHA-256 hex digest of the cache identity."""
    blob = json.dumps(
        {
            "provider": provider_name,
            "model": model,
            "payload": payload,
            "kwargs": kwargs,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CachedLLMProvider(LLMProvider):
    """Wraps an `LLMProvider`, caching `generate` responses to disk.

    `generate_structured` is intentionally NOT cached on day 1 — schemas serialize
    awkwardly and structured output is comparatively cheap to re-derive once the
    underlying generation is cached. Subclass and override if you need it.
    """

    def __init__(
        self,
        inner: LLMProvider,
        model: str,
        provider_name: str,
        cache_dir: Path,
    ) -> None:
        self._inner = inner
        self._model = model
        self._provider_name = provider_name
        self._cache_dir = cache_dir / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        key = cache_key(self._provider_name, self._model, prompt, kwargs)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return LLMResponse(**data)

        response = await self._inner.generate(prompt, **kwargs)
        path.write_text(json.dumps(asdict(response), ensure_ascii=False))
        return response

    async def generate_structured(self, prompt: str, schema, **kwargs: Any):
        return await self._inner.generate_structured(prompt, schema, **kwargs)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps an `EmbeddingProvider`, caching per-text vectors to disk.

    Partial cache hits are supported: only the missing texts are forwarded to the
    inner provider, and results are merged in input order.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        model: str,
        provider_name: str,
        cache_dir: Path,
    ) -> None:
        self._inner = inner
        self._model = model
        self._provider_name = provider_name
        self._cache_dir = cache_dir / "embed"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        keys = [cache_key(self._provider_name, self._model, t, {}) for t in texts]
        paths = [self._cache_dir / f"{k}.npy" for k in keys]

        cached: dict[int, np.ndarray] = {}
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, p in enumerate(paths):
            if p.exists():
                cached[i] = np.load(p)
            else:
                missing_idx.append(i)
                missing_texts.append(texts[i])

        if missing_texts:
            fresh = await self._inner.embed(missing_texts)
            if fresh.shape[0] != len(missing_texts):
                raise RuntimeError(
                    f"inner embedder returned {fresh.shape[0]} vectors for "
                    f"{len(missing_texts)} texts"
                )
            for j, idx in enumerate(missing_idx):
                vec = fresh[j]
                np.save(paths[idx], vec)
                cached[idx] = vec

        ordered = [cached[i] for i in range(len(texts))]
        return np.stack(ordered, axis=0)
