from pathlib import Path

import numpy as np

from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.providers.cache import CachedEmbeddingProvider, CachedLLMProvider, cache_key


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, **kwargs):
        self.calls += 1
        return LLMResponse(text=f"r:{prompt}", prompt_tokens=1, completion_tokens=1)

    async def generate_structured(self, prompt, schema, **kwargs):
        raise NotImplementedError


class CountingEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def dim(self) -> int:
        return 4

    async def embed(self, texts):
        self.calls += 1
        return np.array([[float(i + 1)] * 4 for i in range(len(texts))], dtype=np.float32)


def test_cache_key_is_deterministic() -> None:
    k1 = cache_key("openai", "gpt-4o", "hello", {"temperature": 0.5})
    k2 = cache_key("openai", "gpt-4o", "hello", {"temperature": 0.5})
    assert k1 == k2
    assert len(k1) == 64  # sha-256 hex


def test_cache_key_changes_with_input() -> None:
    base = cache_key("openai", "gpt-4o", "hello", {})
    assert base != cache_key("openai", "gpt-4o", "world", {})
    assert base != cache_key("openai", "gpt-4o-mini", "hello", {})
    assert base != cache_key("anthropic", "gpt-4o", "hello", {})
    assert base != cache_key("openai", "gpt-4o", "hello", {"temperature": 0.5})


def test_cache_key_kwargs_order_insensitive() -> None:
    a = cache_key("p", "m", "x", {"a": 1, "b": 2})
    b = cache_key("p", "m", "x", {"b": 2, "a": 1})
    assert a == b


async def test_cached_llm_hits_cache_on_repeat(tmp_path: Path) -> None:
    inner = CountingLLM()
    cached = CachedLLMProvider(inner=inner, model="m", provider_name="p", cache_dir=tmp_path)

    r1 = await cached.generate("hi")
    r2 = await cached.generate("hi")

    assert r1.text == r2.text == "r:hi"
    assert inner.calls == 1  # second call hit the cache


async def test_cached_llm_misses_for_different_prompt(tmp_path: Path) -> None:
    inner = CountingLLM()
    cached = CachedLLMProvider(inner=inner, model="m", provider_name="p", cache_dir=tmp_path)

    await cached.generate("a")
    await cached.generate("b")

    assert inner.calls == 2


async def test_cached_llm_persists_across_instances(tmp_path: Path) -> None:
    a = CachedLLMProvider(inner=CountingLLM(), model="m", provider_name="p", cache_dir=tmp_path)
    await a.generate("x")

    inner = CountingLLM()
    b = CachedLLMProvider(inner=inner, model="m", provider_name="p", cache_dir=tmp_path)
    r = await b.generate("x")

    assert r.text == "r:x"
    assert inner.calls == 0  # served from on-disk cache


async def test_cached_embedder_hits_cache(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = CachedEmbeddingProvider(
        inner=inner, model="e", provider_name="p", cache_dir=tmp_path
    )

    a = await cached.embed(["foo", "bar"])
    b = await cached.embed(["foo", "bar"])

    assert inner.calls == 1
    np.testing.assert_array_equal(a, b)


async def test_cached_embedder_partial_miss(tmp_path: Path) -> None:
    """If only some texts are cached, only the missing ones go to the inner provider."""
    inner = CountingEmbedder()
    cached = CachedEmbeddingProvider(
        inner=inner, model="e", provider_name="p", cache_dir=tmp_path
    )

    await cached.embed(["foo"])
    assert inner.calls == 1

    out = await cached.embed(["foo", "bar"])
    assert inner.calls == 2  # only "bar" was missing → one extra call
    assert out.shape == (2, 4)
