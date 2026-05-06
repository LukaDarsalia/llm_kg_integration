import numpy as np
import pytest

from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


def test_llm_response_dataclass() -> None:
    r = LLMResponse(text="hi", prompt_tokens=3, completion_tokens=1)
    assert r.text == "hi"
    assert r.prompt_tokens == 3
    assert r.cost_usd is None
    assert r.raw is None


def test_llm_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


async def test_concrete_llm_subclass_works() -> None:
    class FakeLLM(LLMProvider):
        async def generate(self, prompt: str, **kwargs):
            return LLMResponse(text="echo:" + prompt, prompt_tokens=1, completion_tokens=1)

        async def generate_structured(self, prompt: str, schema, **kwargs):
            raise NotImplementedError

    llm = FakeLLM()
    r = await llm.generate("hi")
    assert r.text == "echo:hi"


async def test_concrete_embedder_returns_2d_array() -> None:
    class FakeEmbedder(EmbeddingProvider):
        @property
        def dim(self) -> int:
            return 3

        async def embed(self, texts):
            return np.array([[float(len(t))] * 3 for t in texts])

    e = FakeEmbedder()
    out = await e.embed(["ab", "xyz"])
    assert out.shape == (2, 3)
    assert out[0, 0] == 2.0


def test_registries_are_distinct() -> None:
    assert LLM_REGISTRY is not EMBEDDING_REGISTRY
    # Day 1 ships no provider implementations.
    assert "openai" not in LLM_REGISTRY
    assert "openai" not in EMBEDDING_REGISTRY
