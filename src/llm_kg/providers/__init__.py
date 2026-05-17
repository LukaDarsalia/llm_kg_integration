"""LLM and embedding provider abstractions."""

from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.registry import Registry

LLM_REGISTRY: Registry[LLMProvider] = Registry("llm")
EMBEDDING_REGISTRY: Registry[EmbeddingProvider] = Registry("embedding")

# Import concrete providers so their @register decorators run.
from llm_kg.providers import openrouter  # noqa: E402, F401

__all__ = [
    "EMBEDDING_REGISTRY",
    "EmbeddingProvider",
    "LLM_REGISTRY",
    "LLMProvider",
    "LLMResponse",
]
