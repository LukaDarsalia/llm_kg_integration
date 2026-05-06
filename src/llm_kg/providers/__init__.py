"""LLM and embedding provider abstractions."""

from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.registry import Registry

LLM_REGISTRY: Registry[LLMProvider] = Registry("llm")
EMBEDDING_REGISTRY: Registry[EmbeddingProvider] = Registry("embedding")

__all__ = [
    "EMBEDDING_REGISTRY",
    "EmbeddingProvider",
    "LLM_REGISTRY",
    "LLMProvider",
    "LLMResponse",
]
