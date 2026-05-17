"""OpenRouter LLM and embedding providers.

OpenRouter is an OpenAI-compatible gateway that proxies many vendors (OpenAI,
Anthropic, Google, open-source via Together, etc.) through one endpoint. Both
chat completions and embeddings are exposed via the standard OpenAI SDK with a
custom `base_url`.

Reads `OPENROUTER_API_KEY` from the environment (or `.env`) via `Secrets`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from openai import AsyncOpenAI

from llm_kg.config.settings import Secrets
from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse

if TYPE_CHECKING:
    from pydantic import BaseModel


_BASE_URL = "https://openrouter.ai/api/v1"


def _client(referer: str | None, title: str | None) -> AsyncOpenAI:
    """Build a shared AsyncOpenAI client pointed at OpenRouter.

    OpenRouter encourages (but does not require) `HTTP-Referer` and `X-Title`
    headers for analytics and ranking on its public leaderboards.
    """
    headers: dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return AsyncOpenAI(
        base_url=_BASE_URL,
        api_key=Secrets().openrouter_api_key.get_secret_value(),
        default_headers=headers or None,
    )


@LLM_REGISTRY.register("openrouter")
class OpenRouterLLM(LLMProvider):
    """OpenRouter chat-completions provider.

    `model` is the OpenRouter model slug (e.g. "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o-mini", "google/gemini-2.0-flash"). See https://openrouter.ai/models.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = _client(referer, title)

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        }
        r = await self._client.chat.completions.create(**params)
        choice = r.choices[0]
        usage = r.usage
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            raw={"model": r.model, "finish_reason": choice.finish_reason},
        )

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        r = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        text = r.choices[0].message.content or "{}"
        return schema.model_validate_json(text)


@EMBEDDING_REGISTRY.register("openrouter")
class OpenRouterEmbedder(EmbeddingProvider):
    """OpenRouter embeddings provider.

    `model` is an OpenRouter embedding-model slug (e.g.
    "openai/text-embedding-3-small"). `dim` must match the model's output
    dimensionality — the framework needs it up front for vector-store
    construction. See https://openrouter.ai/models?fmt=cards&output_modalities=embeddings.
    """

    def __init__(
        self,
        model: str,
        dim: int,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self.model = model
        self._dim = dim
        self._client = _client(referer, title)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        r = await self._client.embeddings.create(model=self.model, input=texts)
        vectors = np.array([d.embedding for d in r.data], dtype=np.float32)
        if vectors.shape != (len(texts), self._dim):
            raise RuntimeError(
                f"OpenRouter returned vectors of shape {vectors.shape}, "
                f"expected ({len(texts)}, {self._dim}) — check the `dim` param "
                f"matches model '{self.model}'"
            )
        return vectors
