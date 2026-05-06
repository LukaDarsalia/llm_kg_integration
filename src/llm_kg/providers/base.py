"""Abstract base classes for LLM and embedding providers.

Concrete provider implementations live in separate files (one per vendor) and
register themselves on `LLM_REGISTRY` / `EMBEDDING_REGISTRY`. Day 1 ships zero
concrete providers — this module just defines the contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel


@dataclass
class LLMResponse:
    """The result of one `LLMProvider.generate` call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None = None
    raw: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract chat/completion provider.

    Subclasses implement `generate` and `generate_structured`. They MUST be
    safe to call concurrently from `asyncio.gather`.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model's response to `prompt`."""

    @abstractmethod
    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        """Return a pydantic instance of `schema` parsed from the model's output.

        Implementations may use vendor-native structured output, function calling,
        or a parse-and-validate fallback. They MUST raise on a parse failure
        rather than silently returning a partial object.
        """


class EmbeddingProvider(ABC):
    """Abstract text embedding provider."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> "np.ndarray":
        """Embed `texts`. Returns an array of shape (len(texts), self.dim)."""
