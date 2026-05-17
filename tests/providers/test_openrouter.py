"""Tests for the OpenRouter provider.

The openai client is patched out — no real API calls. Tests verify the request
shape we send and that response parsing produces the right `LLMResponse` /
`np.ndarray`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.base import EmbeddingProvider, LLMProvider


def test_openrouter_llm_is_registered() -> None:
    import llm_kg.providers.openrouter  # noqa: F401  triggers registration

    assert "openrouter" in LLM_REGISTRY
    cls = LLM_REGISTRY.get("openrouter")
    assert issubclass(cls, LLMProvider)


def test_openrouter_embedder_is_registered() -> None:
    import llm_kg.providers.openrouter  # noqa: F401

    assert "openrouter" in EMBEDDING_REGISTRY
    cls = EMBEDDING_REGISTRY.get("openrouter")
    assert issubclass(cls, EmbeddingProvider)


@patch("llm_kg.providers.openrouter.AsyncOpenAI")
async def test_openrouter_llm_generate_parses_response(mock_async_openai) -> None:
    from llm_kg.providers.openrouter import OpenRouterLLM

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        model="anthropic/claude-sonnet-4.6",
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="hello world"),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
                    )
                )
            )
        )
    )
    mock_async_openai.return_value = fake_client

    llm = OpenRouterLLM(model="anthropic/claude-sonnet-4.6", temperature=0.0)
    resp = await llm.generate("hi")

    assert resp.text == "hello world"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 3
    assert resp.raw == {"model": "anthropic/claude-sonnet-4.6", "finish_reason": "stop"}

    fake_client.chat.completions.create.assert_awaited_once()
    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4.6"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call_kwargs["temperature"] == 0.0


@patch("llm_kg.providers.openrouter.AsyncOpenAI")
async def test_openrouter_llm_generate_kwarg_overrides_default_temp(mock_async_openai) -> None:
    from llm_kg.providers.openrouter import OpenRouterLLM

    create = AsyncMock(
        return_value=SimpleNamespace(
            model="m",
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
    )
    mock_async_openai.return_value = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    llm = OpenRouterLLM(model="m", temperature=0.0)
    await llm.generate("p", temperature=0.9)

    assert create.await_args.kwargs["temperature"] == 0.9


@patch("llm_kg.providers.openrouter.AsyncOpenAI")
async def test_openrouter_embedder_returns_correct_shape(mock_async_openai) -> None:
    from llm_kg.providers.openrouter import OpenRouterEmbedder

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4]),
                        SimpleNamespace(embedding=[0.5, 0.6, 0.7, 0.8]),
                    ]
                )
            )
        )
    )
    mock_async_openai.return_value = fake_client

    embedder = OpenRouterEmbedder(model="openai/text-embedding-3-small", dim=4)
    out = await embedder.embed(["foo", "bar"])

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [0.1, 0.2, 0.3, 0.4])

    fake_client.embeddings.create.assert_awaited_once_with(
        model="openai/text-embedding-3-small", input=["foo", "bar"]
    )


@patch("llm_kg.providers.openrouter.AsyncOpenAI")
async def test_openrouter_embedder_empty_input_returns_zero_shape(mock_async_openai) -> None:
    from llm_kg.providers.openrouter import OpenRouterEmbedder

    mock_async_openai.return_value = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock())
    )

    embedder = OpenRouterEmbedder(model="m", dim=8)
    out = await embedder.embed([])

    assert out.shape == (0, 8)
    # API not called for empty input
    mock_async_openai.return_value.embeddings.create.assert_not_called()


@patch("llm_kg.providers.openrouter.AsyncOpenAI")
async def test_openrouter_embedder_raises_on_dim_mismatch(mock_async_openai) -> None:
    from llm_kg.providers.openrouter import OpenRouterEmbedder

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2])]  # only 2 dims
                )
            )
        )
    )
    mock_async_openai.return_value = fake_client

    embedder = OpenRouterEmbedder(model="m", dim=4)  # expects 4
    with pytest.raises(RuntimeError, match="expected"):
        await embedder.embed(["foo"])


def test_openrouter_sends_optional_headers() -> None:
    """Referer + title headers should be forwarded as default_headers to the client."""
    with patch("llm_kg.providers.openrouter.AsyncOpenAI") as mock_cls:
        from llm_kg.providers.openrouter import OpenRouterLLM

        OpenRouterLLM(model="m", referer="https://example.com", title="my-app")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["default_headers"] == {
            "HTTP-Referer": "https://example.com",
            "X-Title": "my-app",
        }


def test_openrouter_no_headers_when_unset() -> None:
    """Without referer/title, default_headers should be None (not an empty dict)."""
    with patch("llm_kg.providers.openrouter.AsyncOpenAI") as mock_cls:
        from llm_kg.providers.openrouter import OpenRouterLLM

        OpenRouterLLM(model="m")
        assert mock_cls.call_args.kwargs["default_headers"] is None
