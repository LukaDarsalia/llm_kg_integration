# LLM × KG Pipeline Framework — Day-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the framework scaffolding (interfaces, registries, runner, logger, config, smoke test) for a research project that will eventually compare Naive RAG, HippoRAG, and LightRAG. No real method/provider/storage implementations land in this PR.

**Architecture:** Two-pipeline architecture (offline indexing, online query) talking through a storage layer. Every plug-in family (LLM, embedding, vector store, graph store, KV store, dataset, evaluator, method) is registered by a generic `Registry[T]` and resolved by name from a pydantic-validated YAML config. The `Experiment` runner wires everything end-to-end. See [docs/superpowers/specs/2026-05-06-llm-kg-pipeline-framework-design.md](../specs/2026-05-06-llm-kg-pipeline-framework-design.md) for the full design.

**Tech Stack:** Python ≥3.11 · uv (package manager) · pydantic + pydantic-settings (config) · numpy (vectors) · pyyaml (config files) · networkx (graph store interface signatures) · wandb (logging) · pytest + pytest-asyncio (tests) · rich (CLI output)

---

## Conventions used in this plan

- All commands assume CWD is the project root: `/Users/lukadarsalia/Desktop/GAIA Research Club/llm_kg_integration`.
- Tests use `pytest` with `pytest-asyncio` in **auto mode**, so `async def test_…` functions don't need `@pytest.mark.asyncio`.
- Every task ends in a single commit. Commit messages are conventional (`feat:`, `test:`, `chore:`).
- Imports always use the absolute path (`from llm_kg.registry import Registry`), never relative.
- "Exactly the code in this step" — copy it verbatim.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_kg/__init__.py`
- Create: `src/llm_kg/py.typed` (empty marker)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.python-version`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`
Expected: prints a version like `uv 0.4.x` or higher. If not installed, run `curl -LsSf https://astral.sh/uv/install.sh | sh` and reopen the shell.

- [ ] **Step 2: Pin Python version**

Create `.python-version`:
```
3.11
```

- [ ] **Step 3: Write `pyproject.toml`**

Create `pyproject.toml`:
```toml
[project]
name = "llm-kg"
version = "0.0.1"
description = "Research framework for LLM × Knowledge Graph integration (Naive RAG, HippoRAG, LightRAG and beyond)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.4",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "networkx>=3.2",
    "wandb>=0.17",
    "rich>=13.7",
    "aiofiles>=23.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[project.scripts]
llm-kg = "llm_kg.runner.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_kg"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-ra -q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]
```

- [ ] **Step 4: Create package skeleton**

Create `src/llm_kg/__init__.py`:
```python
"""LLM × KG integration research framework."""

__version__ = "0.0.1"
```

Create `src/llm_kg/py.typed` (empty file, marks the package as typed):
```
```

Create `tests/__init__.py`:
```
```

Create `tests/conftest.py`:
```python
"""Shared pytest fixtures."""
```

- [ ] **Step 5: Sync deps and verify**

Run: `uv sync --extra dev`
Expected: creates `.venv/` and `uv.lock`; prints `Resolved N packages` and `Installed N packages` with no errors.

Run: `uv run pytest --collect-only`
Expected: `0 tests collected` with exit status 5 (no tests found). That's fine — we have no tests yet.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/llm_kg/__init__.py src/llm_kg/py.typed tests/__init__.py tests/conftest.py .python-version
git commit -m "chore: scaffold uv project with pydantic, numpy, networkx, wandb, pytest"
```

---

## Task 2: Generic `Registry[T]`

**Files:**
- Create: `src/llm_kg/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_registry.py`:
```python
import pytest

from llm_kg.registry import Registry


class Animal:
    pass


class Dog(Animal):
    pass


class Cat(Animal):
    pass


def test_register_and_get() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    assert reg.get("dog") is Dog


def test_register_decorator_returns_class() -> None:
    reg: Registry[Animal] = Registry("animal")
    decorated = reg.register("dog")(Dog)
    assert decorated is Dog


def test_get_unknown_raises() -> None:
    reg: Registry[Animal] = Registry("animal")
    with pytest.raises(KeyError) as ei:
        reg.get("hippo")
    assert "animal" in str(ei.value)
    assert "hippo" in str(ei.value)


def test_duplicate_registration_raises() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    with pytest.raises(ValueError) as ei:
        reg.register("dog")(Cat)
    assert "dog" in str(ei.value)


def test_names_lists_registered() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    reg.register("cat")(Cat)
    assert sorted(reg.names()) == ["cat", "dog"]


def test_contains() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    assert "dog" in reg
    assert "cat" not in reg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: collection error or `ModuleNotFoundError: No module named 'llm_kg.registry'`.

- [ ] **Step 3: Implement the Registry**

Create `src/llm_kg/registry.py`:
```python
"""Generic name-keyed registry used by every plug-in family in the framework."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-keyed registry of classes.

    Used as the single resolution point between YAML configs (which name plug-ins
    by string) and Python classes. Each plug-in family (LLM provider, vector store,
    method, etc.) owns one Registry instance.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register `cls` under `name`. Raises if `name` already taken."""

        def _decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self._kind} '{name}' already registered "
                    f"(was {self._items[name].__name__}, tried to add {cls.__name__})"
                )
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind} '{name}' "
                f"(known: {sorted(self._items)})"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/registry.py tests/test_registry.py
git commit -m "feat: add generic name-keyed Registry[T]"
```

---

## Task 3: Core data types

**Files:**
- Create: `src/llm_kg/data/__init__.py`
- Create: `src/llm_kg/data/types.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/__init__.py`:
```
```

Create `tests/data/test_types.py`:
```python
from llm_kg.data.types import Chunk, Document, Entity, QAExample, Relation, ScoredHit


def test_document_minimal() -> None:
    doc = Document(id="d1", text="hello world")
    assert doc.id == "d1"
    assert doc.text == "hello world"
    assert doc.metadata == {}


def test_document_with_metadata() -> None:
    doc = Document(id="d1", text="x", metadata={"src": "wiki"})
    assert doc.metadata["src"] == "wiki"


def test_chunk_links_to_document() -> None:
    c = Chunk(id="c1", text="part of doc", doc_id="d1", position=0)
    assert c.doc_id == "d1"
    assert c.position == 0
    assert c.metadata == {}


def test_qa_example_short_answer() -> None:
    ex = QAExample(id="q1", question="who?", answers=["Alice"])
    assert ex.answers == ["Alice"]
    assert ex.long_answer is None


def test_qa_example_long_form() -> None:
    ex = QAExample(id="q1", question="explain X", answers=[], long_answer="It is …")
    assert ex.long_answer == "It is …"


def test_entity_dataclass() -> None:
    e = Entity(id="alice", name="Alice", type="PERSON")
    assert e.name == "Alice"
    assert e.type == "PERSON"


def test_relation_dataclass() -> None:
    r = Relation(src="alice", dst="bob", predicate="knows")
    assert r.predicate == "knows"


def test_scored_hit() -> None:
    h = ScoredHit(id="c1", score=0.91, meta={"src": "doc1"})
    assert h.score == 0.91
    assert h.meta["src"] == "doc1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_types.py -v`
Expected: `ModuleNotFoundError: No module named 'llm_kg.data'`.

- [ ] **Step 3: Implement the data types**

Create `src/llm_kg/data/__init__.py`:
```python
"""Data types and dataset/corpus abstractions."""
```

Create `src/llm_kg/data/types.py`:
```python
"""Plain dataclasses used throughout the pipeline.

These are intentionally minimal; methods may carry additional info via the
`metadata` dict rather than subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of corpus input (e.g., one Wikipedia page, one PDF, one note)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A contiguous piece of a Document, as produced by a Chunker."""

    id: str
    text: str
    doc_id: str
    position: int  # ordinal within the parent document, 0-indexed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    name: str
    type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    src: str
    dst: str
    predicate: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    """One question-answer record from a dataset.

    For extractive datasets, `answers` holds gold short-form answers (often
    multiple acceptable forms). For long-form datasets, `long_answer` holds the
    reference response and `answers` may be empty.
    """

    id: str
    question: str
    answers: list[str] = field(default_factory=list)
    long_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredHit:
    """A retriever's output: an item id with a score and arbitrary metadata."""

    id: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_types.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/data/ tests/data/
git commit -m "feat: add core data types (Document, Chunk, Entity, Relation, QAExample, ScoredHit)"
```

---

## Task 4: Provider abstracts (LLM + Embedding) and registries

**Files:**
- Create: `src/llm_kg/providers/__init__.py`
- Create: `src/llm_kg/providers/base.py`
- Create: `tests/providers/__init__.py`
- Create: `tests/providers/test_base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/providers/__init__.py`:
```
```

Create `tests/providers/test_base.py`:
```python
import pytest
import numpy as np

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_base.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the abstracts and registries**

Create `src/llm_kg/providers/__init__.py`:
```python
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
```

Create `src/llm_kg/providers/base.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_base.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/providers/ tests/providers/
git commit -m "feat: add LLMProvider and EmbeddingProvider abstracts + registries"
```

---

## Task 5: Disk-cached provider wrappers

**Files:**
- Create: `src/llm_kg/providers/cache.py`
- Create: `tests/providers/test_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/providers/test_cache.py`:
```python
from pathlib import Path

import numpy as np
import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_cache.py -v`
Expected: `ImportError: cannot import name 'CachedLLMProvider' …`.

- [ ] **Step 3: Implement the cache wrappers**

Create `src/llm_kg/providers/cache.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_cache.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/providers/cache.py tests/providers/test_cache.py
git commit -m "feat: add transparent disk cache for LLM and embedding providers"
```

---

## Task 6: Storage abstracts (vector / graph / KV) and registries

**Files:**
- Create: `src/llm_kg/storage/__init__.py`
- Create: `src/llm_kg/storage/base.py`
- Create: `tests/storage/__init__.py`
- Create: `tests/storage/test_base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/storage/__init__.py`:
```
```

Create `tests/storage/test_base.py`:
```python
import pytest

from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY
from llm_kg.storage.base import GraphStore, KVStore, VectorStore


def test_vector_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_graph_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        GraphStore()  # type: ignore[abstract]


def test_kv_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        KVStore()  # type: ignore[abstract]


def test_registries_are_distinct() -> None:
    assert VECTOR_REGISTRY is not GRAPH_REGISTRY
    assert GRAPH_REGISTRY is not KV_REGISTRY
    # Day 1 ships no storage implementations.
    assert "numpy" not in VECTOR_REGISTRY
    assert "networkx" not in GRAPH_REGISTRY
    assert "jsonfile" not in KV_REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/storage/test_base.py -v`
Expected: `ModuleNotFoundError: No module named 'llm_kg.storage'`.

- [ ] **Step 3: Implement the storage abstracts**

Create `src/llm_kg/storage/__init__.py`:
```python
"""Storage backend abstractions (vector, graph, KV)."""

from llm_kg.registry import Registry
from llm_kg.storage.base import GraphStore, KVStore, VectorStore

VECTOR_REGISTRY: Registry[VectorStore] = Registry("vector_store")
GRAPH_REGISTRY: Registry[GraphStore] = Registry("graph_store")
KV_REGISTRY: Registry[KVStore] = Registry("kv_store")

__all__ = [
    "GRAPH_REGISTRY",
    "GraphStore",
    "KV_REGISTRY",
    "KVStore",
    "VECTOR_REGISTRY",
    "VectorStore",
]
```

Create `src/llm_kg/storage/base.py`:
```python
"""Abstract base classes for the three storage backends.

Concrete implementations live alongside (e.g. `numpy_vector_store.py`,
`networkx_graph_store.py`) and self-register on the relevant registry. Day 1
ships zero concrete implementations — only contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from llm_kg.data.types import ScoredHit

if TYPE_CHECKING:
    import numpy as np


class VectorStore(ABC):
    """A dense-vector store. Items are arbitrary string ids with metadata."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: "np.ndarray",
        meta: list[dict[str, Any]],
    ) -> None:
        """Add or overwrite `len(ids)` items. `vectors.shape == (len(ids), dim)`."""

    @abstractmethod
    def search(
        self,
        query: "np.ndarray",
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredHit]:
        """Return top-`k` by descending similarity. `query.shape == (dim,)`."""


class GraphStore(ABC):
    """A directed labelled multigraph store with PPR support."""

    @abstractmethod
    def add_node(self, id: str, **attrs: Any) -> None: ...

    @abstractmethod
    def add_edge(self, src: str, dst: str, **attrs: Any) -> None: ...

    @abstractmethod
    def neighbors(self, id: str) -> list[str]: ...

    @abstractmethod
    def personalized_pagerank(
        self,
        seeds: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """PPR over the graph with bias toward `seeds` (id → weight). Returns score per node."""


class KVStore(ABC):
    """A simple typed key-value store. Used for chunk/document text and method-specific blobs."""

    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def put(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def __contains__(self, key: str) -> bool: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/storage/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/storage/ tests/storage/
git commit -m "feat: add VectorStore/GraphStore/KVStore abstracts + registries"
```

---

## Task 7: Pipeline base (`Stage`, `PipelineContext`)

**Files:**
- Create: `src/llm_kg/pipeline/__init__.py`
- Create: `src/llm_kg/pipeline/stage.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_stage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/__init__.py`:
```
```

Create `tests/pipeline/test_stage.py`:
```python
import pytest

from llm_kg.pipeline.stage import PipelineContext, Stage


class AddOne(Stage[int, int]):
    name = "AddOne"

    async def run(self, inp: int, ctx: PipelineContext) -> int:
        return inp + 1


def test_stage_is_abstract() -> None:
    with pytest.raises(TypeError):
        Stage()  # type: ignore[abstract]


async def test_concrete_stage_runs() -> None:
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    s = AddOne()
    assert await s.run(1, ctx) == 2


def test_pipeline_context_trace_is_mutable() -> None:
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    ctx.trace["x"] = 1
    assert ctx.trace["x"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_stage.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the pipeline base**

Create `src/llm_kg/pipeline/__init__.py`:
```python
"""Pipeline composition primitives and stage abstractions."""
```

Create `src/llm_kg/pipeline/stage.py`:
```python
"""The Stage abstract base and the PipelineContext passed to every stage.

Stages are the unit of composition. Each stage takes one input value, has access
to all providers/storages/loggers via `PipelineContext`, and returns one output
value. Concrete stage families (Chunker, Retriever, …) further restrict
`InT`/`OutT` to specific data types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from llm_kg.logging_.base import ExperimentLogger
    from llm_kg.providers.base import EmbeddingProvider, LLMProvider
    from llm_kg.storage.base import GraphStore, KVStore, VectorStore

InT = TypeVar("InT")
OutT = TypeVar("OutT")


@dataclass
class PipelineContext:
    """Carries everything a stage might need: providers, storage, logger, trace.

    `trace` is a free-form per-run dict where stages can record intermediate
    artifacts for debugging. Day-1 contract is loose — see the design doc's
    "Open questions" section.

    `query` carries the original question through the query pipeline so stages
    that don't see it directly (e.g. Generator, which receives only the assembled
    context) can still reference it. The runner sets it before each query.

    Storage and provider fields are typed as `Any` here to keep the import graph
    shallow; they are typed precisely on construction in the runner.
    """

    llm: Any  # LLMProvider | None
    embedder: Any  # EmbeddingProvider | None
    vector_store: Any  # VectorStore | None
    graph_store: Any  # GraphStore | None
    kv_store: Any  # KVStore | None
    logger: Any  # ExperimentLogger | None
    trace: dict[str, Any]
    query: str | None = None


class Stage(ABC, Generic[InT, OutT]):
    """One step of an indexing or query pipeline.

    Subclasses set a `name` (for logging) and implement `run`. They MAY also
    write side-effects to `ctx.vector_store`, `ctx.graph_store`, or
    `ctx.kv_store`; they SHOULD log timings via `ctx.logger.log_stage(...)`
    only if they do extra internal sub-steps (the pipeline already times the
    top-level call).
    """

    name: ClassVar[str] = "UnnamedStage"

    @abstractmethod
    async def run(self, inp: InT, ctx: PipelineContext) -> OutT: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_stage.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/pipeline/ tests/pipeline/
git commit -m "feat: add Stage abstract base and PipelineContext"
```

---

## Task 8: Indexing stage abstracts and defaults

**Files:**
- Create: `src/llm_kg/pipeline/stages/__init__.py`
- Create: `src/llm_kg/pipeline/stages/chunker.py`
- Create: `src/llm_kg/pipeline/stages/extractor.py`
- Create: `src/llm_kg/pipeline/stages/graph_builder.py`
- Create: `src/llm_kg/pipeline/stages/embedder.py`
- Create: `tests/pipeline/test_indexing_stages.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_indexing_stages.py`:
```python
import numpy as np
import pytest

from llm_kg.data.types import Chunk, Document, Entity, Relation
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.chunker import Chunker, DefaultChunker
from llm_kg.pipeline.stages.embedder import DefaultEmbedder, Embedder
from llm_kg.pipeline.stages.extractor import (
    ExtractionResult,
    InformationExtractor,
    NoOpExtractor,
)
from llm_kg.pipeline.stages.graph_builder import GraphBuilder, NoOpGraphBuilder


def _empty_ctx(**overrides) -> PipelineContext:
    base = dict(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    base.update(overrides)
    return PipelineContext(**base)


def test_chunker_is_abstract() -> None:
    with pytest.raises(TypeError):
        Chunker()  # type: ignore[abstract]


async def test_default_chunker_splits_by_word_count() -> None:
    chunker = DefaultChunker(chunk_size=5, chunk_overlap=0)
    docs = [Document(id="d1", text=" ".join(f"w{i}" for i in range(12)))]
    chunks = await chunker.run(docs, _empty_ctx())
    assert len(chunks) == 3
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    assert chunks[0].text.split() == ["w0", "w1", "w2", "w3", "w4"]


async def test_default_chunker_overlap() -> None:
    chunker = DefaultChunker(chunk_size=4, chunk_overlap=1)
    docs = [Document(id="d1", text=" ".join(f"w{i}" for i in range(7)))]
    chunks = await chunker.run(docs, _empty_ctx())
    # window 4, stride 3 (4 - 1) over 7 words → starts at 0, 3 → 2 chunks
    assert len(chunks) == 2
    # last word of chunk 0 == first word of chunk 1 (the overlap)
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]
    assert chunks[0].text.split()[-1] == "w3"


async def test_default_chunker_short_doc_one_chunk() -> None:
    chunker = DefaultChunker(chunk_size=100, chunk_overlap=0)
    docs = [Document(id="d1", text="only a few words")]
    chunks = await chunker.run(docs, _empty_ctx())
    assert len(chunks) == 1
    assert chunks[0].text == "only a few words"


def test_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        InformationExtractor()  # type: ignore[abstract]


async def test_noop_extractor_returns_empty_extraction() -> None:
    ext = NoOpExtractor()
    chunks = [Chunk(id="c1", text="x", doc_id="d1", position=0)]
    out = await ext.run(chunks, _empty_ctx())
    assert isinstance(out, ExtractionResult)
    assert out.chunks == chunks
    assert out.entities == []
    assert out.relations == []


def test_graph_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        GraphBuilder()  # type: ignore[abstract]


async def test_noop_graph_builder_passes_through() -> None:
    gb = NoOpGraphBuilder()
    er = ExtractionResult(
        chunks=[Chunk(id="c1", text="x", doc_id="d1", position=0)],
        entities=[Entity(id="alice", name="Alice")],
        relations=[Relation(src="alice", dst="bob", predicate="knows")],
    )
    out = await gb.run(er, _empty_ctx())
    assert out is er


def test_embedder_is_abstract() -> None:
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


async def test_default_embedder_writes_chunk_vectors_to_store() -> None:
    class FakeEmbedder:
        @property
        def dim(self) -> int:
            return 3

        async def embed(self, texts):
            return np.array([[float(len(t))] * 3 for t in texts])

    upserts: list[tuple[list[str], np.ndarray, list[dict]]] = []

    class FakeVecStore:
        def upsert(self, ids, vectors, meta):
            upserts.append((ids, vectors, meta))

        def search(self, q, k, filter=None):
            return []

    ctx = _empty_ctx(embedder=FakeEmbedder(), vector_store=FakeVecStore())
    er = ExtractionResult(
        chunks=[
            Chunk(id="c1", text="ab", doc_id="d1", position=0),
            Chunk(id="c2", text="xyz", doc_id="d1", position=1),
        ],
        entities=[],
        relations=[],
    )
    out = await DefaultEmbedder().run(er, ctx)
    assert out is er  # passes through
    assert len(upserts) == 1
    ids, vecs, metas = upserts[0]
    assert ids == ["c1", "c2"]
    assert vecs.shape == (2, 3)
    assert metas[0]["doc_id"] == "d1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_indexing_stages.py -v`
Expected: `ModuleNotFoundError: No module named 'llm_kg.pipeline.stages'`.

- [ ] **Step 3: Implement the stages**

Create `src/llm_kg/pipeline/stages/__init__.py`:
```python
"""Concrete stage abstractions and their default implementations."""
```

Create `src/llm_kg/pipeline/stages/chunker.py`:
```python
"""Chunker stage: splits a list of Documents into a flat list of Chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import Chunk, Document
from llm_kg.pipeline.stage import PipelineContext, Stage


class Chunker(Stage[list[Document], list[Chunk]], ABC):
    """Abstract: split documents into chunks."""

    name = "Chunker"

    @abstractmethod
    async def run(self, inp: list[Document], ctx: PipelineContext) -> list[Chunk]: ...


class DefaultChunker(Chunker):
    """Word-window chunker. Splits each document by whitespace, then groups
    `chunk_size` words at a time with `chunk_overlap` words of overlap.

    Cheap and language-agnostic. Methods that need sentence- or token-aware
    splitting should implement their own Chunker.
    """

    name = "DefaultChunker"

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def run(self, inp: list[Document], ctx: PipelineContext) -> list[Chunk]:
        out: list[Chunk] = []
        stride = self.chunk_size - self.chunk_overlap
        for doc in inp:
            words = doc.text.split()
            if not words:
                continue
            position = 0
            i = 0
            while i < len(words):
                window = words[i : i + self.chunk_size]
                out.append(
                    Chunk(
                        id=f"{doc.id}::chunk{position}",
                        text=" ".join(window),
                        doc_id=doc.id,
                        position=position,
                    )
                )
                position += 1
                if i + self.chunk_size >= len(words):
                    break
                i += stride
        return out
```

Create `src/llm_kg/pipeline/stages/extractor.py`:
```python
"""InformationExtractor stage: chunks → (entities, relations)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_kg.data.types import Chunk, Entity, Relation
from llm_kg.pipeline.stage import PipelineContext, Stage


@dataclass
class ExtractionResult:
    """The artifact passed from extractor → graph_builder → embedder.

    `chunks` is always preserved end-to-end; `entities` / `relations` are added by
    the extractor. Methods may attach extra fields via `meta`.
    """

    chunks: list[Chunk]
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


class InformationExtractor(Stage[list[Chunk], ExtractionResult], ABC):
    """Abstract: pull entities and relations out of chunks."""

    name = "InformationExtractor"

    @abstractmethod
    async def run(self, inp: list[Chunk], ctx: PipelineContext) -> ExtractionResult: ...


class NoOpExtractor(InformationExtractor):
    """Default for methods that don't build a knowledge graph (e.g. naive RAG)."""

    name = "NoOpExtractor"

    async def run(self, inp: list[Chunk], ctx: PipelineContext) -> ExtractionResult:
        return ExtractionResult(chunks=inp, entities=[], relations=[])
```

Create `src/llm_kg/pipeline/stages/graph_builder.py`:
```python
"""GraphBuilder stage: writes entities/relations from an ExtractionResult into the graph store."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.extractor import ExtractionResult


class GraphBuilder(Stage[ExtractionResult, ExtractionResult], ABC):
    """Abstract: persist the extracted graph into `ctx.graph_store`."""

    name = "GraphBuilder"

    @abstractmethod
    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult: ...


class NoOpGraphBuilder(GraphBuilder):
    """Default for methods that don't construct a graph."""

    name = "NoOpGraphBuilder"

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        return inp
```

Create `src/llm_kg/pipeline/stages/embedder.py`:
```python
"""Embedder stage: vectorizes chunks (and optionally entities) and writes them to the vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.extractor import ExtractionResult


class Embedder(Stage[ExtractionResult, ExtractionResult], ABC):
    """Abstract: produce dense vectors and persist them in `ctx.vector_store`."""

    name = "Embedder"

    @abstractmethod
    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult: ...


class DefaultEmbedder(Embedder):
    """Embeds every chunk's text via `ctx.embedder` and upserts to `ctx.vector_store`.

    Does NOT embed entities — methods that need entity embeddings (HippoRAG,
    LightRAG) implement their own Embedder.
    """

    name = "DefaultEmbedder"

    async def run(self, inp: ExtractionResult, ctx: PipelineContext) -> ExtractionResult:
        if not inp.chunks:
            return inp
        if ctx.embedder is None:
            raise RuntimeError("DefaultEmbedder requires ctx.embedder to be set")
        if ctx.vector_store is None:
            raise RuntimeError("DefaultEmbedder requires ctx.vector_store to be set")

        texts = [c.text for c in inp.chunks]
        ids = [c.id for c in inp.chunks]
        meta = [{"doc_id": c.doc_id, "position": c.position, **c.metadata} for c in inp.chunks]

        vectors = await ctx.embedder.embed(texts)
        ctx.vector_store.upsert(ids, vectors, meta)
        return inp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_indexing_stages.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/pipeline/stages/__init__.py src/llm_kg/pipeline/stages/chunker.py src/llm_kg/pipeline/stages/extractor.py src/llm_kg/pipeline/stages/graph_builder.py src/llm_kg/pipeline/stages/embedder.py tests/pipeline/test_indexing_stages.py
git commit -m "feat: add indexing stage abstracts (Chunker/Extractor/GraphBuilder/Embedder) + defaults"
```

---

## Task 9: Query stage abstracts and defaults

**Files:**
- Create: `src/llm_kg/pipeline/stages/query_processor.py`
- Create: `src/llm_kg/pipeline/stages/retriever.py`
- Create: `src/llm_kg/pipeline/stages/context_builder.py`
- Create: `src/llm_kg/pipeline/stages/generator.py`
- Create: `tests/pipeline/test_query_stages.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_query_stages.py`:
```python
import pytest

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.context_builder import ContextBuilder, DefaultContextBuilder
from llm_kg.pipeline.stages.generator import DefaultGenerator, Generator
from llm_kg.pipeline.stages.query_processor import (
    IdentityQueryProcessor,
    ProcessedQuery,
    QueryProcessor,
)
from llm_kg.pipeline.stages.retriever import Retriever
from llm_kg.providers.base import LLMResponse


def _empty_ctx(**overrides) -> PipelineContext:
    base = dict(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    base.update(overrides)
    return PipelineContext(**base)


def test_query_processor_is_abstract() -> None:
    with pytest.raises(TypeError):
        QueryProcessor()  # type: ignore[abstract]


async def test_identity_query_processor_wraps_string() -> None:
    qp = IdentityQueryProcessor()
    out = await qp.run("who is alice?", _empty_ctx())
    assert isinstance(out, ProcessedQuery)
    assert out.text == "who is alice?"
    assert out.keywords == []
    assert out.entities == []


def test_retriever_is_abstract() -> None:
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


def test_retriever_has_no_default() -> None:
    """No DefaultRetriever — every method must define its own."""
    import llm_kg.pipeline.stages.retriever as r

    assert not hasattr(r, "DefaultRetriever")


def test_context_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        ContextBuilder()  # type: ignore[abstract]


async def test_default_context_builder_concatenates_chunks_via_kv() -> None:
    class FakeKV:
        def __init__(self) -> None:
            self.data = {"c1": "alpha text", "c2": "beta text"}

        def get(self, key):
            return self.data[key]

        def put(self, key, value):
            self.data[key] = value

        def __contains__(self, key):
            return key in self.data

    ctx = _empty_ctx(kv_store=FakeKV())
    hits = [ScoredHit(id="c1", score=0.9, meta={}), ScoredHit(id="c2", score=0.8, meta={})]
    ctx_str = await DefaultContextBuilder().run(hits, ctx)
    assert "alpha text" in ctx_str
    assert "beta text" in ctx_str
    # higher score first
    assert ctx_str.find("alpha text") < ctx_str.find("beta text")


async def test_default_context_builder_empty_hits() -> None:
    ctx = _empty_ctx()
    out = await DefaultContextBuilder().run([], ctx)
    assert out == ""


def test_generator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


async def test_default_generator_reads_question_from_ctx_and_calls_llm() -> None:
    seen = {}

    class FakeLLM:
        async def generate(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return LLMResponse(text="42", prompt_tokens=10, completion_tokens=1)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    ctx = _empty_ctx(llm=FakeLLM(), query="what?")
    answer = await DefaultGenerator().run("ctx text", ctx)
    assert answer == "42"
    assert "what?" in seen["prompt"]
    assert "ctx text" in seen["prompt"]


async def test_default_generator_raises_without_query() -> None:
    class FakeLLM:
        async def generate(self, prompt, **kwargs):
            return LLMResponse(text="x", prompt_tokens=1, completion_tokens=1)

        async def generate_structured(self, *a, **k):
            raise NotImplementedError

    ctx = _empty_ctx(llm=FakeLLM(), query=None)
    with pytest.raises(RuntimeError, match="ctx.query"):
        await DefaultGenerator().run("ctx text", ctx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_query_stages.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the stages**

Create `src/llm_kg/pipeline/stages/query_processor.py`:
```python
"""QueryProcessor stage: raw question string → structured ProcessedQuery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_kg.pipeline.stage import PipelineContext, Stage


@dataclass
class ProcessedQuery:
    """A query enriched with extracted keywords / entities for graph retrieval."""

    text: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


class QueryProcessor(Stage[str, ProcessedQuery], ABC):
    """Abstract: turn a raw question into a ProcessedQuery."""

    name = "QueryProcessor"

    @abstractmethod
    async def run(self, inp: str, ctx: PipelineContext) -> ProcessedQuery: ...


class IdentityQueryProcessor(QueryProcessor):
    """Default: wrap the question with no keyword/entity extraction.

    Naive RAG uses this; HippoRAG and LightRAG implement their own.
    """

    name = "IdentityQueryProcessor"

    async def run(self, inp: str, ctx: PipelineContext) -> ProcessedQuery:
        return ProcessedQuery(text=inp)
```

Create `src/llm_kg/pipeline/stages/retriever.py`:
```python
"""Retriever stage: ProcessedQuery → ranked list of ScoredHit.

There is no default retriever — every method must implement its own (this is
the most distinguishing stage between methods).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext, Stage
from llm_kg.pipeline.stages.query_processor import ProcessedQuery


class Retriever(Stage[ProcessedQuery, list[ScoredHit]], ABC):
    """Abstract: return ranked chunk hits for a processed query."""

    name = "Retriever"

    @abstractmethod
    async def run(self, inp: ProcessedQuery, ctx: PipelineContext) -> list[ScoredHit]: ...
```

Create `src/llm_kg/pipeline/stages/context_builder.py`:
```python
"""ContextBuilder stage: ranked hits → a single context string for the LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.data.types import ScoredHit
from llm_kg.pipeline.stage import PipelineContext, Stage


class ContextBuilder(Stage[list[ScoredHit], str], ABC):
    """Abstract: format retrieved hits into the prompt-ready context."""

    name = "ContextBuilder"

    @abstractmethod
    async def run(self, inp: list[ScoredHit], ctx: PipelineContext) -> str: ...


class DefaultContextBuilder(ContextBuilder):
    """Default: pull each hit's text from `ctx.kv_store` and concatenate, highest-score first."""

    name = "DefaultContextBuilder"

    def __init__(self, separator: str = "\n\n---\n\n") -> None:
        self.separator = separator

    async def run(self, inp: list[ScoredHit], ctx: PipelineContext) -> str:
        if not inp:
            return ""
        if ctx.kv_store is None:
            raise RuntimeError("DefaultContextBuilder requires ctx.kv_store to be set")
        ordered = sorted(inp, key=lambda h: h.score, reverse=True)
        parts = [str(ctx.kv_store.get(h.id)) for h in ordered]
        return self.separator.join(parts)
```

Create `src/llm_kg/pipeline/stages/generator.py`:
```python
"""Generator stage: context (str) → answer (str). Reads the question from `ctx.query`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_kg.pipeline.stage import PipelineContext, Stage


class Generator(Stage[str, str], ABC):
    """Abstract: produce the final answer given the assembled context.

    The question itself is read from `ctx.query` (set by the runner before each
    query), not threaded through the pipeline. This lets every intermediate
    stage stay focused on its own data type.
    """

    name = "Generator"

    @abstractmethod
    async def run(self, inp: str, ctx: PipelineContext) -> str: ...


_DEFAULT_PROMPT = """Answer the question using only the context below. If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class DefaultGenerator(Generator):
    """Default: stuff the context into a single prompt and call `ctx.llm.generate`."""

    name = "DefaultGenerator"

    def __init__(self, prompt_template: str = _DEFAULT_PROMPT) -> None:
        self.prompt_template = prompt_template

    async def run(self, inp: str, ctx: PipelineContext) -> str:
        if ctx.llm is None:
            raise RuntimeError("DefaultGenerator requires ctx.llm to be set")
        if ctx.query is None:
            raise RuntimeError(
                "DefaultGenerator requires ctx.query to be set "
                "(the runner should set it before query.run)"
            )
        prompt = self.prompt_template.format(context=inp, question=ctx.query)
        response = await ctx.llm.generate(prompt)
        return response.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_query_stages.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/pipeline/stages/query_processor.py src/llm_kg/pipeline/stages/retriever.py src/llm_kg/pipeline/stages/context_builder.py src/llm_kg/pipeline/stages/generator.py tests/pipeline/test_query_stages.py
git commit -m "feat: add query stage abstracts (QueryProcessor/Retriever/ContextBuilder/Generator) + defaults"
```

---

## Task 10: Pipeline composers (`IndexingPipeline`, `QueryPipeline`)

**Files:**
- Create: `src/llm_kg/pipeline/indexing.py`
- Create: `src/llm_kg/pipeline/query.py`
- Create: `tests/pipeline/test_composers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_composers.py`:
```python
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext, Stage


class AddOne(Stage[int, int]):
    name = "AddOne"

    async def run(self, inp, ctx):
        return inp + 1


class Double(Stage[int, int]):
    name = "Double"

    async def run(self, inp, ctx):
        return inp * 2


def _empty_ctx() -> PipelineContext:
    return PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )


async def test_indexing_pipeline_runs_stages_in_order() -> None:
    p = IndexingPipeline(stages=[AddOne(), Double()])
    out = await p.run(1, _empty_ctx())
    assert out == 4  # (1+1)*2


async def test_query_pipeline_runs_stages_in_order() -> None:
    p = QueryPipeline(stages=[Double(), AddOne()])
    out = await p.run(1, _empty_ctx())
    assert out == 3  # (1*2)+1


async def test_pipeline_logs_each_stage_when_logger_present() -> None:
    logged: list[tuple[str, float]] = []

    class FakeLogger:
        def log_stage(self, stage, duration_s, **extras):
            logged.append((stage, duration_s))

        def init(self, *a, **k): ...
        def log_metrics(self, *a, **k): ...
        def log_predictions(self, *a, **k): ...
        def log_artifact(self, *a, **k): ...
        def finish(self): ...

    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=FakeLogger(), trace={},
    )
    p = IndexingPipeline(stages=[AddOne(), Double()])
    await p.run(0, ctx)
    assert [name for name, _ in logged] == ["AddOne", "Double"]
    assert all(d >= 0 for _, d in logged)


async def test_empty_pipeline_returns_input_unchanged() -> None:
    p = IndexingPipeline(stages=[])
    assert await p.run(42, _empty_ctx()) == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_composers.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the composers**

Create `src/llm_kg/pipeline/indexing.py`:
```python
"""Sequential indexing pipeline composer.

The composer is intentionally type-loose: it threads the output of each stage
into the next without static checks. The Method that assembles the pipeline is
responsible for stage compatibility — see the design doc, §4.4.
"""

from __future__ import annotations

import time
from typing import Any

from llm_kg.pipeline.stage import PipelineContext, Stage


class IndexingPipeline:
    """Runs a list of `Stage`s in order, threading values through.

    Times each stage and reports via `ctx.logger.log_stage(name, duration)` if
    a logger is present.
    """

    def __init__(self, stages: list[Stage[Any, Any]]) -> None:
        self.stages = stages

    async def run(self, inp: Any, ctx: PipelineContext) -> Any:
        current: Any = inp
        for stage in self.stages:
            t0 = time.perf_counter()
            current = await stage.run(current, ctx)
            elapsed = time.perf_counter() - t0
            if ctx.logger is not None:
                ctx.logger.log_stage(stage.name, elapsed)
        return current
```

Create `src/llm_kg/pipeline/query.py`:
```python
"""Sequential query pipeline composer. See `indexing.py` for design notes."""

from __future__ import annotations

import time
from typing import Any

from llm_kg.pipeline.stage import PipelineContext, Stage


class QueryPipeline:
    """Runs a list of `Stage`s in order for one query."""

    def __init__(self, stages: list[Stage[Any, Any]]) -> None:
        self.stages = stages

    async def run(self, inp: Any, ctx: PipelineContext) -> Any:
        current: Any = inp
        for stage in self.stages:
            t0 = time.perf_counter()
            current = await stage.run(current, ctx)
            elapsed = time.perf_counter() - t0
            if ctx.logger is not None:
                ctx.logger.log_stage(stage.name, elapsed)
        return current
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_composers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/pipeline/indexing.py src/llm_kg/pipeline/query.py tests/pipeline/test_composers.py
git commit -m "feat: add IndexingPipeline and QueryPipeline composers with per-stage timing"
```

---

## Task 11: `Method` abstract and `METHOD_REGISTRY`

**Files:**
- Create: `src/llm_kg/methods/__init__.py`
- Create: `src/llm_kg/methods/base.py`
- Create: `tests/methods/__init__.py`
- Create: `tests/methods/test_base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/methods/__init__.py`:
```
```

Create `tests/methods/test_base.py`:
```python
import pytest

from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method, MethodConfig
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext


def test_method_is_abstract() -> None:
    with pytest.raises(TypeError):
        Method()  # type: ignore[abstract]


def test_method_config_holds_arbitrary_params() -> None:
    cfg = MethodConfig(name="x", params={"chunk_size": 256, "top_k": 5})
    assert cfg.params["chunk_size"] == 256
    assert cfg.name == "x"


def test_concrete_method_returns_two_pipelines() -> None:
    class FakeMethod(Method):
        name = "fake"

        def build(self, cfg, ctx):
            return IndexingPipeline(stages=[]), QueryPipeline(stages=[])

    m = FakeMethod()
    ctx = PipelineContext(
        llm=None, embedder=None,
        vector_store=None, graph_store=None, kv_store=None,
        logger=None, trace={},
    )
    indexing, query = m.build(MethodConfig(name="fake"), ctx)
    assert isinstance(indexing, IndexingPipeline)
    assert isinstance(query, QueryPipeline)


def test_method_registry_constructed() -> None:
    # Day 1 ships no method implementations.
    assert "naive_rag" not in METHOD_REGISTRY
    assert "hipporag" not in METHOD_REGISTRY
    assert "lightrag" not in METHOD_REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/methods/test_base.py -v`
Expected: `ModuleNotFoundError: No module named 'llm_kg.methods'`.

- [ ] **Step 3: Implement the Method abstract**

Create `src/llm_kg/methods/__init__.py`:
```python
"""Method abstract + registry. Concrete methods (NaiveRAG, HippoRAG, LightRAG) land later."""

from llm_kg.methods.base import Method, MethodConfig
from llm_kg.registry import Registry

METHOD_REGISTRY: Registry[Method] = Registry("method")

__all__ = ["METHOD_REGISTRY", "Method", "MethodConfig"]
```

Create `src/llm_kg/methods/base.py`:
```python
"""The Method abstract: bundles a (IndexingPipeline, QueryPipeline) pair.

A Method picks which stage implementations go into each slot and how they're
parameterized. Day 1 ships zero concrete methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stage import PipelineContext


class MethodConfig(BaseModel):
    """Method-specific config block from YAML."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Method(ABC):
    """Assembles the indexing + query pipelines for one approach."""

    name: ClassVar[str] = "UnnamedMethod"

    @abstractmethod
    def build(
        self, cfg: MethodConfig, ctx: PipelineContext
    ) -> tuple[IndexingPipeline, QueryPipeline]:
        """Return the two pipelines fully wired for this method."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/methods/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/methods/ tests/methods/
git commit -m "feat: add Method abstract base + METHOD_REGISTRY"
```

---

## Task 12: Dataset abstracts (`Corpus`, `QADataset`)

**Files:**
- Create: `src/llm_kg/data/corpus.py`
- Create: `src/llm_kg/data/dataset.py`
- Create: `tests/data/test_corpus_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_corpus_dataset.py`:
```python
from typing import Iterable

import pytest

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample


def test_corpus_is_abstract() -> None:
    with pytest.raises(TypeError):
        Corpus()  # type: ignore[abstract]


def test_qa_dataset_is_abstract() -> None:
    with pytest.raises(TypeError):
        QADataset()  # type: ignore[abstract]


def test_dataset_registry_constructed() -> None:
    # Day 1 ships no dataset implementations.
    assert "hotpotqa" not in DATASET_REGISTRY
    assert "musique" not in DATASET_REGISTRY


def test_concrete_corpus_iterable() -> None:
    class InMemCorpus(Corpus):
        def __init__(self, docs: list[Document]) -> None:
            self._docs = docs

        def documents(self) -> Iterable[Document]:
            return iter(self._docs)

        def __len__(self) -> int:
            return len(self._docs)

    c = InMemCorpus([Document(id="a", text="x"), Document(id="b", text="y")])
    docs = list(c.documents())
    assert len(c) == 2
    assert [d.id for d in docs] == ["a", "b"]


def test_concrete_qa_dataset() -> None:
    class InMemQA(QADataset):
        def __init__(self, examples: list[QAExample]) -> None:
            self._ex = examples

        def examples(self) -> Iterable[QAExample]:
            return iter(self._ex)

        def __len__(self) -> int:
            return len(self._ex)

    ds = InMemQA([QAExample(id="q1", question="?", answers=["a"])])
    assert len(ds) == 1
    assert next(iter(ds.examples())).question == "?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_corpus_dataset.py -v`
Expected: `ImportError: cannot import name 'DATASET_REGISTRY' …` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dataset abstracts**

Replace `src/llm_kg/data/__init__.py` with:
```python
"""Data types and dataset/corpus abstractions."""

from llm_kg.data.dataset import QADataset
from llm_kg.registry import Registry

DATASET_REGISTRY: Registry[QADataset] = Registry("dataset")

__all__ = ["DATASET_REGISTRY", "QADataset"]
```

Create `src/llm_kg/data/corpus.py`:
```python
"""The Corpus abstract: a stream of Documents to index."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from llm_kg.data.types import Document


class Corpus(ABC):
    """Abstract: a (potentially large) collection of Documents.

    Returned as an Iterable so streaming corpora don't need to fit in memory.
    Implementations that can cheaply provide a count should override `__len__`.
    """

    @abstractmethod
    def documents(self) -> Iterable[Document]: ...

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} does not support len()")
```

Create `src/llm_kg/data/dataset.py`:
```python
"""The QADataset abstract: a corpus + paired QAExamples for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from llm_kg.data.corpus import Corpus
from llm_kg.data.types import Document, QAExample


class QADataset(ABC):
    """Abstract: question-answer dataset paired with its source corpus.

    Default `corpus()` raises — datasets that have an associated corpus should
    override it. Datasets that work against a separately-loaded corpus (e.g.
    open-domain HotpotQA) can leave it unimplemented.
    """

    @abstractmethod
    def examples(self) -> Iterable[QAExample]: ...

    def corpus(self) -> Corpus:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a corpus; "
            "load one separately and pass it to the runner"
        )

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} does not support len()")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_corpus_dataset.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/data/__init__.py src/llm_kg/data/corpus.py src/llm_kg/data/dataset.py tests/data/test_corpus_dataset.py
git commit -m "feat: add Corpus and QADataset abstracts + DATASET_REGISTRY"
```

---

## Task 13: Real EM / F1 / recall@k metrics

**Files:**
- Create: `src/llm_kg/evaluation/__init__.py`
- Create: `src/llm_kg/evaluation/metrics.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/__init__.py`:
```
```

Create `tests/evaluation/test_metrics.py`:
```python
from llm_kg.evaluation.metrics import exact_match, f1_score, normalize_text, recall_at_k


def test_normalize_lowercases_and_strips_articles_and_punct() -> None:
    assert normalize_text("The Cat's pajamas!") == "cats pajamas"
    assert normalize_text("  An apple. ") == "apple"
    assert normalize_text("a b") == "b"


def test_exact_match_simple() -> None:
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("Paris", ["London"]) == 0.0


def test_exact_match_any_of_many() -> None:
    assert exact_match("NYC", ["New York", "NYC", "the big apple"]) == 1.0


def test_exact_match_with_articles() -> None:
    assert exact_match("the answer", ["answer"]) == 1.0


def test_f1_partial_overlap() -> None:
    # pred 2/3 tokens correct, recall 2/2 → P=2/3, R=1, F1 = 2 * (2/3) / (1 + 2/3) = 0.8
    score = f1_score("the quick brown", ["quick brown"])
    assert abs(score - 0.8) < 1e-9


def test_f1_no_overlap() -> None:
    assert f1_score("alpha", ["beta"]) == 0.0


def test_f1_empty_prediction() -> None:
    assert f1_score("", ["something"]) == 0.0


def test_f1_picks_max_across_golds() -> None:
    # against "x" → 0; against "x y" → 1 → max is 1
    assert f1_score("x y", ["x", "x y"]) == 1.0


def test_recall_at_k_full_recall() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c3"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_truncation() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c3"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_no_relevant() -> None:
    # By convention, recall is 1.0 when there are no relevant items (vacuously true).
    assert recall_at_k(["a", "b"], [], k=2) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the metrics**

Create `src/llm_kg/evaluation/__init__.py` (intentionally minimal — full version with `EVALUATOR_REGISTRY` lands in Task 14, once `Evaluator` exists):

```python
"""Evaluation: metrics, judges, evaluators."""
```

Create `src/llm_kg/evaluation/metrics.py`:
```python
"""Standard short-answer QA metrics (HippoRAG / SQuAD style).

These follow the canonical SQuAD-EM/F1 normalization (lowercase, strip articles
and punctuation, collapse whitespace) and pick the max across multiple
acceptable gold answers.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def exact_match(prediction: str, golds: list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold, else 0.0."""
    pred = normalize_text(prediction)
    return 1.0 if any(pred == normalize_text(g) for g in golds) else 0.0


def _f1_one(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, golds: list[str]) -> float:
    """Maximum token-overlap F1 across `golds`."""
    if not golds:
        return 0.0
    return max(_f1_one(prediction, g) for g in golds)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of `relevant` items that appear in the first `k` of `retrieved`.

    By convention, returns 1.0 if `relevant` is empty (vacuously true).
    """
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/evaluation/__init__.py src/llm_kg/evaluation/metrics.py tests/evaluation/__init__.py tests/evaluation/test_metrics.py
git commit -m "feat: add EM, F1, and recall@k metric implementations"
```

---

## Task 14: Evaluator (`ExtractiveEvaluator` + `GenerativeEvaluator` shell)

**Files:**
- Create: `src/llm_kg/evaluation/evaluator.py`
- Create: `src/llm_kg/evaluation/judge.py`
- Modify: `src/llm_kg/evaluation/__init__.py`
- Create: `tests/evaluation/test_evaluator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/test_evaluator.py`:
```python
import pytest

from llm_kg.data.types import ScoredHit
from llm_kg.evaluation import EVALUATOR_REGISTRY, Prediction
from llm_kg.evaluation.evaluator import (
    Evaluator,
    ExtractiveEvaluator,
    GenerativeEvaluator,
)
from llm_kg.evaluation.judge import LLMJudge


def test_evaluator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]


def test_evaluator_registry_lists_extractive_and_generative() -> None:
    assert "extractive" in EVALUATOR_REGISTRY
    assert "generative" in EVALUATOR_REGISTRY


async def test_extractive_evaluator_aggregates_em_and_f1() -> None:
    preds = [
        Prediction(qid="q1", question="?", answer="Paris", retrieved=[], gold=["paris"]),
        Prediction(qid="q2", question="?", answer="London", retrieved=[], gold=["paris"]),
    ]
    ev = ExtractiveEvaluator()
    metrics = await ev.score(preds)
    assert metrics["em"] == 0.5
    assert 0.4 < metrics["f1"] <= 0.5
    assert metrics["n"] == 2


async def test_extractive_evaluator_recall_at_k() -> None:
    preds = [
        Prediction(
            qid="q1",
            question="?",
            answer="x",
            retrieved=[ScoredHit(id="c1", score=1, meta={}), ScoredHit(id="c3", score=0.5, meta={})],
            gold=["x"],
            relevant_ids=["c1"],
        ),
    ]
    ev = ExtractiveEvaluator(recall_ks=(1, 2))
    m = await ev.score(preds)
    assert m["recall@1"] == 1.0
    assert m["recall@2"] == 1.0


async def test_generative_evaluator_calls_judge_for_each_pair() -> None:
    """The shell judge always returns the configured baseline winner; we just verify wiring."""

    class StubJudge(LLMJudge):
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, question, candidate, baseline, axes):
            self.calls += 1
            return {axis: "candidate" for axis in axes}

    judge = StubJudge()
    ev = GenerativeEvaluator(
        judge=judge,
        baseline_predictions={
            "q1": "baseline answer 1",
            "q2": "baseline answer 2",
        },
        axes=("comprehensiveness", "overall"),
    )
    preds = [
        Prediction(qid="q1", question="?", answer="cand 1", retrieved=[], gold=None),
        Prediction(qid="q2", question="?", answer="cand 2", retrieved=[], gold=None),
    ]
    m = await ev.score(preds)
    assert judge.calls == 2
    assert m["winrate_comprehensiveness"] == 1.0
    assert m["winrate_overall"] == 1.0
    assert m["n"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/evaluation/test_evaluator.py -v`
Expected: `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement evaluator + judge shell**

Create `src/llm_kg/evaluation/judge.py`:
```python
"""LLM-as-judge abstraction (LightRAG-style head-to-head)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from llm_kg.providers.base import LLMProvider


class LLMJudge(ABC):
    """Abstract: judge a candidate vs baseline answer on multiple axes.

    Returns a dict mapping each axis to either "candidate" or "baseline" — the
    winner. Day 1 ships only a stub; a real prompt-based judge lands later.
    """

    @abstractmethod
    async def judge(
        self,
        question: str,
        candidate: str,
        baseline: str,
        axes: Iterable[str],
    ) -> dict[str, str]: ...


_JUDGE_PROMPT_STUB = """[STUB JUDGE PROMPT — replace before running real generative evals]

Question: {question}

Answer A (candidate): {candidate}
Answer B (baseline):  {baseline}

For each axis below, decide whether A or B is better and respond with only "A" or "B" on its own line.
Axes: {axes}
"""


class PromptedLLMJudge(LLMJudge):
    """A LightRAG-style judge backed by an LLMProvider.

    Day 1 the prompt is intentionally a placeholder; the real prompt and parsing
    land in the PR that adds the GenerativeEvaluator workflow.
    """

    def __init__(self, llm: LLMProvider, prompt_template: str = _JUDGE_PROMPT_STUB) -> None:
        self._llm = llm
        self._prompt = prompt_template

    async def judge(
        self,
        question: str,
        candidate: str,
        baseline: str,
        axes: Iterable[str],
    ) -> dict[str, str]:
        axes_list = list(axes)
        prompt = self._prompt.format(
            question=question, candidate=candidate, baseline=baseline, axes=", ".join(axes_list)
        )
        response = await self._llm.generate(prompt)
        # PLACEHOLDER parser — real version lands later.
        verdicts: dict[str, str] = {}
        lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
        for axis, line in zip(axes_list, lines):
            verdicts[axis] = "candidate" if line.upper().startswith("A") else "baseline"
        for axis in axes_list:
            verdicts.setdefault(axis, "baseline")
        return verdicts
```

Create `src/llm_kg/evaluation/evaluator.py`:
```python
"""Evaluator abstract + extractive (EM/F1/recall@k) and generative (LLM judge) impls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from llm_kg.data.types import ScoredHit
from llm_kg.evaluation.judge import LLMJudge
from llm_kg.evaluation.metrics import exact_match, f1_score, recall_at_k


@dataclass
class Prediction:
    """One model prediction with everything an evaluator might need to score it."""

    qid: str
    question: str
    answer: str
    retrieved: list[ScoredHit] = field(default_factory=list)
    gold: list[str] | str | None = None
    relevant_ids: list[str] | None = None  # for retrieval recall@k
    metadata: dict = field(default_factory=dict)

    def gold_list(self) -> list[str]:
        if self.gold is None:
            return []
        if isinstance(self.gold, str):
            return [self.gold]
        return list(self.gold)


class Evaluator(ABC):
    @abstractmethod
    async def score(self, predictions: list[Prediction]) -> dict[str, float]: ...


class ExtractiveEvaluator(Evaluator):
    """Scores short-answer QA: aggregate EM, F1, and recall@k for each k in `recall_ks`."""

    def __init__(self, recall_ks: tuple[int, ...] = (1, 5, 10)) -> None:
        self.recall_ks = recall_ks

    async def score(self, predictions: list[Prediction]) -> dict[str, float]:
        if not predictions:
            return {"n": 0.0}
        em_total = 0.0
        f1_total = 0.0
        recall_totals: dict[int, float] = {k: 0.0 for k in self.recall_ks}
        recall_counts: dict[int, int] = {k: 0 for k in self.recall_ks}
        n = len(predictions)
        for p in predictions:
            golds = p.gold_list()
            em_total += exact_match(p.answer, golds)
            f1_total += f1_score(p.answer, golds)
            if p.relevant_ids is not None:
                retrieved_ids = [h.id for h in p.retrieved]
                for k in self.recall_ks:
                    recall_totals[k] += recall_at_k(retrieved_ids, p.relevant_ids, k)
                    recall_counts[k] += 1
        out: dict[str, float] = {
            "em": em_total / n,
            "f1": f1_total / n,
            "n": float(n),
        }
        for k in self.recall_ks:
            if recall_counts[k]:
                out[f"recall@{k}"] = recall_totals[k] / recall_counts[k]
        return out


class GenerativeEvaluator(Evaluator):
    """LightRAG-style head-to-head: each prediction vs a baseline on `axes`.

    `baseline_predictions` maps qid → baseline answer string. Predictions whose
    qid is not in the baseline map are skipped.
    """

    def __init__(
        self,
        judge: LLMJudge,
        baseline_predictions: dict[str, str],
        axes: Iterable[str] = (
            "comprehensiveness",
            "diversity",
            "empowerment",
            "overall",
        ),
    ) -> None:
        self._judge = judge
        self._baseline = baseline_predictions
        self._axes = list(axes)

    async def score(self, predictions: list[Prediction]) -> dict[str, float]:
        wins = {axis: 0 for axis in self._axes}
        n = 0
        for p in predictions:
            baseline = self._baseline.get(p.qid)
            if baseline is None:
                continue
            verdicts = await self._judge.judge(p.question, p.answer, baseline, self._axes)
            for axis, winner in verdicts.items():
                if winner == "candidate":
                    wins[axis] += 1
            n += 1
        out: dict[str, float] = {"n": float(n)}
        if n:
            for axis in self._axes:
                out[f"winrate_{axis}"] = wins[axis] / n
        return out
```

Replace `src/llm_kg/evaluation/__init__.py` with the full version:
```python
"""Evaluation: metrics, judges, evaluators."""

from llm_kg.evaluation.evaluator import (
    Evaluator,
    ExtractiveEvaluator,
    GenerativeEvaluator,
    Prediction,
)
from llm_kg.registry import Registry

EVALUATOR_REGISTRY: Registry[Evaluator] = Registry("evaluator")
EVALUATOR_REGISTRY.register("extractive")(ExtractiveEvaluator)
EVALUATOR_REGISTRY.register("generative")(GenerativeEvaluator)

__all__ = [
    "EVALUATOR_REGISTRY",
    "Evaluator",
    "ExtractiveEvaluator",
    "GenerativeEvaluator",
    "Prediction",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/evaluation/test_evaluator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/evaluation/__init__.py src/llm_kg/evaluation/evaluator.py src/llm_kg/evaluation/judge.py tests/evaluation/test_evaluator.py
git commit -m "feat: add Evaluator abstract + ExtractiveEvaluator + GenerativeEvaluator shell"
```

---

## Task 15: Logger (abstract + `NullLogger` + `WandbLogger`)

**Files:**
- Create: `src/llm_kg/logging_/__init__.py`
- Create: `src/llm_kg/logging_/base.py`
- Create: `src/llm_kg/logging_/wandb_logger.py`
- Create: `tests/logging_/__init__.py`
- Create: `tests/logging_/test_logger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/logging_/__init__.py`:
```
```

Create `tests/logging_/test_logger.py`:
```python
import pytest

from llm_kg.evaluation import Prediction
from llm_kg.logging_ import LOGGER_REGISTRY
from llm_kg.logging_.base import ExperimentLogger, NullLogger


def test_logger_is_abstract() -> None:
    with pytest.raises(TypeError):
        ExperimentLogger()  # type: ignore[abstract]


def test_logger_registry_has_null_and_wandb() -> None:
    assert "null" in LOGGER_REGISTRY
    assert "wandb" in LOGGER_REGISTRY


def test_null_logger_swallows_everything() -> None:
    log = NullLogger()
    log.init({"x": 1}, run_name="test")
    log.log_metrics({"em": 0.5})
    log.log_stage("Chunker", 0.01)
    log.log_predictions([Prediction(qid="q1", question="?", answer="a")])
    log.log_artifact("anything", {"k": "v"})
    log.finish()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/logging_/test_logger.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the loggers**

Create `src/llm_kg/logging_/__init__.py`:
```python
"""Experiment logging (W&B + null)."""

from llm_kg.logging_.base import ExperimentLogger, NullLogger
from llm_kg.logging_.wandb_logger import WandbLogger
from llm_kg.registry import Registry

LOGGER_REGISTRY: Registry[ExperimentLogger] = Registry("logger")
LOGGER_REGISTRY.register("null")(NullLogger)
LOGGER_REGISTRY.register("wandb")(WandbLogger)

__all__ = ["LOGGER_REGISTRY", "ExperimentLogger", "NullLogger", "WandbLogger"]
```

Create `src/llm_kg/logging_/base.py`:
```python
"""Abstract experiment logger and a no-op implementation for tests/offline runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_kg.evaluation import Prediction


class ExperimentLogger(ABC):
    """Abstract: tracks one experiment run (config, per-stage timing, predictions, metrics)."""

    @abstractmethod
    def init(self, config: dict[str, Any], run_name: str) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None: ...

    @abstractmethod
    def log_predictions(self, predictions: list[Prediction]) -> None: ...

    @abstractmethod
    def log_artifact(self, name: str, payload: Any) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class NullLogger(ExperimentLogger):
    """Swallows every call. Used by tests and `--no-log` runs."""

    def init(self, config: dict[str, Any], run_name: str) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None: ...
    def log_predictions(self, predictions: list[Prediction]) -> None: ...
    def log_artifact(self, name: str, payload: Any) -> None: ...
    def finish(self) -> None: ...
```

Create `src/llm_kg/logging_/wandb_logger.py`:
```python
"""Weights & Biases logger.

Imports `wandb` lazily so tests that don't use it (and CI without WANDB_API_KEY)
don't pay the import cost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm_kg.evaluation import Prediction
from llm_kg.logging_.base import ExperimentLogger


class WandbLogger(ExperimentLogger):
    """W&B implementation of ExperimentLogger.

    Stage durations are accumulated and flushed at `finish()` as a histogram
    rather than logged per-call (per-call would spam the run).
    """

    def __init__(self, project: str, run_name: str | None = None) -> None:
        self._project = project
        self._run_name = run_name
        self._run: Any = None
        self._stage_totals: dict[str, float] = defaultdict(float)
        self._stage_counts: dict[str, int] = defaultdict(int)

    def init(self, config: dict[str, Any], run_name: str) -> None:
        import wandb

        self._run = wandb.init(
            project=self._project,
            name=self._run_name or run_name,
            config=config,
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is None:
            return
        if step is None:
            self._run.log(metrics)
        else:
            self._run.log(metrics, step=step)

    def log_stage(self, stage: str, duration_s: float, **extras: Any) -> None:
        self._stage_totals[stage] += duration_s
        self._stage_counts[stage] += 1

    def log_predictions(self, predictions: list[Prediction]) -> None:
        if self._run is None:
            return
        import wandb

        rows = [
            [
                p.qid,
                p.question,
                p.answer,
                "; ".join(p.gold_list()),
                len(p.retrieved),
            ]
            for p in predictions
        ]
        table = wandb.Table(
            columns=["qid", "question", "answer", "gold", "n_retrieved"],
            data=rows,
        )
        self._run.log({"predictions": table})

    def log_artifact(self, name: str, payload: Any) -> None:
        if self._run is None:
            return
        self._run.log({name: payload})

    def finish(self) -> None:
        if self._run is None:
            return
        stage_metrics = {
            f"stage_time/{k}_total_s": v for k, v in self._stage_totals.items()
        }
        stage_metrics.update(
            {
                f"stage_time/{k}_avg_s": v / max(self._stage_counts[k], 1)
                for k, v in self._stage_totals.items()
            }
        )
        if stage_metrics:
            self._run.log(stage_metrics)
        self._run.finish()
        self._run = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/logging_/test_logger.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/logging_/ tests/logging_/
git commit -m "feat: add ExperimentLogger abstract + NullLogger + WandbLogger"
```

---

## Task 16: Config schema, secrets, YAML loader

**Files:**
- Create: `src/llm_kg/config/__init__.py`
- Create: `src/llm_kg/config/schema.py`
- Create: `src/llm_kg/config/settings.py`
- Create: `src/llm_kg/config/loader.py`
- Create: `tests/config/__init__.py`
- Create: `tests/config/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/config/__init__.py`:
```
```

Create `tests/config/test_config.py`:
```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_kg.config.loader import load_config
from llm_kg.config.schema import (
    DatasetConfig,
    EvaluatorConfig,
    LoggerConfig,
    MethodConfig,
    ProviderConfig,
    RootConfig,
    StorageConfig,
)


def _minimal_yaml() -> str:
    return """
method:
  name: naive_rag
  params:
    top_k: 5
llm:
  name: openai
  params: {model: gpt-4o-mini}
embedder:
  name: openai
  params: {model: text-embedding-3-small}
storage:
  vector:
    name: numpy
    params: {dim: 1536}
  graph:
    name: networkx
    params: {}
  kv:
    name: jsonfile
    params: {path: .cache/kv.json}
dataset:
  name: hotpotqa
  params: {split: dev}
evaluator:
  name: extractive
  params: {}
logger:
  name: "null"
  project: null
cache_dir: .cache
seed: 42
"""


def test_load_config_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(_minimal_yaml())
    cfg = load_config(p)
    assert isinstance(cfg, RootConfig)
    assert cfg.method.name == "naive_rag"
    assert cfg.method.params["top_k"] == 5
    assert cfg.llm.params["model"] == "gpt-4o-mini"
    assert cfg.storage.vector.name == "numpy"
    assert cfg.dataset.name == "hotpotqa"
    assert cfg.evaluator.name == "extractive"
    assert cfg.seed == 42


def test_load_config_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text("method: {name: x}\n")  # missing everything else
    with pytest.raises(ValidationError):
        load_config(p)


def test_provider_config_defaults() -> None:
    pc = ProviderConfig(name="openai")
    assert pc.params == {}
    assert pc.cache is True


def test_logger_config_defaults_to_wandb() -> None:
    lc = LoggerConfig()
    assert lc.name == "wandb"
    assert lc.project is None


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    from llm_kg.config.settings import Secrets

    s = Secrets()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.anthropic_api_key.get_secret_value() == "ant-test"


def test_storage_config_round_trip() -> None:
    sc = StorageConfig(
        vector=ProviderConfig(name="v"),
        graph=ProviderConfig(name="g"),
        kv=ProviderConfig(name="k"),
    )
    assert sc.vector.name == "v"


def test_method_dataset_evaluator_configs() -> None:
    assert MethodConfig(name="m").params == {}
    assert DatasetConfig(name="d").params == {}
    assert EvaluatorConfig(name="e").params == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_config.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement config**

Create `src/llm_kg/config/__init__.py`:
```python
"""Pydantic config schema, secrets, and YAML loader."""

from llm_kg.config.loader import load_config
from llm_kg.config.schema import RootConfig
from llm_kg.config.settings import Secrets

__all__ = ["RootConfig", "Secrets", "load_config"]
```

Create `src/llm_kg/config/schema.py`:
```python
"""Pydantic schema for experiment YAML configs.

The schema is intentionally permissive about per-plugin `params` blobs (just a
free-form dict) — each plug-in validates its own params on construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Names a provider (LLM, embedding, vector store, etc.) by registry key."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    cache: bool = True


class StorageConfig(BaseModel):
    vector: ProviderConfig
    graph: ProviderConfig
    kv: ProviderConfig


class MethodConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class LoggerConfig(BaseModel):
    name: str = "wandb"
    project: str | None = None
    run_name: str | None = None


class RootConfig(BaseModel):
    method: MethodConfig
    llm: ProviderConfig
    embedder: ProviderConfig
    storage: StorageConfig
    dataset: DatasetConfig
    evaluator: EvaluatorConfig
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    cache_dir: Path = Path(".cache")
    seed: int = 0
```

Create `src/llm_kg/config/settings.py`:
```python
"""Secrets loaded from environment variables. Never written to YAML."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """All API-key-style secrets the framework knows about.

    Add new keys here as you add providers. Loaded from env vars or .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    wandb_api_key: SecretStr = SecretStr("")
```

Create `src/llm_kg/config/loader.py`:
```python
"""YAML → RootConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_kg.config.schema import RootConfig


def load_config(path: Path | str) -> RootConfig:
    """Read a YAML file and validate it against RootConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ValueError(f"empty config: {path}")
    return RootConfig.model_validate(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_config.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/config/ tests/config/
git commit -m "feat: add pydantic config schema, env-based secrets, and YAML loader"
```

---

## Task 17: Experiment runner + CLI

**Files:**
- Create: `src/llm_kg/runner/__init__.py`
- Create: `src/llm_kg/runner/experiment.py`
- Create: `src/llm_kg/runner/cli.py`
- Create: `tests/runner/__init__.py`
- Create: `tests/runner/test_experiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/runner/__init__.py`:
```
```

Create `tests/runner/test_experiment.py`:
```python
"""Sanity tests for the runner — full E2E with stubs lives in test_smoke_e2e.py."""

from pathlib import Path

import pytest

from llm_kg.runner.experiment import Experiment


# NOTE: YAML unquoted `null` parses to None — use quoted "null" for the string registry name.
_NOPE_YAML = """
method: {name: nope, params: {}}
llm: {name: nope, params: {}}
embedder: {name: nope, params: {}}
storage:
  vector: {name: nope, params: {}}
  graph: {name: nope, params: {}}
  kv: {name: nope, params: {}}
dataset: {name: nope, params: {}}
evaluator: {name: extractive, params: {}}
logger: {name: "null"}
"""


def test_experiment_construct_only_loads_config(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(_NOPE_YAML)
    exp = Experiment(p)
    assert exp.config.method.name == "nope"


async def test_experiment_run_fails_for_unknown_plugin(tmp_path: Path) -> None:
    """The runner resolves plug-ins in order (LLM first); any unknown name raises KeyError."""
    p = tmp_path / "exp.yaml"
    p.write_text(_NOPE_YAML)
    with pytest.raises(KeyError, match="unknown llm"):
        await Experiment(p).run()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/runner/test_experiment.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement runner + CLI**

Create `src/llm_kg/runner/__init__.py`:
```python
"""Experiment runner and CLI entry point."""

from llm_kg.runner.experiment import Experiment

__all__ = ["Experiment"]
```

Create `src/llm_kg/runner/experiment.py`:
```python
"""Experiment: takes a YAML config, wires everything via registries, runs end-to-end."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from llm_kg.config import RootConfig, load_config
from llm_kg.data import DATASET_REGISTRY
from llm_kg.evaluation import EVALUATOR_REGISTRY, Prediction
from llm_kg.logging_ import LOGGER_REGISTRY
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import MethodConfig
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.cache import CachedEmbeddingProvider, CachedLLMProvider
from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY


class Experiment:
    """One end-to-end experiment run.

    Construction loads and validates the config but does not allocate any
    plug-ins. Call `run()` to actually instantiate everything and execute the
    pipeline.
    """

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config: RootConfig = load_config(self.config_path)

    async def run(self) -> dict[str, float]:
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # --- providers ---
        llm_cls = LLM_REGISTRY.get(cfg.llm.name)
        llm = llm_cls(**cfg.llm.params)
        if cfg.llm.cache:
            llm = CachedLLMProvider(
                inner=llm,
                model=str(cfg.llm.params.get("model", "default")),
                provider_name=cfg.llm.name,
                cache_dir=cfg.cache_dir,
            )

        embedder_cls = EMBEDDING_REGISTRY.get(cfg.embedder.name)
        embedder = embedder_cls(**cfg.embedder.params)
        if cfg.embedder.cache:
            embedder = CachedEmbeddingProvider(
                inner=embedder,
                model=str(cfg.embedder.params.get("model", "default")),
                provider_name=cfg.embedder.name,
                cache_dir=cfg.cache_dir,
            )

        # --- storage ---
        vec_store = VECTOR_REGISTRY.get(cfg.storage.vector.name)(**cfg.storage.vector.params)
        graph_store = GRAPH_REGISTRY.get(cfg.storage.graph.name)(**cfg.storage.graph.params)
        kv_store = KV_REGISTRY.get(cfg.storage.kv.name)(**cfg.storage.kv.params)

        # --- logger ---
        logger_kwargs: dict[str, Any] = {}
        if cfg.logger.project is not None:
            logger_kwargs["project"] = cfg.logger.project
        if cfg.logger.run_name is not None:
            logger_kwargs["run_name"] = cfg.logger.run_name
        logger = LOGGER_REGISTRY.get(cfg.logger.name)(**logger_kwargs)
        run_name = cfg.logger.run_name or f"{cfg.method.name}-{cfg.dataset.name}"
        logger.init(cfg.model_dump(mode="json"), run_name=run_name)

        # --- dataset & method ---
        dataset = DATASET_REGISTRY.get(cfg.dataset.name)(**cfg.dataset.params)
        method = METHOD_REGISTRY.get(cfg.method.name)()

        ctx = PipelineContext(
            llm=llm,
            embedder=embedder,
            vector_store=vec_store,
            graph_store=graph_store,
            kv_store=kv_store,
            logger=logger,
            trace={},
        )
        indexing, query = method.build(MethodConfig(**cfg.method.model_dump()), ctx)

        # --- index ---
        corpus = dataset.corpus()
        await indexing.run(list(corpus.documents()), ctx)

        # --- query each example ---
        predictions: list[Prediction] = []
        for ex in dataset.examples():
            ctx.query = ex.question  # so the Generator can read the question
            answer = await query.run(ex.question, ctx)
            predictions.append(
                Prediction(
                    qid=ex.id,
                    question=ex.question,
                    answer=answer if isinstance(answer, str) else str(answer),
                    gold=ex.answers if ex.answers else ex.long_answer,
                )
            )

        # --- evaluate & log ---
        evaluator = EVALUATOR_REGISTRY.get(cfg.evaluator.name)(**cfg.evaluator.params)
        metrics = await evaluator.score(predictions)

        logger.log_metrics(metrics)
        logger.log_predictions(predictions)
        logger.finish()

        return metrics
```

Create `src/llm_kg/runner/cli.py`:
```python
"""CLI: `python -m llm_kg run path/to/config.yaml`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_kg.runner.experiment import Experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-kg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("config", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        metrics = asyncio.run(Experiment(args.config).run())
        console = Console()
        table = Table(title=f"Results: {args.config}")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for k, v in sorted(metrics.items()):
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

Create `src/llm_kg/__main__.py`:
```python
from llm_kg.runner.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/runner/test_experiment.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_kg/runner/ src/llm_kg/__main__.py tests/runner/
git commit -m "feat: add Experiment runner and CLI entry point"
```

---

## Task 18: End-to-end smoke test with fakes

**Files:**
- Create: `tests/test_smoke_e2e.py`

This test wires fake providers, fake storage, a fake dataset, and a fake method into a real `Experiment` and asserts the whole thing runs.

- [ ] **Step 1: Write the smoke test**

Create `tests/test_smoke_e2e.py`:
```python
"""End-to-end smoke test: stub everything that needs a network or disk-heavy
backend, then run a real Experiment through the real runner."""

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data.corpus import Corpus
from llm_kg.data.dataset import QADataset
from llm_kg.data.types import Document, QAExample, ScoredHit
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.methods.base import Method
from llm_kg.pipeline.indexing import IndexingPipeline
from llm_kg.pipeline.query import QueryPipeline
from llm_kg.pipeline.stages.chunker import DefaultChunker
from llm_kg.pipeline.stages.context_builder import DefaultContextBuilder
from llm_kg.pipeline.stages.embedder import DefaultEmbedder
from llm_kg.pipeline.stages.extractor import NoOpExtractor
from llm_kg.pipeline.stages.generator import DefaultGenerator
from llm_kg.pipeline.stages.graph_builder import NoOpGraphBuilder
from llm_kg.pipeline.stages.query_processor import IdentityQueryProcessor
from llm_kg.pipeline.stages.retriever import Retriever
from llm_kg.providers import EMBEDDING_REGISTRY, LLM_REGISTRY
from llm_kg.providers.base import EmbeddingProvider, LLMProvider, LLMResponse
from llm_kg.runner.experiment import Experiment
from llm_kg.storage import GRAPH_REGISTRY, KV_REGISTRY, VECTOR_REGISTRY
from llm_kg.storage.base import GraphStore, KVStore, VectorStore


# ---------- fake providers ----------

@LLM_REGISTRY.register("fake_llm")
class FakeLLM(LLMProvider):
    def __init__(self, model: str = "fake") -> None:
        self.model = model

    async def generate(self, prompt: str, **kwargs):
        return LLMResponse(text="paris", prompt_tokens=1, completion_tokens=1)

    async def generate_structured(self, prompt, schema, **kwargs):
        raise NotImplementedError


@EMBEDDING_REGISTRY.register("fake_embed")
class FakeEmbedder(EmbeddingProvider):
    def __init__(self, model: str = "fake", dim: int = 4) -> None:
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts):
        # deterministic hash-based vectors
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")[: self._dim]):
                out[i, j] = float(ch) / 255.0
        return out


# ---------- fake storage ----------

@VECTOR_REGISTRY.register("fake_vec")
class FakeVecStore(VectorStore):
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vecs: np.ndarray | None = None
        self.meta: list[dict] = []

    def upsert(self, ids, vectors, meta):
        self.ids.extend(ids)
        self.meta.extend(meta)
        self.vecs = vectors if self.vecs is None else np.concatenate([self.vecs, vectors], axis=0)

    def search(self, query, k, filter=None):
        if self.vecs is None or len(self.ids) == 0:
            return []
        # cosine-ish: dot product with normalized
        scores = self.vecs @ query
        top = np.argsort(-scores)[:k]
        return [ScoredHit(id=self.ids[i], score=float(scores[i]), meta=self.meta[i]) for i in top]


@GRAPH_REGISTRY.register("fake_graph")
class FakeGraphStore(GraphStore):
    def __init__(self) -> None:
        self._adj: dict[str, list[str]] = {}

    def add_node(self, id, **attrs):
        self._adj.setdefault(id, [])

    def add_edge(self, src, dst, **attrs):
        self._adj.setdefault(src, []).append(dst)

    def neighbors(self, id):
        return list(self._adj.get(id, []))

    def personalized_pagerank(self, seeds, **kwargs):
        return dict.fromkeys(seeds, 1.0)


@KV_REGISTRY.register("fake_kv")
class FakeKV(KVStore):
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key):
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value

    def __contains__(self, key):
        return key in self._d


# ---------- fake retriever, fake method, fake dataset ----------

class FakeRetriever(Retriever):
    name = "FakeRetriever"

    async def run(self, inp, ctx):
        if ctx.embedder is None or ctx.vector_store is None:
            return []
        qvec = (await ctx.embedder.embed([inp.text]))[0]
        return ctx.vector_store.search(qvec, k=3)


class _ChunkPersistingEmbedder(DefaultEmbedder):
    """Embedder that also stuffs each chunk's text into the KV store."""

    name = "_ChunkPersistingEmbedder"

    async def run(self, inp, ctx):
        for c in inp.chunks:
            ctx.kv_store.put(c.id, c.text)
        return await super().run(inp, ctx)


@METHOD_REGISTRY.register("fake_method")
class FakeMethod(Method):
    name = "fake_method"

    def build(self, cfg, ctx):
        indexing = IndexingPipeline(
            stages=[
                DefaultChunker(chunk_size=4, chunk_overlap=0),
                NoOpExtractor(),
                NoOpGraphBuilder(),
                _ChunkPersistingEmbedder(),
            ]
        )
        query = QueryPipeline(
            stages=[
                IdentityQueryProcessor(),
                FakeRetriever(),
                DefaultContextBuilder(),
                DefaultGenerator(),
            ]
        )
        return indexing, query


@DATASET_REGISTRY.register("fake_ds")
class FakeDataset(QADataset):
    def __init__(self) -> None:
        self._docs = [
            Document(id="d1", text="The capital of France is Paris."),
            Document(id="d2", text="Berlin is the capital of Germany."),
        ]
        self._ex = [QAExample(id="q1", question="capital of france?", answers=["paris"])]

    def examples(self) -> Iterable[QAExample]:
        return iter(self._ex)

    def corpus(self) -> Corpus:
        docs = self._docs

        class _C(Corpus):
            def documents(self):
                return iter(docs)

            def __len__(self):
                return len(docs)

        return _C()

    def __len__(self):
        return len(self._ex)


# ---------- the test ----------

async def test_smoke_e2e_pipeline_runs(tmp_path: Path) -> None:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
method: {name: fake_method, params: {}}
llm: {name: fake_llm, params: {model: fake}, cache: false}
embedder: {name: fake_embed, params: {model: fake, dim: 4}, cache: false}
storage:
  vector: {name: fake_vec, params: {}}
  graph: {name: fake_graph, params: {}}
  kv: {name: fake_kv, params: {}}
dataset: {name: fake_ds, params: {}}
evaluator: {name: extractive, params: {}}
logger: {name: "null"}
cache_dir: """ + str(tmp_path / ".cache") + """
seed: 0
"""
    )
    metrics = await Experiment(cfg).run()
    # FakeLLM always answers "paris" → EM = 1 for the one question.
    assert metrics["em"] == 1.0
    assert metrics["n"] == 1.0
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/test_smoke_e2e.py -v`
Expected: 1 passed. If a registry-conflict error fires (because fakes were registered in another test file too), refactor that conftest first — but on a clean tree this should pass.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: every test from every previous task plus this one passes. Around 60 tests total.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_e2e.py
git commit -m "test: add end-to-end smoke test with fake providers/storage/method/dataset"
```

---

## Task 19: Example YAML configs (stubs)

**Files:**
- Create: `configs/_base.yaml`
- Create: `configs/examples/naive_rag.yaml`
- Create: `configs/examples/hipporag.yaml`
- Create: `configs/examples/lightrag.yaml`

These are templates that reference not-yet-existing registry names — they exist to (1) document the YAML shape, (2) be the starting point for future PRs that add real plug-ins.

- [ ] **Step 1: Create the base config**

Create `configs/_base.yaml`:
```yaml
# Defaults shared by every experiment. Individual configs override fields below.
# NOTE: Plugin names below ("openai", "numpy", "networkx", …) refer to registry
# entries that are NOT YET implemented — these YAMLs are templates for upcoming
# PRs. Running `llm-kg run configs/examples/naive_rag.yaml` against the day-1
# tree will fail with KeyError("unknown llm 'openai'"), which is the expected
# state until provider implementations land.

llm:
  name: openai
  params:
    model: gpt-4o-mini
    temperature: 0.0
  cache: true

embedder:
  name: openai
  params:
    model: text-embedding-3-small
  cache: true

storage:
  vector:
    name: numpy
    params: {}
  graph:
    name: networkx
    params: {}
  kv:
    name: jsonfile
    params:
      path: .cache/kv.json

logger:
  name: wandb
  project: llm-kg-research

cache_dir: .cache
seed: 0
```

- [ ] **Step 2: Create the naive RAG config**

Create `configs/examples/naive_rag.yaml`:
```yaml
# Naive RAG baseline: chunk → embed → vector search → LLM.

method:
  name: naive_rag
  params:
    chunk_size: 256
    chunk_overlap: 32
    top_k: 5

llm:
  name: openai
  params:
    model: gpt-4o-mini

embedder:
  name: openai
  params:
    model: text-embedding-3-small

storage:
  vector: {name: numpy, params: {}}
  graph: {name: networkx, params: {}}
  kv: {name: jsonfile, params: {path: .cache/naive/kv.json}}

dataset:
  name: hotpotqa
  params:
    split: dev
    n_examples: 100

evaluator:
  name: extractive
  params:
    recall_ks: [1, 5, 10]

logger:
  name: wandb
  project: llm-kg-research
  run_name: naive-hotpot

cache_dir: .cache/naive
seed: 0
```

- [ ] **Step 3: Create the HippoRAG config**

Create `configs/examples/hipporag.yaml`:
```yaml
# HippoRAG: OpenIE entity extraction → graph → query entities → PPR → top passages.

method:
  name: hipporag
  params:
    chunk_size: 256
    top_k: 5
    ppr_alpha: 0.5
    synonym_threshold: 0.85

llm:
  name: openai
  params:
    model: gpt-4o-mini

embedder:
  name: openai
  params:
    model: text-embedding-3-small

storage:
  vector: {name: numpy, params: {}}
  graph: {name: networkx, params: {}}
  kv: {name: jsonfile, params: {path: .cache/hippo/kv.json}}

dataset:
  name: musique
  params:
    split: dev
    n_examples: 100

evaluator:
  name: extractive
  params:
    recall_ks: [1, 5, 10]

logger:
  name: wandb
  project: llm-kg-research
  run_name: hippo-musique

cache_dir: .cache/hippo
seed: 0
```

- [ ] **Step 4: Create the LightRAG config**

Create `configs/examples/lightrag.yaml`:
```yaml
# LightRAG: dual-level (entity + concept) extraction → graph + chunk index →
# dual-level keyword query → graph walk → generate.

method:
  name: lightrag
  params:
    chunk_size: 1200
    chunk_overlap: 100
    top_k_entities: 60
    top_k_chunks: 5

llm:
  name: openai
  params:
    model: gpt-4o-mini

embedder:
  name: openai
  params:
    model: text-embedding-3-small

storage:
  vector: {name: numpy, params: {}}
  graph: {name: networkx, params: {}}
  kv: {name: jsonfile, params: {path: .cache/light/kv.json}}

dataset:
  name: ultradomain
  params:
    domain: cs
    n_examples: 50

evaluator:
  name: generative
  params:
    axes: [comprehensiveness, diversity, empowerment, overall]
    baseline_path: results/naive-cs/predictions.json

logger:
  name: wandb
  project: llm-kg-research
  run_name: light-cs

cache_dir: .cache/light
seed: 0
```

- [ ] **Step 5: Verify they parse (will fail at registry lookup, not at YAML/pydantic)**

Run:
```bash
uv run python -c "from llm_kg.config import load_config; load_config('configs/examples/naive_rag.yaml'); load_config('configs/examples/hipporag.yaml'); load_config('configs/examples/lightrag.yaml'); print('all parsed ok')"
```
Expected: `all parsed ok`.

- [ ] **Step 6: Commit**

```bash
git add configs/
git commit -m "docs: add example YAML configs for naive_rag, hipporag, lightrag"
```

---

## Task 20: Final verification — full test suite green

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -v`
Expected: every test from tasks 2–18 passes. Around 60 tests, ~1–2 seconds.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src tests`
Expected: `All checks passed!` (or fix any reported issues).

- [ ] **Step 3: Verify CLI loads**

Run: `uv run llm-kg --help`
Expected: argparse usage message printed; exit 0.

- [ ] **Step 4: Sanity-check the package can import cleanly**

Run:
```bash
uv run python -c "
import llm_kg
from llm_kg.providers import LLM_REGISTRY, EMBEDDING_REGISTRY
from llm_kg.storage import VECTOR_REGISTRY, GRAPH_REGISTRY, KV_REGISTRY
from llm_kg.methods import METHOD_REGISTRY
from llm_kg.data import DATASET_REGISTRY
from llm_kg.evaluation import EVALUATOR_REGISTRY
from llm_kg.logging_ import LOGGER_REGISTRY
print('llm_kg', llm_kg.__version__)
print('llm:', LLM_REGISTRY.names())
print('embedder:', EMBEDDING_REGISTRY.names())
print('vector:', VECTOR_REGISTRY.names())
print('graph:', GRAPH_REGISTRY.names())
print('kv:', KV_REGISTRY.names())
print('method:', METHOD_REGISTRY.names())
print('dataset:', DATASET_REGISTRY.names())
print('evaluator:', EVALUATOR_REGISTRY.names())
print('logger:', LOGGER_REGISTRY.names())
"
```
Expected: prints version `0.0.1`; the LLM/embedder/vector/graph/kv/method/dataset registries print as empty `[]`; evaluator prints `['extractive', 'generative']`; logger prints `['null', 'wandb']`. Day 1 ships zero domain plug-ins, by design.

- [ ] **Step 5: Final commit (if anything changed during verification)**

If lint or import surfaced any fixes:
```bash
git add -A
git commit -m "chore: post-verification fixes"
```

Otherwise nothing to do — the plan is complete.
