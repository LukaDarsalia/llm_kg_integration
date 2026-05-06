# LLM × KG Pipeline Framework — Design

**Date:** 2026-05-06
**Status:** Approved (design phase)
**Scope:** Day-1 framework only — no method, provider, storage, or dataset implementations.

## 1. Purpose

Build a research framework for evaluating and extending KG-augmented RAG methods. The framework lets us:

- Reproduce published baselines (Naive RAG, HippoRAG, LightRAG) under a single common interface.
- Run controlled ablations by swapping individual pipeline stages between methods.
- Add new methods cheaply by overriding only the stages that differ from the defaults.
- Sweep across LLMs, embedding models, and storage backends without touching method code.

The day-1 deliverable is the **scaffolding**: abstract interfaces, registries, the runner, the logger, the config system, and a smoke test that proves the whole thing composes end-to-end with stubs. Real method/provider/storage implementations are explicitly out of scope and land in subsequent PRs.

## 2. Design decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Reimplement methods against shared interfaces (port logic from upstream) | True modularity (swap any stage). One template can't cleanly wrap two divergent codebases. |
| 2 | Support both extractive (EM/F1) and generative (LLM-judge) evaluation | HippoRAG-style and LightRAG-style benchmarks are both required for full comparison. |
| 3 | LLM and Embedding both abstract + registry-backed | Embedding choice is a research variable; symmetry with LLM is the right shape. |
| 4 | Fine-grained pipeline stages with optional overrides + sensible defaults | A method only writes the stages it actually changes. |
| 5 | Pydantic + YAML for config (no Hydra) | Type safety, no magic. Sweeps not yet a first-class concern. |
| 6 | Disk cache for all LLM/embedding calls (transparent) | Re-running the same pipeline must not re-pay model costs. |
| 7 | W&B logging with `NullLogger` fallback | Standard for research; tests don't need W&B. |
| 8 | uv + `pyproject.toml` | User requirement. |

## 3. Architecture

### 3.1 Data flow

Two pipelines (indexing offline, query online) communicate through a storage layer. Everything else (providers, datasets, evaluator, logger) is cross-cutting plumbing wired in by the `Experiment` runner.

```
                    ┌─────────────────────────────┐
                    │     Experiment Config       │
                    │   (pydantic, loaded YAML)   │
                    └──────────────┬──────────────┘
                                   │ resolves names → registries
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   Providers   │          │    Method     │          │   Datasets    │
│  LLM / Embed  │          │   (registry)  │          │  (registry)   │
│  (registry,   │          │               │          │ corpus + qa   │
│  disk-cached) │          │ ─ assembles ─►│          └───────┬───────┘
└───────┬───────┘          │  Indexing &   │                  │
        │                  │  Query stages │                  │
        │                  └───────┬───────┘                  │
        │                          │                          │
        │      ┌───────────────────┴─────────────────┐        │
        │      ▼                                     ▼        │
        │  ┌───────────────────────┐    ┌──────────────────┐  │
        │  │  Indexing Pipeline    │    │ Query Pipeline   │  │
        │  │  Chunker              │    │ QueryProcessor   │◄─┤ (questions)
        │  │   ↓                   │    │   ↓              │  │
        │  │  InfoExtractor (opt)  │    │ Retriever        │  │
        │  │   ↓                   │    │   ↓              │  │
        │  │  GraphBuilder (opt)   │    │ ContextBuilder   │  │
        │  │   ↓                   │    │   ↓              │  │
        │  │  Embedder             │    │ Generator        │  │
        │  │                       │    │   ↓              │  │
        │  │  (stages write to ───►│    │ (Retriever reads │  │
        │  │   storage via ctx)    │    │  storage via ctx)│  │
        │  └───────────────────────┘    └────────┬─────────┘  │
        │                                        │            │
        │              ┌──────────────────┐      ▼            │
        ▼              │  Storage Layer   │  Predictions      │
   (any stage that  ──►│ Vector / Graph / │      │            │
   needs a model)      │  KV (registry)   │      ▼            │
                       └──────────────────┘  ┌──────────────┐ │
                                             │  Evaluator   │◄┘
                                             │ EM/F1 │ Judge│
                                             └──────┬───────┘
                                                    │
                                                    ▼
                                             ┌──────────────┐
                                             │  W&B Logger  │
                                             │ config/preds │
                                             │ metrics/cost │
                                             └──────────────┘
```

### 3.2 Repository layout

```
llm_kg_integration/
├── pyproject.toml                      # uv-managed
├── uv.lock
├── README.md
├── configs/
│   ├── _base.yaml
│   └── examples/                       # naive_rag.yaml, hipporag.yaml, lightrag.yaml (stubs)
├── src/llm_kg/
│   ├── registry.py                     # generic Registry[T]
│   │
│   ├── providers/
│   │   ├── base.py                     # LLMProvider, EmbeddingProvider abstracts
│   │   ├── cache.py                    # transparent disk cache wrapper
│   │   └── __init__.py                 # LLM_REGISTRY, EMBEDDING_REGISTRY
│   │
│   ├── storage/
│   │   ├── base.py                     # VectorStore, GraphStore, KVStore abstracts
│   │   └── __init__.py                 # VECTOR_REGISTRY, GRAPH_REGISTRY, KV_REGISTRY
│   │
│   ├── data/
│   │   ├── types.py                    # Document, Chunk, QAExample dataclasses
│   │   ├── corpus.py                   # Corpus / DocumentLoader abstract
│   │   ├── dataset.py                  # QADataset abstract
│   │   └── __init__.py                 # DATASET_REGISTRY
│   │
│   ├── pipeline/
│   │   ├── stage.py                    # Stage[InT, OutT] base + PipelineContext
│   │   ├── indexing.py                 # IndexingPipeline
│   │   ├── query.py                    # QueryPipeline
│   │   └── stages/
│   │       ├── chunker.py              # Chunker abstract + DefaultChunker
│   │       ├── extractor.py            # InformationExtractor abstract + NoOpExtractor
│   │       ├── graph_builder.py        # GraphBuilder abstract + NoOpGraphBuilder
│   │       ├── embedder.py             # Embedder abstract + DefaultEmbedder
│   │       ├── query_processor.py      # QueryProcessor abstract + IdentityProcessor
│   │       ├── retriever.py            # Retriever abstract (no default — required)
│   │       ├── context_builder.py      # ContextBuilder abstract + DefaultContextBuilder
│   │       └── generator.py            # Generator abstract + DefaultGenerator
│   │
│   ├── methods/
│   │   ├── base.py                     # Method abstract: build() → (Indexing, Query)
│   │   └── __init__.py                 # METHOD_REGISTRY (empty day 1)
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # EM, F1, recall@k (real implementations)
│   │   ├── judge.py                    # LLM-as-judge abstract + LightRAG-style impl shell
│   │   ├── evaluator.py                # Evaluator abstract + ExtractiveEvaluator + GenerativeEvaluator
│   │   └── __init__.py                 # EVALUATOR_REGISTRY
│   │
│   ├── logging_/                       # trailing _ avoids stdlib clash
│   │   ├── base.py                     # ExperimentLogger abstract + NullLogger
│   │   └── wandb_logger.py             # W&B impl
│   │
│   ├── config/
│   │   ├── schema.py                   # pydantic models (RootConfig, MethodConfig, …)
│   │   ├── settings.py                 # pydantic-settings for API keys (env vars)
│   │   └── loader.py                   # YAML → RootConfig
│   │
│   └── runner/
│       ├── experiment.py               # Experiment.run()
│       └── cli.py                      # `python -m llm_kg run configs/foo.yaml`
│
└── tests/
    ├── conftest.py                     # shared fakes
    ├── test_registry.py
    ├── test_cache.py
    ├── test_pipeline_compose.py
    └── test_smoke_e2e.py               # fake provider/storage → indexing → query → eval → log
```

## 4. Core abstractions

### 4.1 Registry

Generic, decorator-based, used by every plug-in family.

```python
T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, kind: str): ...
    def register(self, name: str) -> Callable[[type[T]], type[T]]: ...
    def get(self, name: str) -> type[T]: ...
    def names(self) -> list[str]: ...
```

Used as:
```python
LLM_REGISTRY: Registry[LLMProvider] = Registry("llm")

@LLM_REGISTRY.register("openai")
class OpenAIProvider(LLMProvider): ...
```

YAML refers to plug-ins by name: `llm: {name: openai, model: gpt-4o-mini}`.

### 4.2 Providers

```python
@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None = None
    raw: dict | None = None

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...

    @abstractmethod
    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], **kwargs
    ) -> BaseModel: ...

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:  # shape (N, dim)
        ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
```

`CachedProvider` wraps any provider. Cache key = SHA-256 of `(provider_name, model, prompt, sorted_kwargs)`. Cache value = serialized response. On-disk path under `.cache/<llm|embed>/<sha>.json`. Cache is enabled by default and can be disabled in config.

### 4.3 Storage

```python
@dataclass
class ScoredHit:
    id: str
    score: float
    meta: dict

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, meta: list[dict]) -> None: ...
    @abstractmethod
    def search(self, query: np.ndarray, k: int, filter: dict | None = None) -> list[ScoredHit]: ...

class GraphStore(ABC):
    @abstractmethod
    def add_node(self, id: str, **attrs) -> None: ...
    @abstractmethod
    def add_edge(self, src: str, dst: str, **attrs) -> None: ...
    @abstractmethod
    def neighbors(self, id: str) -> list[str]: ...
    @abstractmethod
    def personalized_pagerank(self, seeds: dict[str, float], **kwargs) -> dict[str, float]: ...

class KVStore(ABC):
    @abstractmethod
    def get(self, key: str) -> Any: ...
    @abstractmethod
    def put(self, key: str, value: Any) -> None: ...
```

### 4.4 Pipeline stages

```python
InT = TypeVar("InT")
OutT = TypeVar("OutT")

@dataclass
class PipelineContext:
    llm: LLMProvider
    embedder: EmbeddingProvider
    vector_store: VectorStore
    graph_store: GraphStore
    kv_store: KVStore
    logger: ExperimentLogger
    trace: dict[str, Any]      # per-run telemetry written by stages

class Stage(ABC, Generic[InT, OutT]):
    @abstractmethod
    async def run(self, inp: InT, ctx: PipelineContext) -> OutT: ...
```

Stage-specific abstracts (`Chunker(Stage[Document, list[Chunk]])`, `Retriever(Stage[ProcessedQuery, list[ScoredHit]])`, etc.) live under `pipeline/stages/`. Each one provides a default implementation where defaulting makes sense; `Retriever` has no default since every method must define one.

`IndexingPipeline` and `QueryPipeline` are sequential composers that pass values from one stage's output to the next stage's input and call `logger.log_stage(name, duration, …)` around each stage. Type compatibility between adjacent stages is the responsibility of the `Method` that assembles them — pipelines themselves do no static enforcement beyond runtime errors.

Storage is **not** a pipeline stage. Stages that need to persist (Embedder writes vectors; GraphBuilder writes nodes/edges; Chunker may write chunks to KV) do so directly via `PipelineContext`. Likewise the `Retriever` reads from storage via `ctx`. This keeps pipelines focused on data transformation; persistence is a side-effect of the stages that own the data.

### 4.5 Method

```python
class Method(ABC):
    name: ClassVar[str]

    @abstractmethod
    def build(
        self, cfg: MethodConfig, ctx: PipelineContext
    ) -> tuple[IndexingPipeline, QueryPipeline]: ...
```

A `Method` selects which stage implementation goes into each slot. Day 1 the registry is empty; in subsequent PRs `NaiveRAG`, `HippoRAG`, `LightRAG` register here.

Composition example (illustrative — not part of day-1 code):
```python
@METHOD_REGISTRY.register("lightrag")
class LightRAG(Method):
    name = "lightrag"
    def build(self, cfg, ctx):
        return (
            IndexingPipeline([
                DefaultChunker(cfg.chunker),
                LightRAGExtractor(cfg.extractor),
                LightRAGGraphBuilder(cfg.graph),
                DefaultEmbedder(cfg.embedder),
            ]),
            QueryPipeline([
                LightRAGQueryProcessor(cfg.qproc),
                LightRAGRetriever(cfg.retriever),
                DefaultContextBuilder(cfg.ctx),
                DefaultGenerator(cfg.gen),
            ]),
        )
```

### 4.6 Evaluator

```python
@dataclass
class Prediction:
    qid: str
    question: str
    answer: str
    retrieved: list[ScoredHit]
    gold: str | list[str] | None

class Evaluator(ABC):
    @abstractmethod
    async def score(self, predictions: list[Prediction]) -> dict[str, float]: ...
```

`ExtractiveEvaluator` computes EM, F1, recall@k from `gold`. `GenerativeEvaluator` runs an `LLMJudge` head-to-head against a baseline's predictions and reports win rates on configurable axes (defaults: comprehensiveness, diversity, empowerment, overall).

### 4.7 Logger

```python
class ExperimentLogger(ABC):
    @abstractmethod
    def init(self, config: dict, run_name: str) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    @abstractmethod
    def log_stage(self, stage: str, duration_s: float, **extras) -> None: ...
    @abstractmethod
    def log_predictions(self, predictions: list[Prediction]) -> None: ...
    @abstractmethod
    def log_artifact(self, name: str, payload: Any) -> None: ...
    @abstractmethod
    def finish(self) -> None: ...
```

`WandbLogger` is the only real implementation on day 1. `NullLogger` swallows everything (used by tests and offline runs).

### 4.8 Config

```python
class ProviderConfig(BaseModel):
    name: str            # registry key
    params: dict = {}    # provider-specific kwargs
    cache: bool = True

class MethodConfig(BaseModel):
    name: str            # registry key
    params: dict = {}    # method-specific kwargs (passed through to stages)

class StorageConfig(BaseModel):
    vector: ProviderConfig
    graph: ProviderConfig
    kv: ProviderConfig

class DatasetConfig(BaseModel):
    name: str
    params: dict = {}

class EvaluatorConfig(BaseModel):
    name: str            # "extractive" | "generative"
    params: dict = {}

class LoggerConfig(BaseModel):
    name: str = "wandb"  # or "null"
    project: str | None = None
    run_name: str | None = None

class RootConfig(BaseModel):
    method: MethodConfig
    llm: ProviderConfig
    embedder: ProviderConfig
    storage: StorageConfig
    dataset: DatasetConfig
    evaluator: EvaluatorConfig
    logger: LoggerConfig = LoggerConfig()
    cache_dir: Path = Path(".cache")
    seed: int = 0
```

Secrets (API keys) are loaded via `pydantic-settings` from environment variables, never written to YAML.

### 4.9 Runner

```python
class Experiment:
    def __init__(self, config_path: Path): ...
    async def run(self) -> dict[str, float]: ...
```

`run()` executes:
1. Load YAML → `RootConfig`.
2. Resolve registry names → instantiate providers, storage, dataset, evaluator, logger, method.
3. Build `PipelineContext`.
4. Build `(IndexingPipeline, QueryPipeline)` from method.
5. Run indexing over corpus.
6. Run query pipeline over each QA example, collect `Prediction`s.
7. Score with evaluator.
8. Log everything to W&B; return metrics.

CLI: `python -m llm_kg run configs/examples/naive_rag.yaml`.

## 5. Day-1 deliverables

**In scope:**

- `pyproject.toml` (uv) with deps: `pydantic`, `pydantic-settings`, `numpy`, `pyyaml`, `wandb`, `networkx`, `aiofiles`, `rich`, `pytest`, `pytest-asyncio`.
- `Registry[T]` + tests.
- All abstract bases and dataclasses listed in section 4.
- Default stage implementations: `DefaultChunker`, `NoOpExtractor`, `NoOpGraphBuilder`, `DefaultEmbedder` (just calls `EmbeddingProvider`), `IdentityProcessor`, `DefaultContextBuilder`, `DefaultGenerator` (just calls `LLMProvider`).
- `CachedProvider` wrapper + cache-key tests + cache-hit tests.
- Pydantic config schema + YAML loader + secrets via `pydantic-settings`.
- `Experiment` runner + CLI entry point.
- `WandbLogger` + `NullLogger`.
- **Real implementations** of EM, F1, recall@k metrics (small enough to include).
- `Evaluator` abstracts + `ExtractiveEvaluator` (working, since metrics are real). `GenerativeEvaluator` is a shell — the abstract is concrete but the judge prompt is a stub that calls the LLM with a placeholder prompt.
- One end-to-end **smoke test** that uses fake `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `GraphStore`, `KVStore` and a fake `Method` to drive indexing → query → eval → log all the way through, asserting the wiring composes.
- Example YAML configs (`naive_rag.yaml`, `hipporag.yaml`, `lightrag.yaml`) referencing not-yet-existing registry names, kept as stubs so the structure is visible.

**Out of scope (later PRs, in roughly this order):**

1. One real `LLMProvider` (suggest: OpenAI) + one real `EmbeddingProvider`.
2. One real `VectorStore` (suggest: in-memory NumPy or NanoVectorDB) + `GraphStore` (NetworkX) + `KVStore` (JSON file).
3. One real dataset loader + the Naive RAG method (smallest end-to-end real run).
4. HippoRAG method.
5. LightRAG method.
6. `GenerativeEvaluator` real prompt, additional providers, additional storage backends.

## 6. Non-goals

- Distributed execution. Single-machine for now.
- Hyperparameter sweep orchestration. Re-evaluate when needed.
- Wrapping upstream HippoRAG/LightRAG repos. We port logic into our interfaces.
- A frontend or interactive playground.

## 7. Open questions deferred to implementation

- Async vs sync at the stage boundary: design is async. If stages turn out to mostly be CPU-bound, we may relax to sync with `asyncio.to_thread` on provider boundaries only.
- Cache invalidation policy: day 1 caches forever, keyed by content. Add TTL/version key if it bites.
- Whether `PipelineContext.trace` should be append-only structured events vs free-form dict. Starting free-form; tighten later.
