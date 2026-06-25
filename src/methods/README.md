# Methods

A **method** is a GraphRAG approach: an *indexer* (builds a knowledge structure from a
corpus) and a *retriever* (uses that structure to answer questions). Methods are
pluggable — the indexer and evaluator stages are method-agnostic and select one via
`--method <name>`.

```
src/methods/
├── registry.py          # name -> (Indexer, Retriever); get_method() lazily imports
├── lightrag/            # reference method (official lightrag-hku SDK)
│   ├── __init__.py      # register_method("lightrag", ...)
│   ├── models.py        # providers.yaml -> LightRAG llm_model_func / embedding_func
│   ├── data_indexer.py  # LightRAGIndexer(BaseIndexer)
│   └── data_retriever.py# LightRAGRetriever(BaseRetriever)
└── customrag/           # copy-me template (registers, raises NotImplementedError)
```

## The contract

Both classes live in [../pipeline/shared/contracts.py](../pipeline/shared/contracts.py).

```python
class BaseIndexer:
    def __init__(self, working_dir, providers, params): ...
    async def build(self, corpus: list[CorpusDoc]) -> None      # index + persist to working_dir
    async def close(self) -> None                               # flush (optional)

class BaseRetriever:
    def __init__(self, working_dir, providers, params): ...
    async def initialize(self) -> None                          # reload index from working_dir
    async def answer(self, question, query_params) -> tuple[str, list[str]]   # (answer, contexts)
    async def close(self) -> None
```

- `working_dir` — one directory per corpus. The indexer writes the index here; it is
  uploaded to S3 and handed back (untouched) to the retriever in a **separate process**.
  Persist everything needed to answer; assume no in-memory carryover.
- `providers` — the resolved `{"llm": {...}, "embedding": {...}}` from
  [providers.yaml](../configs/providers.yaml). Resolve API keys with
  `src.pipeline.shared.providers.resolve_api_key`.
- `params` — your method's config dict (e.g. `lightrag_params.yaml`).
- The retriever **must** use the same embedding configuration as the indexer (the
  evaluator passes it back via the index's `method_meta.json`).

## Add a method

Copy `customrag/` to `src/methods/<name>/`, implement the two classes, update the
`register_method(...)` call in `__init__.py`, and add `src/configs/<name>_params.yaml`.
That's it — see the repo [CONTRIBUTING.md](../../CONTRIBUTING.md) §1.
