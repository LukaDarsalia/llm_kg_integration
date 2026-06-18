# Contributing

This repo is a **GraphRAG benchmarking workspace**. The pipeline has three stages —
**loader → indexer → evaluator** — and a set of pluggable **methods** (a method = a
GraphRAG indexer + retriever). Most contributions are one of four things; each touches a
small, predictable set of files. Read [the contracts](src/pipeline/shared/contracts.py)
first — they define every boundary below.

## Ground rules

- **One stage = one folder** under [src/pipeline/](src/pipeline/): `runner.py` (Click CLI
  + W&B/S3 lineage), `<stage>.py` (the orchestrator class), `registry.py` (where a stage
  has pluggable parts), `README.md`. Stages talk to each other **only** through named
  W&B artifacts (`dataset` → `index_<method>` → `results_<method>`).
- **Never hardcode secrets.** API keys are referenced by env-var *name* in
  [providers.yaml](src/configs/providers.yaml) and resolved at runtime from `.env`.
- **Fail loud on contract drift.** Use the registry validators
  (`validate_dataset_output`, `validate_predictions`) so a malformed output is caught at
  its source, not three stages later.
- **Match the surrounding style.** Module + Google-style docstrings, `click` runners,
  `✓/✗/⚠️/🚀` status prints, type hints.
- Branch off `main`, open a PR. Run `uv run python -m compileall src` before pushing.

## 1. Add a new GraphRAG method (most common)

A method implements the contract in [contracts.py](src/pipeline/shared/contracts.py):
`BaseIndexer` (build + persist an index per corpus) and `BaseRetriever` (reload it and
answer questions). See [src/methods/README.md](src/methods/README.md) and the
[LightRAG reference](src/methods/lightrag/).

**Files to add/change:**
1. `src/methods/<name>/data_indexer.py` — a `BaseIndexer` subclass.
2. `src/methods/<name>/data_retriever.py` — a `BaseRetriever` subclass.
3. `src/methods/<name>/__init__.py` — call `register_method("<name>", Indexer, Retriever, "desc")`.
4. `src/configs/<name>_params.yaml` — your method's parameters (the indexer/evaluator
   load this as `--config`).

Quickest path: copy [src/methods/customrag/](src/methods/customrag/) (a working
template), rename, and fill in the two `NotImplementedError`s. Then:
`--method <name>` works for both the indexer and evaluator stages with no other changes.

**The persistence contract is load-bearing:** the indexer writes everything to
`working_dir`; that directory is uploaded to S3 and is the *only* thing the retriever
gets back (in a fresh process). The retriever must rebuild from `working_dir` + config
alone, using the **same embedding configuration** as indexing (handled for you via the
index's `method_meta.json`).

## 2. Add a new dataset / benchmark

A loader yields `(subset_name, corpus_df, qa_df)` groups meeting the dataset contract
(`corpus_name, context` and `id, source, question, answer, question_type, evidence`).

**Files to change:**
1. `src/pipeline/loader/loaders.py` — add a `@register_loader("name", "desc")` generator.
2. `src/configs/loader.yaml` — add an entry (`name`, `enabled`, `params`).

See `load_hotpotqa` (a stub) and `load_graphrag_bench` (full) for the shape. Validation
against the contract is automatic.

## 3. Change which models/providers are used

Edit **only** [src/configs/providers.yaml](src/configs/providers.yaml) (LLM + embedding
provider, base_url, model, env-var name). Everything is OpenAI-compatible. To swap to a
non-OpenAI-compatible provider you'd extend the small factories in
[src/methods/lightrag/models.py](src/methods/lightrag/models.py) (method side) and
[src/pipeline/evaluator/benchmark_adapter.py](src/pipeline/evaluator/benchmark_adapter.py)
(judge side). Add any new key name to [example_env.txt](example_env.txt).

## 4. Change the evaluation metrics

The scoring metrics are **vendored from GraphRAG-Bench** (the
[third_party/GraphRAG-Benchmark](third_party/) git submodule) so our numbers stay
comparable to the public leaderboard — don't fork them lightly. The wiring (judge
construction, per-question_type metric selection, aggregation) lives in
[benchmark_adapter.py](src/pipeline/evaluator/benchmark_adapter.py); add a *new* dimension
or a non-leaderboard metric there. To bump the benchmark itself:
`cd third_party/GraphRAG-Benchmark && git pull`, then commit the new submodule pointer.

## Running the checks

```bash
uv sync                              # install the environment
uv run python -m compileall src      # syntax
uv run python -m src.pipeline.loader.runner --help   # smoke-test a runner
```
