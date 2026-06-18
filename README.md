# llm_kg_integration

A workspace for **benchmarking and improving GraphRAG methods** (LLM × knowledge-graph
retrieval) on the [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
benchmark. Index once, cache it in S3, and re-run evaluation any number of times — with
every run versioned in Weights & Biases.

```
  loader  ──▶  dataset      ──▶  indexer  ──▶  index_<method>  ──▶  evaluator  ──▶  results_<method>
 (raw → predefined         (build + persist            (reload from S3,            (scores: generation
  format, to S3)            a graph index per           retrieve + answer,          + retrieval, to W&B)
                            corpus, to S3)              score vs leaderboard)
```

Each arrow is a **named W&B artifact** (data lives in S3, the artifact holds an
`s3://` reference), so the pipeline has full lineage and nothing is ever re-indexed by
accident.

## Concepts

- **Method** = a GraphRAG approach: an *indexer* + a *retriever*. Methods are pluggable
  (`--method <name>`). [LightRAG](src/methods/lightrag/) is the reference implementation,
  built on the official `lightrag-hku` SDK. [customrag](src/methods/customrag/) is a
  copy-me template.
- **Stage** = one step of the pipeline (loader / indexer / evaluator). Each is an
  independent CLI under [src/pipeline/](src/pipeline/) and talks to the others only
  through artifacts.
- **Benchmark** = [GraphRAG-Bench](third_party/GraphRAG-Benchmark) (a git submodule). We
  load its corpora/questions and score with its exact metrics for leaderboard parity.

## Setup

```bash
# 1. Python deps (uses uv + pyproject.toml)
uv sync

# 2. Pull the benchmark submodule (datasets + evaluation metrics)
git submodule update --init --recursive

# 3. Secrets
cp example_env.txt .env       # then fill in OPENROUTER_KEY, AWS_*, WANDB_API_KEY
```

Models default to the GraphRAG-Bench leaderboard setup — `gpt-4o-mini` (generation +
judge, via OpenRouter pinned to the OpenAI upstream) and local `BAAI/bge-large-en-v1.5`
(embeddings, 1024-dim). Swap any provider by editing
[src/configs/providers.yaml](src/configs/providers.yaml) only.

## Run the pipeline

```bash
# Stage 1 — load GraphRAG-Bench into the predefined format
uv run python -m src.pipeline.loader.runner \
    --description "load graphrag-bench"

# Stage 2 — build a LightRAG index (cached to S3)
uv run python -m src.pipeline.indexer.runner \
    --dataset-artifact-version latest --method lightrag \
    --description "lightrag index"

# Stage 3 — retrieve, answer, and score (start small with --num-samples)
uv run python -m src.pipeline.evaluator.runner \
    --index-artifact-version latest --method lightrag \
    --num-samples 20 --description "eval lightrag"
```

> Scoring uses an LLM-as-judge: it makes several `gpt-4o-mini` calls **per question**.
> Use `--num-samples` while iterating; run the full set only for a real number.

## Layout

```
.
├── pyproject.toml            # uv-managed deps (pinned eval set for leaderboard parity)
├── example_env.txt           # copy to .env
├── CONTRIBUTING.md           # how to add a method / dataset / metric (read this)
├── src/
│   ├── configs/              # providers.yaml, loader.yaml, lightrag_params.yaml, evaluator.yaml
│   ├── pipeline/
│   │   ├── loader/           # raw benchmark -> predefined dataset format   (stage 1)
│   │   ├── indexer/          # build + persist a method's index             (stage 2)
│   │   ├── evaluator/        # retrieve, answer, score vs benchmark         (stage 3)
│   │   └── shared/           # contracts (base classes), storage (S3), providers
│   └── methods/
│       ├── registry.py       # method name -> (indexer, retriever)
│       ├── lightrag/         # reference method (lightrag-hku SDK)
│       └── customrag/        # template — copy to add your own
└── third_party/
    └── GraphRAG-Benchmark/   # git submodule: datasets + evaluation metrics
```

Each stage and the methods folder has its own `README.md`. The boundaries between them
(dataset format, method interface, prediction schema) are the **contracts** in
[src/pipeline/shared/contracts.py](src/pipeline/shared/contracts.py).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The common cases: add a GraphRAG **method**
(copy `customrag/`), add a **dataset** loader, switch **providers** (edit one YAML), or
adjust the **evaluation** wiring.
