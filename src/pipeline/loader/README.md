# Loader stage

Loads raw benchmarks into the pipeline's **predefined dataset format** and publishes them
as the W&B `dataset` artifact (mirrored to S3 by reference).

```
loader/
├── runner.py     # Click CLI -> wandb run -> DatasetLoader -> upload S3 -> log artifact
├── loader.py     # DatasetLoader: iterate config, validate, write parquet, log samples
├── loaders.py    # @register_loader functions (graphrag_bench, hotpotqa stub)
└── registry.py   # LoaderRegistry + validate_dataset_output (the dataset contract)
```

## Output contract

Per subset, the loader writes two files into the artifact:

| file | required columns |
|------|------------------|
| `<subset>/corpus.parquet` | `corpus_name`, `context` |
| `<subset>/qa.parquet` | `id`, `source`, `question`, `answer`, `question_type`, `evidence` (+ extras) |

`source` in `qa.parquet` matches `corpus_name` in `corpus.parquet` — that join is how the
indexer/evaluator know which questions belong to which corpus.
`validate_dataset_output` enforces these columns; a loader that violates it fails here.

## Run

```bash
uv run python -m src.pipeline.loader.runner \
    --description "load graphrag-bench medical+novel" \
    --config src/configs/loader.yaml
```

| option | default | meaning |
|--------|---------|---------|
| `--description` | *(required)* | run note (W&B) |
| `--config` | `src/configs/loader.yaml` | which loaders + params |
| `--bucket` | `graphrag-bench-data` | S3 bucket |
| `--project` | `GraphRAG_Bench` | W&B project |
| `--develop` | off | anonymous/offline run (no W&B key) |

## Add a loader

Add a `@register_loader("name", "desc")` generator to `loaders.py` that `yield`s
`(subset_name, corpus_df, qa_df)`, then add an entry to `src/configs/loader.yaml`. See
[CONTRIBUTING.md](../../../CONTRIBUTING.md) §2.
