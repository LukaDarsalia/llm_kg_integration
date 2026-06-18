# Indexer stage

Builds a method's index over a `dataset` artifact and publishes it as `index_<method>`
(mirrored to S3). Method-agnostic: it selects the method via `--method` and delegates to
that method's `BaseIndexer`.

```
indexer/
├── runner.py    # Click CLI -> consume dataset artifact -> Indexer -> upload S3 -> log artifact
└── indexer.py   # Indexer: per subset, per corpus_name -> method.indexer.build(); writes method_meta.json
```

## What it produces

For each subset and each `corpus_name`, one reloadable index directory:

```
<output>/
├── method_meta.json                 # method + params + (secrets-free) providers, subsets
└── <subset>/
    ├── qa.parquet                   # copied through, so the index is self-contained
    └── <corpus_dir>/                # one LightRAG working_dir per corpus
        ├── kv_store_*.json
        ├── vdb_*.json
        └── graph_chunk_entity_relation.graphml
```

`method_meta.json` is what lets the evaluator reconstruct the retriever with the **exact
index-time embedding config** (so the query vectors land in the same space). This is the
core of the S3 *cache-and-reload* design: index once, evaluate many times.

## Run

```bash
uv run python -m src.pipeline.indexer.runner \
    --dataset-artifact-version latest \
    --method lightrag \
    --description "lightrag index of graphrag-bench"
```

| option | default | meaning |
|--------|---------|---------|
| `--dataset-artifact-version` | *(required)* | `latest`, `v1`, … |
| `--method` | *(required)* | method name (e.g. `lightrag`) |
| `--subset` | all | index only this subset |
| `--config` | `src/configs/<method>_params.yaml` | method params |
| `--providers` | `src/configs/providers.yaml` | LLM + embedding config |
| `--description` | *(required)* | run note |
| `--develop` | off | anonymous/offline run |

The method is chosen at runtime; to add one, see
[src/methods/README.md](../../methods/README.md).
