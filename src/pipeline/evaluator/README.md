# Evaluator stage

Runs a method's retriever over an `index_<method>` artifact, produces GraphRAG-Bench
unified prediction records, and scores them with the **vendored benchmark metrics**
(gpt-4o-mini judge). Publishes `results_<method>` and logs scores to W&B.

```
evaluator/
├── runner.py             # Click CLI -> consume index artifact -> Evaluator -> upload S3 -> log artifact
├── evaluator.py          # Evaluator: predict (retriever) then score (benchmark_adapter)
└── benchmark_adapter.py  # judge construction + scoring via third_party/GraphRAG-Benchmark
```

## What it does

1. **Predict.** Group questions by `source`; reload each corpus's index with the method's
   `BaseRetriever` (config from the index's `method_meta.json`); per question emit a
   record: `id, question, source, context (list), evidence, question_type,
   generated_answer, ground_truth`. `validate_predictions` enforces the schema.
2. **Score.** Feed records to the GraphRAG-Bench metric functions:
   - **generation** (per `question_type`): `rouge_score`, `answer_correctness`,
     `coverage_score`, `faithfulness` — selected exactly as the leaderboard does.
   - **retrieval**: `context_relevancy`, `evidence_recall`.

   The judge is `gpt-4o-mini` (via [providers.yaml](../../configs/providers.yaml), temp 0
   / seed 42) and embeddings are local `BAAI/bge-large-en-v1.5` — matching the public
   leaderboard.

Outputs `predictions.json` + `scores.json`, flattened metrics to the W&B run summary, and
a prediction sample table.

## Run

```bash
uv run python -m src.pipeline.evaluator.runner \
    --index-artifact-version latest \
    --method lightrag \
    --num-samples 20 \
    --description "eval lightrag on graphrag-bench"
```

| option | default | meaning |
|--------|---------|---------|
| `--index-artifact-version` | *(required)* | `latest`, `v1`, … |
| `--method` | *(required)* | method name |
| `--subset` | all | evaluate only this subset |
| `--mode` | from config | override retrieval mode (`naive\|local\|global\|hybrid\|mix`) |
| `--num-samples` | all | cap questions per subset (dev — judge calls cost money) |
| `--config` | `src/configs/evaluator.yaml` | dimensions, judge, retrieval overrides |
| `--providers` | `src/configs/providers.yaml` | judge LLM config |
| `--develop` | off | anonymous/offline run |

> **Cost note:** generation + retrieval scoring make several `gpt-4o-mini` calls **per
> question**. Use `--num-samples` while iterating; run the full ~4000-question set only
> for a real leaderboard number.

The scoring metrics come from the `third_party/GraphRAG-Benchmark` submodule — see
[CONTRIBUTING.md](../../../CONTRIBUTING.md) §4.
