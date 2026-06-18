"""Evaluator stage: run a method's retriever over the benchmark and score it.

Two phases:
  1. **Predict** — for each subset, group questions by ``source``, reload that corpus's
     index with the method's :class:`BaseRetriever` (reconstructed from the index's
     ``method_meta.json`` so the embedding space matches), and produce one GraphRAG-Bench
     unified record per question.
  2. **Score** — feed those records to the vendored benchmark metrics (gpt-4o-mini judge).

Predictions + scores are written locally and logged to W&B (summary metrics + a sample
table); the runner mirrors the output dir to S3.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb
import yaml

from src.methods.registry import get_method
from src.pipeline.evaluator.benchmark_adapter import (
    build_judge_embeddings,
    build_judge_llm,
    score_generation,
    score_retrieval,
)
from src.pipeline.shared.contracts import Prediction, corpus_dir_name, validate_predictions
from src.pipeline.shared.providers import load_providers

_PREDICTION_TABLE_COLS = [
    "id",
    "source",
    "question_type",
    "question",
    "generated_answer",
    "ground_truth",
]


class Evaluator:
    def __init__(
        self,
        artifact: wandb.Artifact,
        input_folder_dir: str,
        output_folder_dir: str,
        method: str,
        config_path: str,
        providers_path: str,
        subset: Optional[str] = None,
        source: Optional[str] = None,
        num_samples: Optional[int] = None,
        mode: Optional[str] = None,
    ):
        self.artifact = artifact
        self.input_folder_dir = Path(input_folder_dir)
        self.output_folder_dir = Path(output_folder_dir)
        self.method_name = method
        self.method = get_method(method)
        self.config_path = Path(config_path)
        self.providers_path = Path(providers_path)
        self.config = self._load_yaml(self.config_path)
        self.providers = load_providers(str(self.providers_path))  # judge provider
        self.subset_filter = subset
        self.source_filter = source
        self.cli_num_samples = num_samples
        self.cli_mode = mode
        self.method_meta = self._load_method_meta()
        self._log_config_to_wandb()

    # --- config / meta ----------------------------------------------------------------
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_method_meta(self) -> Dict[str, Any]:
        meta_path = self.input_folder_dir / "method_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        print(f"  ⚠️  {meta_path} missing; falling back to providers.yaml for the retriever.")
        return {}

    def _log_config_to_wandb(self) -> None:
        if self.config_path.exists():
            self.artifact.add_file(str(self.config_path), name="config.yaml")
        for fp in [
            "src/pipeline/evaluator/evaluator.py",
            "src/pipeline/evaluator/benchmark_adapter.py",
        ]:
            if Path(fp).exists():
                self.artifact.add_file(fp, name=f"evaluator_code/{Path(fp).name}")

    # --- helpers ----------------------------------------------------------------------
    def _discover_subsets(self) -> List[str]:
        subsets = [
            p.name
            for p in sorted(self.input_folder_dir.iterdir())
            if p.is_dir() and (p / "qa.parquet").exists()
        ]
        if self.subset_filter:
            subsets = [s for s in subsets if s == self.subset_filter]
        return subsets

    def _retriever_providers(self) -> Dict[str, Any]:
        # Use the index-time providers so the embedding space matches; keys come from env.
        return self.method_meta.get("providers") or self.providers

    def _resolve_query_params(self) -> Dict[str, Any]:
        query = dict((self.method_meta.get("params") or {}).get("query") or {})
        query.update(self.config.get("retrieval") or {})
        if self.cli_mode:
            query["mode"] = self.cli_mode
        return query

    def _num_samples(self) -> Optional[int]:
        return self.cli_num_samples if self.cli_num_samples is not None else self.config.get("num_samples")

    def _max_concurrency(self) -> int:
        return int((self.config.get("judge") or {}).get("max_concurrency", 8))

    # --- run --------------------------------------------------------------------------
    def run_evaluation(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        records = await self._generate_predictions()
        validate_predictions(records)
        preds_path = self.output_folder_dir / "predictions.json"
        preds_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"  ✓ wrote {len(records)} predictions -> {preds_path}")

        dims = self.config.get("dimensions", {})
        judge_llm = build_judge_llm(self.providers["llm"])
        scores: Dict[str, Any] = {}

        if dims.get("generation", True):
            judge_emb = build_judge_embeddings(
                (self.config.get("judge") or {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
            )
            scores["generation"] = await score_generation(
                records, judge_llm, judge_emb, max_concurrency=self._max_concurrency()
            )
            print("  ✓ generation scored")

        if dims.get("retrieval", True):
            scores["retrieval"] = await score_retrieval(
                records, judge_llm, max_concurrency=self._max_concurrency()
            )
            print("  ✓ retrieval scored")

        (self.output_folder_dir / "scores.json").write_text(json.dumps(scores, indent=2))
        self._log_scores(records, scores)

    async def _generate_predictions(self) -> List[Dict[str, Any]]:
        subsets = self._discover_subsets()
        if not subsets:
            raise ValueError(f"No subsets with qa.parquet found under {self.input_folder_dir}.")

        query_params = self._resolve_query_params()
        retriever_providers = self._retriever_providers()
        method_params = self.method_meta.get("params") or {}
        num_samples = self._num_samples()
        predictions: List[Prediction] = []

        for subset in subsets:
            qa_df = pd.read_parquet(self.input_folder_dir / subset / "qa.parquet")
            if self.source_filter:
                qa_df = qa_df[qa_df["source"] == self.source_filter]
            if num_samples:
                qa_df = qa_df.head(num_samples)
            for source, group in qa_df.groupby("source"):
                working_dir = self.input_folder_dir / subset / corpus_dir_name(source)
                if not working_dir.exists():
                    print(f"  ⚠️  no index for source '{source}' ({working_dir}); skipping {len(group)} q.")
                    continue
                retriever = self.method.retriever(
                    working_dir=str(working_dir), providers=retriever_providers, params=method_params
                )
                await retriever.initialize()
                try:
                    for _, row in group.iterrows():
                        answer, contexts = await retriever.answer(row["question"], query_params)
                        predictions.append(
                            Prediction(
                                id=row["id"],
                                question=row["question"],
                                source=row["source"],
                                context=contexts,
                                evidence=str(row.get("evidence", "")),
                                question_type=row.get("question_type", "Uncategorized"),
                                generated_answer=answer,
                                ground_truth=row["answer"],
                            )
                        )
                finally:
                    await retriever.close()
                print(f"  · [{subset}] {source}: answered {len(group)} question(s)")

        return [p.to_record() for p in predictions]

    # --- logging ----------------------------------------------------------------------
    def _log_scores(self, records: List[Dict[str, Any]], scores: Dict[str, Any]) -> None:
        flat: Dict[str, float] = {}
        gen = scores.get("generation", {})
        for qt, md in gen.get("by_question_type", {}).items():
            for metric, value in md.items():
                flat[f"generation/{qt}/{metric}"] = value
        for metric, value in gen.get("overall", {}).items():
            flat[f"generation/overall/{metric}"] = value
        for metric, value in scores.get("retrieval", {}).get("overall", {}).items():
            flat[f"retrieval/{metric}"] = value
        for metric, count in gen.get("failures", {}).items():
            flat[f"generation/failures/{metric}"] = count
        for metric, count in scores.get("retrieval", {}).get("failures", {}).items():
            flat[f"retrieval/failures/{metric}"] = count

        wandb.log(flat)
        wandb.summary.update(flat)
        self.artifact.metadata.update(
            {
                "method": self.method_name,
                "num_predictions": len(records),
                "scores": {
                    "generation_overall": gen.get("overall", {}),
                    "retrieval_overall": scores.get("retrieval", {}).get("overall", {}),
                },
            }
        )

        sample = pd.DataFrame(
            [{k: r[k] for k in _PREDICTION_TABLE_COLS} for r in records[:50]]
        )
        self.artifact.add(wandb.Table(dataframe=sample), "prediction_sample")
        # predictions.json / scores.json live in output_folder_dir and are captured by the
        # runner's add_reference("s3://.../output_folder_dir") — do NOT add_file them here
        # (that would duplicate the manifest path and raise "Cannot add the same path twice").

        print("\n=== SCORES ===")
        for key, value in flat.items():
            print(f"  {key}: {value:.4f}")
