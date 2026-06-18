"""Indexer stage: a method-agnostic orchestrator.

Reads the ``dataset`` artifact, and for each subset indexes every corpus (grouped by
``corpus_name``) with the selected method's :class:`BaseIndexer`, writing one reloadable
index per corpus under ``<output>/<subset>/<corpus_dir>/``. It also copies each subset's
``qa.parquet`` and writes ``method_meta.json`` so the resulting index artifact is
self-contained: the evaluator can reconstruct the retriever (same embedding config) and
score it with only the index artifact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import wandb
from src.methods.registry import get_method
from src.pipeline.shared.config import load_yaml_mapping
from src.pipeline.shared.contracts import CorpusDoc, corpus_dir_name
from src.pipeline.shared.providers import load_providers, public_summary


class Indexer:
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
    ):
        self.artifact = artifact
        self.input_folder_dir = Path(input_folder_dir)
        self.output_folder_dir = Path(output_folder_dir)
        self.method_name = method
        self.method = get_method(method)
        self.config_path = Path(config_path)
        self.providers_path = Path(providers_path)
        self.params = self._load_yaml(self.config_path)
        self.providers = load_providers(str(self.providers_path))
        self.subset_filter = subset
        self.source_filter = source
        self._log_config_to_wandb()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        # Required: a wrong/missing --config must fail loudly, never silently fall back to
        # empty params. A method that genuinely has no parameters should ship an empty
        # yaml file (which load_yaml_mapping reads as {}); a *missing* file is an error.
        return load_yaml_mapping(path, description="method config")

    def _log_config_to_wandb(self) -> None:
        if self.config_path.exists():
            self.artifact.add_file(str(self.config_path), name="config.yaml")
        if self.providers_path.exists():
            self.artifact.add_file(str(self.providers_path), name="providers.yaml")
        method_dir = Path("src/methods") / self.method_name
        for py in sorted(method_dir.glob("*.py")):
            self.artifact.add_file(str(py), name=f"method_code/{py.name}")

    def _discover_subsets(self) -> List[str]:
        subsets = [
            p.name
            for p in sorted(self.input_folder_dir.iterdir())
            if p.is_dir() and (p / "corpus.parquet").exists()
        ]
        if self.subset_filter:
            subsets = [s for s in subsets if s == self.subset_filter]
        return subsets

    def run_indexing(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        subsets = self._discover_subsets()
        if not subsets:
            raise ValueError(f"No subsets with corpus.parquet found under {self.input_folder_dir}.")

        corpora_indexed = 0
        for subset in subsets:
            corpus_df = pd.read_parquet(self.input_folder_dir / subset / "corpus.parquet")
            out_subset = self.output_folder_dir / subset
            out_subset.mkdir(parents=True, exist_ok=True)

            # Copy qa.parquet so the index artifact is self-contained for the evaluator.
            qa_src = self.input_folder_dir / subset / "qa.parquet"
            if qa_src.exists():
                pd.read_parquet(qa_src).to_parquet(out_subset / "qa.parquet", index=False)

            groups: Dict[str, List[CorpusDoc]] = {}
            for _, row in corpus_df.iterrows():
                groups.setdefault(row["corpus_name"], []).append(
                    CorpusDoc(corpus_name=row["corpus_name"], context=row["context"])
                )

            if self.source_filter:
                groups = {k: v for k, v in groups.items() if k == self.source_filter}
                if not groups:
                    print(f"  ⚠️  source '{self.source_filter}' not found in subset '{subset}'")

            for corpus_name, docs in groups.items():
                working_dir = out_subset / corpus_dir_name(corpus_name)
                working_dir.mkdir(parents=True, exist_ok=True)
                print(f"🚀 Indexing [{subset}] {corpus_name} ({len(docs)} doc(s)) -> {working_dir}")
                indexer = self.method.indexer(
                    working_dir=str(working_dir), providers=self.providers, params=self.params
                )
                try:
                    await indexer.build(docs)
                finally:
                    await indexer.close()
                corpora_indexed += 1

        # method_meta.json lets the evaluator reconstruct the retriever with the exact
        # index-time config (api keys are resolved from env, never stored here).
        meta = {
            "method": self.method_name,
            "params": self.params,
            "providers": public_summary(self.providers),
            "subsets": subsets,
        }
        (self.output_folder_dir / "method_meta.json").write_text(json.dumps(meta, indent=2))

        self.artifact.metadata.update(
            {
                "method": self.method_name,
                "subsets": subsets,
                "corpora_indexed": corpora_indexed,
                "providers": public_summary(self.providers),
            }
        )
        print(f"\n✓ Indexed {corpora_indexed} corpus/corpora across subsets {subsets}")
