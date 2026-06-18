"""Loader stage: turn raw benchmarks into the pipeline's predefined dataset format.

For each enabled entry in loader.yaml, run its registered loader, validate every
``(subset, corpus_df, qa_df)`` group against the dataset contract, and write
``<output>/<subset>/corpus.parquet`` + ``qa.parquet``. Samples and counts are logged to
the W&B artifact for lineage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import wandb
import yaml

from .loaders import *  # noqa: F401,F403  (import registers all loaders)
from .registry import loader_registry

_CODE_FILES = [
    "src/pipeline/loader/loader.py",
    "src/pipeline/loader/loaders.py",
    "src/pipeline/loader/registry.py",
]


class DatasetLoader:
    def __init__(
        self,
        artifact: wandb.Artifact,
        output_folder_dir: str,
        config_path: str = "src/configs/loader.yaml",
    ):
        self.artifact = artifact
        self.output_folder_dir = Path(output_folder_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._log_config_to_wandb()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _log_config_to_wandb(self) -> None:
        self.artifact.add_file(str(self.config_path), name="config.yaml")
        for fp in _CODE_FILES:
            if Path(fp).exists():
                self.artifact.add_file(fp, name=f"loader_code/{Path(fp).name}")

    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            d["name"]: {"enabled": d.get("enabled", True), "description": d.get("description", "")}
            for d in self.config.get("datasets", [])
        }

    def run_loading(self) -> None:
        for entry in self.config.get("datasets", []):
            name = entry["name"]
            if not entry.get("enabled", True):
                print(f"  ✗ skipping disabled dataset: {name}")
                continue
            loader_fn = loader_registry.get_loader(name)
            if loader_fn is None:
                raise ValueError(
                    f"Loader '{name}' not found. Available: {list(loader_registry.list_loaders())}"
                )
            print(f"🚀 Loading dataset: {name}")
            for subset_name, corpus_df, qa_df in loader_fn(entry.get("params", {})):
                loader_registry.validate_dataset_output(corpus_df, qa_df, name)
                out = self.output_folder_dir / subset_name
                out.mkdir(parents=True, exist_ok=True)
                corpus_df.to_parquet(out / "corpus.parquet", index=False)
                qa_df.to_parquet(out / "qa.parquet", index=False)
                self._log_subset(subset_name, corpus_df, qa_df)
                print(
                    f"  ✓ {subset_name}: {len(corpus_df)} corpus doc(s), {len(qa_df)} question(s)"
                )

    def _log_subset(self, name: str, corpus_df, qa_df) -> None:
        self.artifact.metadata.update(
            {f"{name}_corpus_docs": int(len(corpus_df)), f"{name}_questions": int(len(qa_df))}
        )
        # Sample the QA table only (corpus `context` blobs are far too large to log).
        self.artifact.add(wandb.Table(dataframe=qa_df.head(20)), f"{name}_qa_sample")
