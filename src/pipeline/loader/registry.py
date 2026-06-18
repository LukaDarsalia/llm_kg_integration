"""Dataset-loader registry + the dataset output contract validator.

A loader is a function that yields ``(subset_name, corpus_df, qa_df)`` groups. The
registry validates each group against the dataset contract (required columns) so a
broken loader fails loudly here rather than deep in the indexer.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from src.pipeline.shared.contracts import CORPUS_REQUIRED_COLUMNS, QA_REQUIRED_COLUMNS


class LoaderRegistry:
    def __init__(self) -> None:
        self._loaders: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        def decorator(func: Callable) -> Callable:
            self._loaders[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_loader(self, name: str) -> Optional[Callable]:
        return self._loaders.get(name)

    def list_loaders(self) -> Dict[str, str]:
        return dict(self._descriptions)

    def validate_dataset_output(
        self, corpus_df: pd.DataFrame, qa_df: pd.DataFrame, loader_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        missing_corpus = [c for c in CORPUS_REQUIRED_COLUMNS if c not in corpus_df.columns]
        missing_qa = [c for c in QA_REQUIRED_COLUMNS if c not in qa_df.columns]
        if missing_corpus:
            raise ValueError(
                f"Loader '{loader_name}' corpus output missing columns {missing_corpus}. "
                f"Required: {CORPUS_REQUIRED_COLUMNS}"
            )
        if missing_qa:
            raise ValueError(
                f"Loader '{loader_name}' QA output missing columns {missing_qa}. "
                f"Required: {QA_REQUIRED_COLUMNS}"
            )
        return corpus_df, qa_df


loader_registry = LoaderRegistry()


def register_loader(name: str, description: str = "") -> Callable:
    """Convenience decorator: ``@register_loader("name", "desc")``."""
    return loader_registry.register(name, description)
