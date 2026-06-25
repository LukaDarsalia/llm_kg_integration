"""Registered dataset loaders.

Each loader takes its ``params`` dict (from loader.yaml) and yields one or more
``(subset_name, corpus_df, qa_df)`` groups:
  * ``corpus_df`` columns: corpus_name, context
  * ``qa_df``     columns: id, source, question, answer, question_type, evidence (+ extras)

Add a new benchmark by writing a ``@register_loader``-decorated generator here and adding
an entry to src/configs/loader.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pandas as pd

from .registry import register_loader


@register_loader(
    "graphrag_bench",
    "GraphRAG-Bench medical + novel corpora and QA (from the git submodule).",
)
def load_graphrag_bench(params: Dict[str, Any]) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    source_dir = Path(params.get("source_dir", "third_party/GraphRAG-Benchmark/Datasets"))
    for subset in params.get("subsets", ["medical", "novel"]):
        corpus_path = source_dir / "Corpus" / f"{subset}.parquet"
        qa_path = source_dir / "Questions" / f"{subset}_questions.parquet"
        if not corpus_path.exists() or not qa_path.exists():
            raise FileNotFoundError(
                f"GraphRAG-Bench data missing for subset '{subset}'.\n"
                f"  expected: {corpus_path}\n            {qa_path}\n"
                "Initialize the submodule:  git submodule update --init --recursive"
            )
        yield subset, pd.read_parquet(corpus_path), pd.read_parquet(qa_path)


@register_loader("hotpotqa", "HotpotQA (example stub — implement me).")
def load_hotpotqa(params: Dict[str, Any]) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    raise NotImplementedError(
        "load_hotpotqa is a stub. Build corpus_df (corpus_name, context) and qa_df "
        "(id, source, question, answer, question_type, evidence) and `yield` them. "
        "See load_graphrag_bench above for the shape."
    )
    yield  # pragma: no cover  (keeps this a generator function)
