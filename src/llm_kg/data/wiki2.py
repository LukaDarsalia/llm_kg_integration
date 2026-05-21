"""2WikiMultiHopQA dataset (HippoRAG's 1000-query sample)."""

from __future__ import annotations

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data._hippo_loader import HippoMultiHopDataset


@DATASET_REGISTRY.register("2wikimultihopqa")
class MultiHopWiki2Dataset(HippoMultiHopDataset):
    """2WikiMultiHopQA from HippoRAG's `reproduce/dataset/2wikimultihopqa.json`.

    1000 queries × 10 candidate paragraphs each; deduped corpus has ~6,119
    unique passages. Same shape as HotpotQA in HippoRAG's repo (context is a
    list of [title, [sentences]] pairs; supporting_facts is sentence-level).
    """

    name = "2wikimultihopqa"
