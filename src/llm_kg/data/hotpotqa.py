"""HotpotQA dataset (HippoRAG's 1000-query sample, distractor setting)."""

from __future__ import annotations

from llm_kg.data import DATASET_REGISTRY
from llm_kg.data._hippo_loader import HippoMultiHopDataset


@DATASET_REGISTRY.register("hotpotqa")
class HotpotQADataset(HippoMultiHopDataset):
    """HotpotQA from HippoRAG's `reproduce/dataset/hotpotqa.json`.

    1000 queries × ~10 candidate paragraphs each; deduped corpus has ~9,811
    unique passages. Supporting passages are derived from each query's
    `supporting_facts` (sentence-level [title, sent_idx] pairs) by taking
    the unique titles.
    """

    name = "hotpotqa"
