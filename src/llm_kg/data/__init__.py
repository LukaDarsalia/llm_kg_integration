"""Data types and dataset/corpus abstractions."""

from llm_kg.data.dataset import QADataset
from llm_kg.registry import Registry

DATASET_REGISTRY: Registry[QADataset] = Registry("dataset")

# Import concrete datasets so @register decorators run.
from llm_kg.data import musique  # noqa: E402, F401

__all__ = ["DATASET_REGISTRY", "QADataset"]
