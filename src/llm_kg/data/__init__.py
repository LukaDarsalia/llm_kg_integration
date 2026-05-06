"""Data types and dataset/corpus abstractions."""

from llm_kg.data.dataset import QADataset
from llm_kg.registry import Registry

DATASET_REGISTRY: Registry[QADataset] = Registry("dataset")

__all__ = ["DATASET_REGISTRY", "QADataset"]
