"""LightRAG method — registers itself with the method registry on import."""

from src.methods.registry import register_method

from .data_indexer import LightRAGIndexer
from .data_retriever import LightRAGRetriever

register_method(
    "lightrag",
    indexer=LightRAGIndexer,
    retriever=LightRAGRetriever,
    description="LightRAG (HKUDS) via the official lightrag-hku SDK.",
)
