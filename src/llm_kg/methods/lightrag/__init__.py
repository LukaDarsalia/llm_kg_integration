"""LightRAG method (KG-augmented RAG with dual-level keyword retrieval).

Imports `method` so the `@METHOD_REGISTRY.register("lightrag")` decorator runs
on first import of this package.
"""

from llm_kg.methods.lightrag import method  # noqa: F401
