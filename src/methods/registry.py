"""Method registry: maps a method name to its indexer + retriever classes.

A "method" is a GraphRAG approach (e.g. LightRAG) that implements the contract in
``src/pipeline/shared/contracts.py``. Methods self-register on import; ``get_method``
imports the method package lazily so heavy dependencies (torch, lightrag, ...) are only
pulled in for the method actually selected on the CLI.

Adding a method:
    1. Create ``src/methods/<name>/`` with ``data_indexer.py`` (a ``BaseIndexer``
       subclass) and ``data_retriever.py`` (a ``BaseRetriever`` subclass).
    2. In ``src/methods/<name>/__init__.py`` call ``register_method(...)``.
That's it — ``--method <name>`` then works for the indexer and evaluator stages.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Type

from src.pipeline.shared.contracts import BaseIndexer, BaseRetriever


@dataclass
class Method:
    name: str
    indexer: Type[BaseIndexer]
    retriever: Type[BaseRetriever]
    description: str = ""


_REGISTRY: Dict[str, Method] = {}


def register_method(
    name: str,
    indexer: Type[BaseIndexer],
    retriever: Type[BaseRetriever],
    description: str = "",
) -> None:
    """Register a method's indexer + retriever classes under ``name``."""
    _REGISTRY[name] = Method(
        name=name, indexer=indexer, retriever=retriever, description=description
    )


def list_methods() -> Dict[str, str]:
    """All currently-registered methods (only those already imported)."""
    return {name: m.description for name, m in _REGISTRY.items()}


def get_method(name: str) -> Method:
    """Look up a method by name, importing ``src.methods.<name>`` to trigger registration."""
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"src.methods.{name}")
        except ImportError as exc:
            raise ValueError(
                f"Method '{name}' could not be imported ({exc}). "
                f"Registered methods: {sorted(_REGISTRY)}"
            ) from exc
    if name not in _REGISTRY:
        raise ValueError(
            f"Method '{name}' did not register itself. Ensure src/methods/{name}/__init__.py "
            f"calls register_method(...). Registered methods: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
