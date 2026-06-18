"""CustomRAG method template — registers a not-yet-implemented method.

Selecting ``--method customrag`` will run, but raise a clear NotImplementedError telling
you what to fill in. Copy this folder to ``src/methods/<your_method>/`` to start.
"""

from src.methods.registry import register_method
from .data_indexer import CustomRAGIndexer
from .data_retriever import CustomRAGRetriever

register_method(
    "customrag",
    indexer=CustomRAGIndexer,
    retriever=CustomRAGRetriever,
    description="Template method — copy src/methods/customrag/ to start a new GraphRAG method.",
)
