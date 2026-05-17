"""Method abstract + registry. Concrete methods (NaiveRAG, HippoRAG, LightRAG)
register themselves on first import."""

from llm_kg.methods.base import Method, MethodConfig
from llm_kg.registry import Registry

METHOD_REGISTRY: Registry[Method] = Registry("method")

# Import method packages so their @register decorators run.
from llm_kg.methods import lightrag  # noqa: E402, F401

__all__ = ["METHOD_REGISTRY", "Method", "MethodConfig"]
