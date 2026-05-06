"""Method abstract + registry. Concrete methods (NaiveRAG, HippoRAG, LightRAG) land later."""

from llm_kg.methods.base import Method, MethodConfig
from llm_kg.registry import Registry

METHOD_REGISTRY: Registry[Method] = Registry("method")

__all__ = ["METHOD_REGISTRY", "Method", "MethodConfig"]
