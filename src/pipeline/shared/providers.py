"""Model provider configuration: load providers.yaml and resolve API keys from env.

This module is intentionally *method-agnostic*. It loads/validates the provider config
and resolves env-var API keys. The translation of a provider config into the concrete
callables a method needs lives next to that method (e.g.
``src/methods/lightrag/models.py``), and the judge LLM/embedding construction lives in
``src/pipeline/evaluator/benchmark_adapter.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from src.pipeline.shared.config import load_yaml_mapping

DEFAULT_PROVIDERS_PATH = "src/configs/providers.yaml"


def load_providers(config_path: str = DEFAULT_PROVIDERS_PATH) -> Dict[str, Any]:
    """Load and validate the providers config (a mapping with `llm` + `embedding`)."""
    cfg = load_yaml_mapping(config_path, description="providers config")
    for role in ("llm", "embedding"):
        if role not in cfg:
            raise ValueError(
                f"providers config ({config_path}) is missing the required '{role}' "
                f"section. Found: {sorted(cfg)}"
            )
        if not isinstance(cfg[role], dict):
            raise ValueError(
                f"providers config '{role}' must be a mapping, got {type(cfg[role]).__name__}."
            )
    return cfg


def resolve_api_key(role_cfg: Dict[str, Any]) -> str:
    """Read the API key named by ``api_key_env`` from the environment."""
    env_name = role_cfg.get("api_key_env")
    if not env_name:
        raise ValueError(
            f"Provider config {role_cfg.get('provider')!r} has no `api_key_env`. "
            "Add it (the name of the env var holding the key)."
        )
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_name!r} is not set. "
            "Copy example_env.txt to .env and fill it in."
        )
    return key


def public_summary(providers: Dict[str, Any]) -> Dict[str, Any]:
    """A secrets-free snapshot of the provider config, for run metadata / method_meta.json."""

    def _strip(role: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in role.items() if k != "api_key"}

    return {"llm": _strip(providers["llm"]), "embedding": _strip(providers["embedding"])}
