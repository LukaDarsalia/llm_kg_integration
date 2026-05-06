"""YAML → RootConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_kg.config.schema import RootConfig


def load_config(path: Path | str) -> RootConfig:
    """Read a YAML file and validate it against RootConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ValueError(f"empty config: {path}")
    return RootConfig.model_validate(raw)
