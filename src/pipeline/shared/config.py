"""Shared YAML config loading with fail-loud validation.

Every stage loads its YAML config through ``load_yaml_mapping`` so a malformed config
fails clearly and consistently here, instead of surfacing later as a confusing
``AttributeError``/``TypeError`` (e.g. ``None.get(...)`` when the file is empty):

  * empty file (YAML ``null``)         -> ``{}``
  * missing file                       -> ``FileNotFoundError`` (or ``{}`` if not required)
  * non-mapping top level (list/scalar)-> ``ValueError`` naming the file and the wrong type

It never returns ``None`` and never silently swallows a bad config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml_mapping(
    path: Union[str, Path], description: str = "config", required: bool = True
) -> Dict[str, Any]:
    """Load a YAML file that must be a mapping (dict) at the top level."""
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{description} not found: {p}")
        print(f"  ⚠️  {description} not found ({p}); using empty config.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:  # empty file
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{description} ({p}) must be a YAML mapping at the top level, "
            f"got {type(data).__name__}."
        )
    return data
