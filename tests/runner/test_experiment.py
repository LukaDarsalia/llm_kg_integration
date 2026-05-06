"""Sanity tests for the runner — full E2E with stubs lives in test_smoke_e2e.py."""

from pathlib import Path

import pytest

from llm_kg.runner.experiment import Experiment

# NOTE: YAML unquoted `null` parses to None — use quoted "null" for the string registry name.
_NOPE_YAML = """
method: {name: nope, params: {}}
llm: {name: nope, params: {}}
embedder: {name: nope, params: {}}
storage:
  vector: {name: nope, params: {}}
  graph: {name: nope, params: {}}
  kv: {name: nope, params: {}}
dataset: {name: nope, params: {}}
evaluator: {name: extractive, params: {}}
logger: {name: "null"}
"""


def test_experiment_construct_only_loads_config(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(_NOPE_YAML)
    exp = Experiment(p)
    assert exp.config.method.name == "nope"


async def test_experiment_run_fails_for_unknown_plugin(tmp_path: Path) -> None:
    """The runner resolves plug-ins in order (LLM first); any unknown name raises KeyError."""
    p = tmp_path / "exp.yaml"
    p.write_text(_NOPE_YAML)
    with pytest.raises(KeyError, match="unknown llm"):
        await Experiment(p).run()
