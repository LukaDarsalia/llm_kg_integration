from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_kg.config.loader import load_config
from llm_kg.config.schema import (
    DatasetConfig,
    EvaluatorConfig,
    LoggerConfig,
    MethodConfig,
    ProviderConfig,
    RootConfig,
    StorageConfig,
)


def _minimal_yaml() -> str:
    return """
method:
  name: naive_rag
  params:
    top_k: 5
llm:
  name: openai
  params: {model: gpt-4o-mini}
embedder:
  name: openai
  params: {model: text-embedding-3-small}
storage:
  vector:
    name: numpy
    params: {dim: 1536}
  graph:
    name: networkx
    params: {}
  kv:
    name: jsonfile
    params: {path: .cache/kv.json}
dataset:
  name: hotpotqa
  params: {split: dev}
evaluator:
  name: extractive
  params: {}
logger:
  name: "null"
  project: null
cache_dir: .cache
seed: 42
"""


def test_load_config_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(_minimal_yaml())
    cfg = load_config(p)
    assert isinstance(cfg, RootConfig)
    assert cfg.method.name == "naive_rag"
    assert cfg.method.params["top_k"] == 5
    assert cfg.llm.params["model"] == "gpt-4o-mini"
    assert cfg.storage.vector.name == "numpy"
    assert cfg.dataset.name == "hotpotqa"
    assert cfg.evaluator.name == "extractive"
    assert cfg.seed == 42


def test_load_config_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text("method: {name: x}\n")  # missing everything else
    with pytest.raises(ValidationError):
        load_config(p)


def test_provider_config_defaults() -> None:
    pc = ProviderConfig(name="openai")
    assert pc.params == {}
    assert pc.cache is True


def test_logger_config_defaults_to_wandb() -> None:
    lc = LoggerConfig()
    assert lc.name == "wandb"
    assert lc.project is None


def test_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    from llm_kg.config.settings import Secrets

    s = Secrets()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.anthropic_api_key.get_secret_value() == "ant-test"


def test_storage_config_round_trip() -> None:
    sc = StorageConfig(
        vector=ProviderConfig(name="v"),
        graph=ProviderConfig(name="g"),
        kv=ProviderConfig(name="k"),
    )
    assert sc.vector.name == "v"


def test_method_dataset_evaluator_configs() -> None:
    assert MethodConfig(name="m").params == {}
    assert DatasetConfig(name="d").params == {}
    assert EvaluatorConfig(name="e").params == {}
