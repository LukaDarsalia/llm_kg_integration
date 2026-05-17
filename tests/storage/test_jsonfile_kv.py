import json
from pathlib import Path

import pytest

from llm_kg.storage import KV_REGISTRY
from llm_kg.storage.jsonfile_kv import JsonFileKVStore


def test_registered_as_jsonfile() -> None:
    assert "jsonfile" in KV_REGISTRY
    assert KV_REGISTRY.get("jsonfile") is JsonFileKVStore


def test_put_get_round_trip(tmp_path: Path) -> None:
    kv = JsonFileKVStore(path=tmp_path / "kv.json")
    kv.put("a", "alpha")
    kv.put("b", {"nested": [1, 2, 3]})
    assert kv.get("a") == "alpha"
    assert kv.get("b") == {"nested": [1, 2, 3]}


def test_get_missing_raises_keyerror(tmp_path: Path) -> None:
    kv = JsonFileKVStore(path=tmp_path / "kv.json")
    with pytest.raises(KeyError):
        kv.get("nope")


def test_contains(tmp_path: Path) -> None:
    kv = JsonFileKVStore(path=tmp_path / "kv.json")
    kv.put("x", 1)
    assert "x" in kv
    assert "y" not in kv


def test_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "kv.json"
    a = JsonFileKVStore(path=p)
    a.put("k", {"v": 42})

    b = JsonFileKVStore(path=p)
    assert b.get("k") == {"v": 42}


def test_loads_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "kv.json"
    p.write_text(json.dumps({"k1": "v1", "k2": 2}))
    kv = JsonFileKVStore(path=p)
    assert kv.get("k1") == "v1"
    assert kv.get("k2") == 2


def test_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "kv.json"
    kv = JsonFileKVStore(path=p)
    kv.put("k", "v")
    assert p.exists()
    assert p.parent.exists()


def test_put_many_writes_once(tmp_path: Path) -> None:
    """put_many should batch the disk write."""
    p = tmp_path / "kv.json"
    kv = JsonFileKVStore(path=p)
    kv.put_many({"a": 1, "b": 2, "c": 3})
    on_disk = json.loads(p.read_text())
    assert on_disk == {"a": 1, "b": 2, "c": 3}


def test_overwrites_existing_key(tmp_path: Path) -> None:
    kv = JsonFileKVStore(path=tmp_path / "kv.json")
    kv.put("k", "v1")
    kv.put("k", "v2")
    assert kv.get("k") == "v2"


def test_default_path_is_in_memory(tmp_path: Path, monkeypatch) -> None:
    """If path is None, behaves as in-memory (no disk writes)."""
    monkeypatch.chdir(tmp_path)  # don't pollute cwd
    kv = JsonFileKVStore(path=None)
    kv.put("a", 1)
    assert kv.get("a") == 1
    # no .json file should have been written anywhere
    assert list(tmp_path.glob("**/*.json")) == []
