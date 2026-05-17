"""JSON-file-backed key-value store.

Loaded once on construction (if `path` exists), written through to disk on
every `put`/`put_many`. Suitable for ~tens of thousands of small entries
(chunk text, entity descriptions); for larger or higher-write-rate workloads
use SQLite/RocksDB.

`path=None` makes it pure in-memory (useful for tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_kg.storage import KV_REGISTRY
from llm_kg.storage.base import KVStore


@KV_REGISTRY.register("jsonfile")
class JsonFileKVStore(KVStore):
    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._data: dict[str, Any] = {}
        if self._path is not None and self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, key: str) -> Any:
        return self._data[key]  # raises KeyError on miss, per contract

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def put_many(self, items: dict[str, Any]) -> None:
        self._data.update(items)
        self._flush()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def _flush(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False))
