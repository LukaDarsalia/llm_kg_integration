"""Generic name-keyed registry used by every plug-in family in the framework."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-keyed registry of classes.

    Used as the single resolution point between YAML configs (which name plug-ins
    by string) and Python classes. Each plug-in family (LLM provider, vector store,
    method, etc.) owns one Registry instance.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register `cls` under `name`. Raises if `name` already taken."""

        def _decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self._kind} '{name}' already registered "
                    f"(was {self._items[name].__name__}, tried to add {cls.__name__})"
                )
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind} '{name}' "
                f"(known: {sorted(self._items)})"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
