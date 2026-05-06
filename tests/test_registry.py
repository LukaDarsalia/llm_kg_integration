import pytest

from llm_kg.registry import Registry


class Animal:
    pass


class Dog(Animal):
    pass


class Cat(Animal):
    pass


def test_register_and_get() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    assert reg.get("dog") is Dog


def test_register_decorator_returns_class() -> None:
    reg: Registry[Animal] = Registry("animal")
    decorated = reg.register("dog")(Dog)
    assert decorated is Dog


def test_get_unknown_raises() -> None:
    reg: Registry[Animal] = Registry("animal")
    with pytest.raises(KeyError) as ei:
        reg.get("hippo")
    assert "animal" in str(ei.value)
    assert "hippo" in str(ei.value)


def test_duplicate_registration_raises() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    with pytest.raises(ValueError) as ei:
        reg.register("dog")(Cat)
    assert "dog" in str(ei.value)


def test_names_lists_registered() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    reg.register("cat")(Cat)
    assert sorted(reg.names()) == ["cat", "dog"]


def test_contains() -> None:
    reg: Registry[Animal] = Registry("animal")
    reg.register("dog")(Dog)
    assert "dog" in reg
    assert "cat" not in reg
