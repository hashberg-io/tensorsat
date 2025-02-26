"""Assorted utility functions, classes and types for TensorSAT."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Mapping, ParamSpec, Type, TypeVar


type ValueSetter[K, V] = V | Callable[[K], V] | Mapping[K, V]
"""
A value setter, which can be one of:

- a constant value
- a callable, producing a value from a key
- a mapping of keys to values

A callable setter can raise :class:`KeyError` to signal that a value cannot be
produced on some given key.
"""

P = ParamSpec("P")
"""Param specification variable."""

R = TypeVar("R")
"""Invariant type variable."""

S = TypeVar("S")
"""Invariant type variable."""


def default_on_error(
    fun: Callable[P, R],
    default: dict[Type[Exception], S],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | S:
    """Calls a function, returning a given default value in case of the given error."""
    try:
        return fun(*args, **kwargs)
    except tuple(default) as e:
        return default[type(e)]


def apply_setter[_K, _V](setter: ValueSetter[_K, _V], k: _K) -> _V | None:
    """
    Applies a setter to the given key.
    Returns :obj:`None` if the setter could not produce a value on the given key.
    """
    if callable(setter):
        return default_on_error(setter, {KeyError: None}, k)
    if isinstance(setter, Mapping):
        return setter.get(k)
    return setter


def dict_deep_copy[_T](val: _T) -> _T:
    """Utility function for deep copy of nested dictionaries."""
    if type(val) != dict:  # noqa: E721
        # T != dict[K, V] => return == T
        return val
    # T == dict[K, V] => return == dict[K, V] (by induction)
    return {k: dict_deep_copy(v) for k, v in val.items()}  # type: ignore[return-value]


def dict_deep_update(to_update: Any, new: Any) -> Any:
    """
    Utility function for deep update of nested dictionaries.
    Behaviour depends on the types of the arguments:

    - if ``type(to_update) == dict`` and ``type(new) == dict``,
      the the function recursively deep updates ``to_update`` and returns it;
    - otherwise, the function makes no change and returns ``new``.
    """
    if type(to_update) != dict or type(new) != dict:  # noqa: E721
        return new
    to_update.update(
        {k: dict_deep_update(v, new[k]) for k, v in to_update.items() if k in new}
    )
    return to_update
