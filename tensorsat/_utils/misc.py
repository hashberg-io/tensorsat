"""
Miscellaneous utility functions, classes and types for TensorSat, for internal use.

.. warning::

    This module is subject to frequent refactoring.

"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import product
from typing import Any, cast, Mapping, ParamSpec, Type, TypeVar
import numpy as np


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


def rewire_array[
    T: np.dtype[Any]
](a: np.ndarray[Any, T], out_ports: Sequence[int]) -> np.ndarray[Any, T]:
    """Rewires an array, with ports duplication and discarding."""
    a_shape = a.shape
    a_ports = range(len(a_shape))
    b_shape = tuple(a_shape[port] for port in out_ports)
    b = cast(np.ndarray[Any, T], np.zeros(b_shape, dtype=a.dtype))
    repeat_ports = sorted(p for p, c in Counter(out_ports).items() if c > 1)
    repeat_ports += sorted(set(a_ports) - set(out_ports))
    port_i = {p: i for i, p in enumerate(repeat_ports)}
    for _idxs in product(*(range(a_shape[p]) for p in repeat_ports)):
        a_idxs = tuple(
            _idxs[port_i[p]] if p in port_i else slice(None) for p in a_ports
        )
        b_idxs = tuple(
            _idxs[port_i[p]] if p in port_i else slice(None) for p in out_ports
        )
        b[*b_idxs] += a[*a_idxs]
    return b
