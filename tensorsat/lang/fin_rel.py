"""
The language ``fin_rel`` of finite, explicitly enumerated sets (cf. :class:`FinSet`)
and relations between them, represented as Boolean tensors (cf. :class:`FinRel`).
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
from collections.abc import Sequence
from typing import Any, ClassVar, Self, final

import numpy as np
from hashcons import InstanceStore
import xxhash
from ..diagrams import Port, Shape, Type, Box, Wire

if __debug__:
    from typing_validation import validate

type Size = int
"""Type alias for integers used as sizes of :class:`FinSet`s."""

type El = int
"""Type alias for integers used as elements of :class:`FinSet`s."""

type Point = tuple[El, ...]
"""Type alias for tuples of integers, used as points of :class:`FinRel`s."""


@final
class FinSet(Type):
    """
    Type class for finite, explicitly enumerated sets.
    Parametrises sets in the form ``{0, ..., size-1}`` by their ``size >= 1``.
    """

    _store: ClassVar[InstanceStore] = InstanceStore()

    @classmethod
    def _new(cls, size: Size) -> Self:
        """Protected constructor."""
        with FinSet._store.instance(cls, size) as self:
            if self is None:
                self = super().__new__(cls)
                self.__size = size
                FinSet._store.register(self)
            return self

    __size: Size

    __slots__ = ("__size",)

    def __new__(cls, size: int) -> Self:
        """Public constructor."""
        validate(size, int)
        if size <= 0:
            raise ValueError("Finite set size must be strictly positive.")
        return cls._new(size)

    @property
    def size(self) -> Size:
        """Size of the finite set."""
        return self.__size

    def _spider(self, num_ports: int) -> FinRel:
        size = self.__size
        tensor = np.zeros(shape=(size,) * num_ports, dtype=np.uint8)
        for i in range(size):
            tensor[(i,) * num_ports] = 1
        return FinRel._new(tensor)

    def __repr__(self) -> str:
        return f"FinSet({self.__size})"


type NumpyUInt8Array = np.ndarray[tuple[Size, ...], np.dtype[np.uint8]]
"""Type alias for Numpy's UInt8 arrays."""


@final
class FinRel(Box[FinSet]):
    """
    Type class for finite, densely represented relations between finite, explicitly
    enumerated sets.
    Relations are parametrised by their representation as Boolean tensors, where each
    component of the relation corresponds to a component of the tensor.
    """

    @staticmethod
    def _contract2(
        lhs: FinRel,
        lhs_wires: Sequence[Wire],
        rhs: FinRel,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire],
    ) -> FinRel:
        raise NotImplementedError()  # TODO: implement this

    @classmethod
    def _new(
        cls,
        tensor: NumpyUInt8Array,
    ) -> Self:
        """
        Protected constructor.
        Presumes that the tensor is already validated, and that it is not going to be
        accessible from anywhere else (i.e. no copy is performed).
        """
        if tensor.flags["OWNDATA"]:
            tensor.setflags(write=False)
            tensor = tensor.view()
        self = super().__new__(cls)
        self.__tensor = tensor
        self.__shape = Shape(map(FinSet._new, tensor.shape))
        return self

    __tensor: NumpyUInt8Array
    __shape: Shape[FinSet]
    __hash_cache: int
    __is_function_graph_cache: bool

    __slots__ = ("__tensor", "__shape", "__hash_cache", "__is_function_graph_cache")

    def __new__(cls, tensor: NumpyUInt8Array) -> Self:
        """Constructs a relation from a Boolean tensor."""
        assert validate(tensor, NumpyUInt8Array)
        return cls._new(tensor)

    @property
    def tensor(self) -> NumpyUInt8Array:
        """The Boolean tensor defining the relation."""
        return self.__tensor

    @property
    def shape(self) -> Shape[FinSet]:
        """The shape of the relation."""
        return self.__shape

    def _transpose(self, perm: Sequence[Port]) -> Self:
        raise NotImplementedError()  # TODO: implement

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FinRel):
            return NotImplemented
        if self is other:
            return True
        try:
            if self.__hash_cache != other.__hash_cache:
                return False
        except AttributeError:
            pass
        return np.array_equal(self.__tensor, other.__tensor)

    def __hash__(self) -> int:
        """Computes the hash of the finite relation, based on the bytes in the tensor."""
        try:
            return self.__hash_cache
        except AttributeError:
            self.__hash_cache = h = xxhash.xxh64(self.__tensor.data).intdigest()
            return h
