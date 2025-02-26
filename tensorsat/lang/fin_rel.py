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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any, ClassVar, Self, TypeAlias, final

import numpy as np
from hashcons import InstanceStore
import xxhash

from ..diagrams import Port, Shape, Type, Box, Wire

if __debug__:
    from typing_validation import validate

Size: TypeAlias = int
"""Type alias for integers used as sizes of :class:`FinSet`."""

El: TypeAlias = int
"""Type alias for integers used as elements of :class:`FinSet`."""

Point: TypeAlias = tuple[El, ...]
"""Type alias for tuples of integers, used as points of :class:`FinRel`."""


def _wrap_el(el_or_point: El | Point, /) -> Point:
    """
    Wraps an element of a finite set into a singleton point of a relation.
    If a point is passed, it is returned unchanged.
    """
    if isinstance(el_or_point, (int, np.integer)):
        return (el_or_point,)
    assert validate(el_or_point, Point)
    return el_or_point


type ItemOrIterable[T] = T | Iterable[T]


def _extract_sizes(
    sizes_or_finsets: ItemOrIterable[Size | FinSet], /
) -> tuple[Size, ...]:
    if isinstance(sizes_or_finsets, int):
        return (sizes_or_finsets,)
    if isinstance(sizes_or_finsets, FinSet):
        return (sizes_or_finsets.size,)
    return tuple(
        size if isinstance(size, int) else size.size for size in sizes_or_finsets
    )


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
        """
        Public constructor.

        :meta public:
        """
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


NumpyUInt8Array: TypeAlias = np.ndarray[tuple[Size, ...], np.dtype[np.uint8]]
"""Type alias for Numpy's UInt8 arrays."""


@final
class FinRel(Box[FinSet]):
    """
    Type class for finite, densely represented relations between finite, explicitly
    enumerated sets.
    Relations are parametrised by their representation as Boolean tensors, where each
    component of the relation corresponds to a component of the tensor.
    """

    @classmethod
    def from_set(
        cls,
        shape: ItemOrIterable[Size | FinSet],
        points: Iterable[El | Point],
    ) -> Self:
        """Constructs a relation from a set of points."""
        shape = _extract_sizes(shape)
        assert validate(shape, tuple[Size, ...])
        data = np.zeros(shape, dtype=np.uint8)
        if any(dim == 0 for dim in data.shape):
            raise ValueError("Zero dimension in shape.")
        for point in points:
            if isinstance(point, int):
                point = (point,)
            assert validate(point, Point)
            if len(point) != len(shape):
                raise ValueError(f"Length of {point = } is invalid for {shape = }.")
            if not all(0 <= i < d for i, d in zip(point, data.shape)):
                raise ValueError(f"Values of {point = } are invalid for {shape = }.")
            data[point] = 1
        return cls._new(data)

    @classmethod
    def from_mapping(
        cls,
        input_shape: ItemOrIterable[Size | FinSet],
        output_shape: ItemOrIterable[Size | FinSet],
        mapping: Mapping[Point, El | Point],
    ) -> Self:
        """
        Constructs a function graph from a mapping of points to points.
        The relation shape is given by ``input_shape + output_shape``.
        """
        input_shape = _extract_sizes(input_shape)
        output_shape = _extract_sizes(output_shape)
        rel = cls.from_set(
            input_shape + output_shape,
            (k + _wrap_el(v) for k, v in mapping.items()),
        )
        if len(mapping) != prod(input_shape):
            raise ValueError("Mapping does not cover the entire input space.")
        return rel

    @classmethod
    def singleton(
        cls,
        shape: FinSet | Iterable[Size],
        point: El | Point,
    ) -> Self:
        """Constructs a singleton relation with the given point."""
        return cls.from_mapping((), shape, {(): point})

    @classmethod
    def from_callable(
        cls,
        input_shape: ItemOrIterable[Size | FinSet],
        output_shape: ItemOrIterable[Size | FinSet],
        func: Callable[..., El | Point],
        # Without type bounds for ParamSpec or TypeVarTuple,
        # there is no good way to type this as a function which can take
        # any fixed number of El arguments, but we don't care how many.
    ) -> Self:
        """
        Constructs a function graph from a callable mapping points to points
        The relation shape is given by ``input_shape + output_shape``.

        The function passed is expected to take exactly as many positional arguments
        as the length of ``input_shape``, all of type :obj:`El` (i.e. integers).
        """
        input_shape = _extract_sizes(input_shape)
        output_shape = _extract_sizes(output_shape)
        mapping = {idx: func(*idx) for idx in np.ndindex(input_shape)}
        return cls.from_mapping(input_shape, output_shape, mapping)

    @classmethod
    def from_wiring(
        cls,
        out_mapping: Sequence[Wire],
        wire_types: Mapping[Port, Size | FinSet],
    ) -> Self:
        """Creates the spider relation for the given wiring."""
        # 1. Extract and validate wires and their sizes:
        assert validate(out_mapping, Sequence[Wire])
        assert validate(wire_types, Mapping[Port, Size | FinSet])
        wire_sizes = dict(zip(wire_types.keys(), _extract_sizes(wire_types.values())))
        wires = sorted(set(out_mapping))
        for wire in wires:
            if wire not in wire_sizes:
                raise ValueError(f"Size missing for wire {wire}.")
        if any(dim <= 0 for dim in wire_sizes.values()):
            raise ValueError("Wire sizes must be strictly positive.")
        # 2. Re-index the wires:
        _wire_idx = {wire: i for i, wire in enumerate(wires)}
        out_mapping = [_wire_idx[node] for node in out_mapping]
        wire_sizes = {_wire_idx[node]: wire_sizes[node] for node in wires}
        # 3. Construct and return the relation:
        ports = range(len(out_mapping))
        shape = tuple(wire_sizes[out_mapping[port]] for port in ports)
        subset = frozenset(
            tuple(values[out_mapping[port]] for port in ports)
            for values in product(*(range(wire_sizes[node]) for node in wires))
        )
        return cls.from_set(shape, subset)

    @staticmethod
    def _contract2(
        lhs: FinRel,
        lhs_wires: Sequence[Wire],
        rhs: FinRel,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire],
    ) -> FinRel:
        lhs_tensor = lhs.__tensor
        rhs_tensor = rhs.__tensor
        contracted_size = prod(
            dim
            for dim, w in zip(lhs_tensor.shape, lhs_wires)
            if w in (set(lhs_wires) & set(rhs_wires)) - set(out_wires)
        )
        if contracted_size >= 256:
            dt: np.dtype[Any]
            if contracted_size < 2**16:
                dt = np.dtype("uint16")
            elif contracted_size < 2**32:
                dt = np.dtype("uint32")
            else:
                dt = np.dtype("uint64")
            lhs_tensor, rhs_tensor = lhs_tensor.astype(dt), rhs_tensor.astype(dt)
        res_tensor = np.einsum(lhs_tensor, lhs_wires, rhs_tensor, rhs_wires, out_wires)
        res_tensor = np.sign(res_tensor, dtype=np.uint8)
        return FinRel._new(res_tensor)

    @classmethod
    def _new(cls, tensor: NumpyUInt8Array) -> Self:
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

    __slots__ = ("__tensor", "__shape", "__hash_cache", "__is_function_graph_cache")

    def __new__(cls, tensor: NumpyUInt8Array) -> Self:
        """
        Constructs a relation from a Boolean tensor.

        :meta public:
        """
        assert validate(tensor, NumpyUInt8Array)
        if not np.all(tensor <= 1):
            raise ValueError("Values in a Boolean tensor must be 0 or 1.")
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
        return FinRel._new(np.transpose(self.__tensor, perm))

    def to_set(self) -> Iterator[Point]:
        """
        Iterates over the subset of points in the relation.
        An inverse to the constructor :meth:`FinRel.from_set`.
        """
        tensor = self.tensor
        for idxs in np.ndindex(tensor.shape):
            if tensor[*idxs]:
                yield tuple(map(int, idxs))

    def _function_matrix(
        self, input_ports: tuple[Port, ...], /
    ) -> tuple[tuple[Port, ...], NumpyUInt8Array]:
        ports = self.ports
        if not all(p in ports for p in input_ports):
            raise ValueError("Invalid input ports.")
        if len(input_ports) != len(set(input_ports)):
            raise ValueError("Input ports cannot be repeated.")
        output_ports = tuple(p for p in ports if p not in input_ports)
        input_sizes = tuple(s.size for s in self.shape[input_ports])
        output_sizes = tuple(s.size for s in self.shape[output_ports])
        transposed_tensor = np.transpose(self.__tensor, input_ports + output_ports)
        return (
            output_ports,
            transposed_tensor.reshape(prod(input_sizes), prod(output_sizes)),
        )

    def is_function_graph(self, input_ports: Sequence[Port], /) -> bool:
        """
        Whether the relation is a function graph in the case where the given ports
        are taken to be inputs and the remaining ports are taken to be outputs.
        """
        assert validate(input_ports, Sequence[Port])
        _, matrix = self._function_matrix(tuple(input_ports))
        return bool(np.all(np.count_nonzero(matrix, axis=1) == 1))

    def to_mapping(self, input_ports: Sequence[Port], /) -> Mapping[Point, El | Point]:
        """
        Returns an input-output mapping for this relation, where the given ports are
        selected as the input ports.
        An inverse to the constructor :meth:`FinRel.from_mapping`.

        :raises ValueError: if the relation is not a function graph for the given
                            selection of input ports (see :meth:`is_function_graph`).
        """
        assert validate(input_ports, Sequence[Port])
        input_ports = tuple(input_ports)
        output_ports, matrix = self._function_matrix(input_ports)
        if not np.all(np.count_nonzero(matrix, axis=1) == 1):
            raise ValueError(
                "Relation is not a function graph with the given input ports."
            )
        input_sizes = tuple(s.size for s in self.shape[input_ports])
        output_sizes = tuple(s.size for s in self.shape[output_ports])
        out_points = list(np.ndindex(output_sizes))
        return {
            in_point: out_points[np.argmax(col)]
            for in_point, col in zip(np.ndindex(input_sizes), matrix)
        }

    def to_callable(self, input_ports: Sequence[Port], /) -> Callable[..., El | Point]:
        """
        Returns an input-output callable for this relation, where the given ports are
        selected as the input ports.
        An inverse to the constructor :meth:`FinRel.from_callable`.

        :raises ValueError: if the relation is not a function graph for the given
                            selection of input ports (see :meth:`is_function_graph`).
        """
        mapping = self.to_mapping(input_ports)
        args_list = ", ".join(f"i{i}" for i in input_ports)
        code = f"func = lambda {args_list}: mapping[({args_list})]"
        namespace = {"mapping": mapping}
        exec(code, namespace)
        return namespace["func"]  # type: ignore

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
