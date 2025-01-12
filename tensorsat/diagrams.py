"""
Hybrid diagrams for compact-closed categories.
"""

# Part of TensorSAT

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Self, TypeAlias, TypedDict, final, overload

if __debug__:
    from typing_validation import validate


class Type(ABC):
    """
    Abstract base class for types in diagrams.

    Types are primarily used to signal compatibility of contractions, but they can also
    carry useful additional information, such as the cardinality of a set or the
    dimensionality of a vector space.
    """

    __slots__ = ()


@final
class Shape:
    """
    A Shape, as a finite tuple of types.
    """

    __components: tuple[Type, ...]

    __slots__ = ("__weakref__", "__components")

    @classmethod
    def _new(cls, components: tuple[Type, ...]) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__components = components
        return self

    def __new__(cls, *components: Type | Iterable[Type]) -> Self:
        """
        Constructs a new shape with given component types.
        If iterables of types are passed, their types are extracted and inserted
        into the shape at the selected point.
        """
        _components = sum(
            ((c,) if isinstance(c, Type) else tuple(c) for c in components), ()
        )
        assert validate(_components, tuple[Type, ...])
        return cls._new(_components)

    def __add__(self, rhs: Shape, /) -> Shape:
        """Concatenates two shapes."""
        assert validate(rhs, Shape)
        return Shape(*self, *rhs)

    def __mul__(self, rhs: int, /) -> Shape:
        """Repeats a shape a given number of times."""
        assert validate(rhs, int)
        return Shape(tuple(self) * rhs)

    def __iter__(self) -> Iterator[Type]:
        """Iterates over the components of the shape."""
        return iter(self.__components)

    def __len__(self) -> int:
        """Returns the number of components in the shape."""
        return len(self.__components)

    @overload
    def __getitem__(self, index: int, /) -> Type: ...
    @overload
    def __getitem__(self, index: slice, /) -> Shape: ...
    def __getitem__(self, index: int | slice, /) -> Type | Shape:
        """Returns the component(s) at the given index(es)."""
        if isinstance(index, slice):
            return Shape(*self.__components[index])
        assert validate(index, int)
        return self.__components[index]

    def __repr__(self) -> str:
        return f"Shape{self.__components}"


Slot: TypeAlias = int
"""Type alias for (the index of) a slot in a diagram."""

SlotPort: TypeAlias = tuple[Slot, int]
"""
Type alias for (the multi-index of) a slot port in a diagram,
as a pair of the slot and the port within the slot.
"""

Port: TypeAlias = int
"""Type alias for (the index of) a port in a diagram."""

Wire: TypeAlias = int
"""Type alias for (the index of) a wire in a diagram."""


class WiringData(TypedDict, total=True):
    """Data for a wiring."""

    num_slot_ports: Sequence[int]
    """Number of ports for each slot."""

    num_outer_ports: int
    """Number of outer ports."""

    wire_types: Sequence[Type]
    """Wire types."""

    slot_mapping: Mapping[SlotPort, Wire]
    """Mapping of slot ports to wires."""

    outer_mapping: Mapping[Port, Wire]
    """Mapping of output ports to wires."""


class WiringBase(ABC):
    """Abstract base class for wiring and wiring builder."""

    @property
    @abstractmethod
    def slot_shapes(self) -> tuple[Shape, ...]:
        """Shapes for the slots."""

    @property
    @abstractmethod
    def outer_shape(self) -> Shape:
        """Outer shape."""

    @property
    @abstractmethod
    def wire_types(self) -> Shape:
        """Wire types."""

    @property
    @abstractmethod
    def slot_mapping(self) -> Mapping[SlotPort, Wire]:
        """Mapping of slot ports to (indices of) wires."""

    @property
    @abstractmethod
    def outer_mapping(self) -> Mapping[Port, Wire]:
        """Mapping of outer ports to (indices of) wires."""

    @property
    def num_slots(self) -> int:
        """Number of slots."""
        return len(self.slot_shapes)

    @property
    def slots(self) -> tuple[Slot, ...]:
        """Sequence of (the indices of) slots."""
        return tuple(range(self.num_slots))

    def num_slot_ports(self, slot: Slot) -> int:
        """Number of ports for the given slot."""
        return len(self.slot_shapes[slot])

    def slot_ports(self, slot: Slot) -> tuple[Wire, ...]:
        """Tuple of (indices of) wires to which ports for the slot are connected."""
        num_inputs = self.num_slot_ports(slot)
        slot_mapping = self.slot_mapping
        return tuple(slot_mapping[slot, i] for i in range(num_inputs))

    @property
    def num_outer_ports(self) -> int:
        """Number of outer ports."""
        return len(self.outer_shape)

    @property
    def outer_ports(self) -> tuple[Wire, ...]:
        """Tuple of (indices of) wires to which outer ports are connected."""
        outer_mapping = self.outer_mapping
        return tuple(outer_mapping[o] for o in range(self.num_outer_ports))

    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @property
    def wires(self) -> tuple[Wire, ...]:
        """Sequence of (the indices of) wires."""
        return tuple(range(self.num_wires))


class Wiring(WiringBase):
    """An immutable wiring."""

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape, ...],
        outer_shape: Shape,
        wire_types: Shape,
        slot_mapping: MappingProxyType[SlotPort, Wire],
        outer_mapping: MappingProxyType[Port, Wire],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__slot_shapes = slot_shapes
        self.__outer_shape = outer_shape
        self.__wire_types = wire_types
        self.__slot_mapping = slot_mapping
        self.__outer_mapping = outer_mapping
        return self

    __slot_shapes: tuple[Shape, ...]
    __outer_shape: Shape
    __wire_types: Shape
    __slot_mapping: Mapping[SlotPort, Wire]
    __outer_mapping: Mapping[Port, Wire]

    def __new__(cls, data: WiringData) -> Self:
        """Constructs a wiring from the given data."""
        assert validate(data, WiringData)
        # Destructure the data:
        slot_num_ports = tuple(data["num_slot_ports"])
        num_ports = data["num_outer_ports"]
        wire_types = Shape(*data["wire_types"])
        slot_mapping = MappingProxyType(data["slot_mapping"])
        outer_mapping = MappingProxyType(data["outer_mapping"])
        # Validate the data:
        num_wires = len(wire_types)
        if set(slot_mapping.keys()) != {
            (k, i) for k, num_in in enumerate(slot_num_ports) for i in range(num_in)
        }:
            raise ValueError("Incorrect domain for input wiring mapping.")
        if not all(0 <= w < num_wires for w in slot_mapping.values()):
            raise ValueError("Incorrect image for input wiring mapping.")
        if set(outer_mapping.keys()) != set(range(num_ports)):
            raise ValueError("Incorrect domain for output wiring mapping.")
        if not all(0 <= w < num_wires for w in outer_mapping.values()):
            raise ValueError("Incorrect image for output wiring mapping.")
        # Create and return the instance:
        slot_shapes = tuple(
            Shape._new(tuple(wire_types[i] for i in range(num_in)))
            for num_in in slot_num_ports
        )
        outer_shape = Shape._new(tuple(wire_types[o] for o in range(num_ports)))
        return cls._new(
            slot_shapes, outer_shape, wire_types, slot_mapping, outer_mapping
        )

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        return self.__slot_shapes

    @property
    def outer_shape(self) -> Shape:
        return self.__outer_shape

    @property
    def wire_types(self) -> Shape:
        return self.__wire_types

    @property
    def slot_mapping(self) -> Mapping[SlotPort, Wire]:
        return self.__slot_mapping

    @property
    def outer_mapping(self) -> Mapping[Port, Wire]:
        return self.__outer_mapping


class WiringBuilder(WiringBase):
    """A wiring builder."""

    __slot_shapes: list[list[Type]]
    __outer_shape: list[Type]
    __wire_types: list[Type]
    __slot_mapping: dict[SlotPort, Wire]
    __outer_mapping: dict[Port, Wire]

    def __new__(cls) -> Self:
        """Constructs a blank wiring builder."""
        self = super().__new__(cls)
        self.__slot_shapes = []
        self.__outer_shape = []
        self.__wire_types = []
        self.__slot_mapping = {}
        self.__outer_mapping = {}
        return self

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        return tuple(Shape(shape) for shape in self.__slot_shapes)

    @property
    def outer_shape(self) -> Shape:
        return Shape(self.__outer_shape)

    @property
    def wire_types(self) -> Shape:
        return Shape(self.__wire_types)

    @property
    def slot_mapping(self) -> MappingProxyType[SlotPort, Wire]:
        return MappingProxyType(self.__slot_mapping)

    @property
    def outer_mapping(self) -> MappingProxyType[Port, Wire]:
        return MappingProxyType(self.__outer_mapping)

    @property
    def wiring(self) -> Wiring:
        """The wiring built thus far."""
        return Wiring._new(
            self.slot_shapes,
            self.outer_shape,
            self.wire_types,
            self.slot_mapping,
            self.outer_mapping,
        )

    def clone(self) -> WiringBuilder:
        """Copy of this wiring builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__outer_shape = self.__outer_shape.copy()
        clone.__wire_types = self.__wire_types.copy()
        clone.__slot_mapping = self.__slot_mapping.copy()
        clone.__outer_mapping = self.__outer_mapping.copy()
        return clone

    def add_wire(self, t: Type) -> Wire:
        """Adds a new wire with the given type."""
        return self._add_wires([t])[0]

    def add_wires(self, ts: Sequence[Type]) -> tuple[Wire, ...]:
        """Adds new wires with the given types."""
        return self._add_wires(ts)

    def _add_wires(self, ts: Sequence[Type]) -> tuple[Wire, ...]:
        wire_types = self.__wire_types
        len_before = len(wire_types)
        wire_types.extend(ts)
        return tuple(range(len_before, len(wire_types)))

    def add_outer_port(self, wire: Wire) -> Port:
        """Adds a new outer port, connected to the given wire."""
        if not 0 <= wire < len(self.__wire_types):
            raise ValueError(f"Invalid wire index {wire}.")
        return self._add_outer_ports([wire])[0]

    def add_outer_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        """Adds new outer ports, connected the given wires."""
        num_wires = len(self.__wire_types)
        if not all(0 <= w < num_wires for w in wires):
            raise ValueError("Invalid wire index.")
        return self._add_outer_ports(wires)

    def _add_outer_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        outer_shape, wire_types = self.__outer_shape, self.__wire_types
        len_before = len(outer_shape)
        outer_shape.extend(wire_types[w] for w in wires)
        new_outputs = tuple(range(len_before, len(outer_shape)))
        self.__outer_mapping.update(zip(new_outputs, wires))
        return new_outputs

    @overload
    def add_slot(self) -> Slot: ...
    @overload
    def add_slot(self, wires: Sequence[Wire]) -> tuple[Slot, tuple[SlotPort, ...]]: ...
    def add_slot(
        self, wires: Sequence[Wire] | None = None
    ) -> Slot | tuple[Slot, tuple[SlotPort, ...]]:
        """Adds a new slot."""
        k = len(self.__slot_shapes)
        self.__slot_shapes.append([])
        if wires is None:
            return k
        try:
            new_inputs = self.add_slot_ports(k, wires)
        except ValueError:
            self.__slot_shapes.pop()
            raise
        return k, new_inputs

    def add_slot_port(self, slot: Slot, wire: Wire) -> SlotPort:
        """Adds a new port for the given slot, connected the given wire."""
        return self.add_slot_ports(slot, (wire,))[0]

    def add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> tuple[SlotPort, ...]:
        """Adds new ports for the given slot, connected the given wires."""
        if not 0 <= slot < len(self.__slot_shapes):
            raise ValueError(f"Invalid inner slot index {slot}.")
        if not all(0 <= w < len(self.__wire_types) for w in wires):
            raise ValueError("Invalid wire index.")
        return self._add_slot_ports(slot, wires)

    def _add_slot_ports(
        self, slot: Slot, wires: Sequence[Wire]
    ) -> tuple[SlotPort, ...]:
        slot_shape, wire_types = self.__slot_shapes[slot], self.__wire_types
        len_before = len(slot_shape)
        slot_shape.extend(wire_types[w] for w in wires)
        new_inputs = tuple((slot, i) for i in range(len_before, len(slot_shape)))
        self.__slot_mapping.update(zip(new_inputs, wires))
        return new_inputs
