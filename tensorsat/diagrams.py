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

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """Constructs a new type."""
        if cls is Type:
            raise TypeError("Cannot instantiate abstract class Type.")
        return super().__new__(cls)


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

    def __new__(cls, components: Iterable[Type]) -> Self:
        """
        Constructs a new shape with given component types.
        If iterables of types are passed, their types are extracted and inserted
        into the shape at the selected point.
        """
        components = tuple(components)
        assert validate(components, tuple[Type, ...])
        return cls._new(components)

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
            return Shape(self.__components[index])
        assert validate(index, int)
        return self.__components[index]

    def __repr__(self) -> str:
        return f"Shape({self.__components})"


Slot: TypeAlias = int
"""Type alias for (the index of) a slot in a diagram."""

Port: TypeAlias = int
"""Type alias for (the index of) a port in a diagram."""

Wire: TypeAlias = int
"""
Type alias for (the index of) a wire in a diagram.
Each port is connected to exactly one wire, but a wire can connect any number of ports.
"""


class WiringData(TypedDict, total=True):
    """Data for a wiring."""

    num_slot_ports: Sequence[int]
    """Number of ports for each slot."""

    num_outer_ports: int
    """Number of outer ports."""

    wire_types: Sequence[Type]
    """Wire types."""

    slot_mappings: Sequence[Sequence[Wire]]
    """Assignment of a wire to each port of each slot."""

    outer_mapping: Sequence[Wire]
    """Assignment of a wire to each outer port."""


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
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        """Assignment of (the index of) a wire to each port of each slot."""

    @property
    @abstractmethod
    def outer_mapping(self) -> tuple[Wire, ...]:
        """Assignment of (the index of) a wire to each outer port."""

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

    @property
    def num_outer_ports(self) -> int:
        """Number of outer ports."""
        return len(self.outer_shape)

    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @property
    def wires(self) -> tuple[Wire, ...]:
        """Sequence of (the indices of) wires."""
        return tuple(range(self.num_wires))


@final
class Wiring(WiringBase):
    """An immutable wiring."""

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape, ...],
        outer_shape: Shape,
        wire_types: Shape,
        slot_mappings: tuple[tuple[Wire, ...], ...],
        outer_mapping: tuple[Wire, ...],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__slot_shapes = slot_shapes
        self.__outer_shape = outer_shape
        self.__wire_types = wire_types
        self.__slot_mappings = slot_mappings
        self.__outer_mapping = outer_mapping
        return self

    __slot_shapes: tuple[Shape, ...]
    __outer_shape: Shape
    __wire_types: Shape
    __slot_mappings: tuple[tuple[Wire, ...], ...]
    __outer_mapping: tuple[Wire, ...]

    def __new__(cls, data: WiringData) -> Self:
        """Constructs a wiring from the given data."""
        assert validate(data, WiringData)
        # Destructure the data:
        slot_num_ports = tuple(data["num_slot_ports"])
        num_outer_ports = data["num_outer_ports"]
        wire_types = Shape(data["wire_types"])
        slot_mappings = tuple(map(tuple, data["slot_mappings"]))
        outer_mapping = tuple(data["outer_mapping"])
        # Validate the data:
        num_slots = len(slot_num_ports)
        num_wires = len(wire_types)
        if len(slot_mappings) != num_slots:
            raise ValueError(
                "Incorrect number of slot mappings:"
                f" expected {num_slots}, got {len(slot_mappings)}."
            )
        for slot in range(num_slots):
            num_in = slot_num_ports[slot]
            if len(slot_mappings[slot]) != num_in:
                raise ValueError(
                    f"Incorrect number of wires in mapping for slot {slot}:"
                    f" expected {num_in}, got {len(slot_mappings[slot])}."
                )
            for wire in slot_mappings[slot]:
                if wire not in range(num_wires):
                    raise ValueError(
                        f"Invalid wire index {wire} in slot mapping for slot {slot}."
                    )
        if len(outer_mapping) != num_outer_ports:
            raise ValueError(
                "Incorrect number of wires in outer mapping:"
                f" expected {num_outer_ports}, got {len(outer_mapping)}."
            )
        for wire in outer_mapping:
            if wire not in range(num_wires):
                raise ValueError(f"Invalid wire index {wire} in outer mapping.")
        # Create and return the instance:
        slot_shapes = tuple(
            Shape(wire_types[i] for i in range(num_in)) for num_in in slot_num_ports
        )
        outer_shape = Shape(wire_types[o] for o in range(num_outer_ports))
        return cls._new(
            slot_shapes, outer_shape, wire_types, slot_mappings, outer_mapping
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
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        return self.__slot_mappings

    @property
    def outer_mapping(self) -> tuple[Wire, ...]:
        return self.__outer_mapping


@final
class WiringBuilder(WiringBase):
    """A wiring builder."""

    __slot_shapes: list[list[Type]]
    __outer_shape: list[Type]
    __wire_types: list[Type]
    __slot_mappings: list[list[Wire]]
    __outer_mapping: list[Wire]

    def __new__(cls) -> Self:
        """Constructs a blank wiring builder."""
        self = super().__new__(cls)
        self.__slot_shapes = []
        self.__outer_shape = []
        self.__wire_types = []
        self.__slot_mappings = []
        self.__outer_mapping = []
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
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        return tuple(map(tuple, self.__slot_mappings))

    @property
    def outer_mapping(self) -> tuple[Wire, ...]:
        return tuple(self.__outer_mapping)

    @property
    def wiring(self) -> Wiring:
        """The wiring built thus far."""
        return Wiring._new(
            self.slot_shapes,
            self.outer_shape,
            self.wire_types,
            self.slot_mappings,
            self.outer_mapping,
        )

    def copy(self) -> WiringBuilder:
        """Returns a deep copy of this wiring builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__outer_shape = self.__outer_shape.copy()
        clone.__wire_types = self.__wire_types.copy()
        clone.__slot_mappings = [m.copy() for m in self.__slot_mappings]
        clone.__outer_mapping = self.__outer_mapping.copy()
        return clone

    def add_wire(self, t: Type) -> Wire:
        """Adds a new wire with the given type."""
        assert validate(t, Type)
        return self._add_wires([t])[0]

    def add_wires(self, ts: Sequence[Type]) -> tuple[Wire, ...]:
        """Adds new wires with the given types."""
        assert validate(ts, Sequence[Type])
        return self._add_wires(ts)

    def _add_wires(self, ts: Sequence[Type]) -> tuple[Wire, ...]:
        wire_types = self.__wire_types
        len_before = len(wire_types)
        wire_types.extend(ts)
        return tuple(range(len_before, len(wire_types)))

    def _validate_wires(self, wires: Sequence[Wire]) -> None:
        num_wires = self.num_wires
        for wire in wires:
            if wire not in range(num_wires):
                raise ValueError(f"Invalid wire index {wire}.")

    def add_outer_port(self, wire: Wire) -> Port:
        """Adds a new outer port, connected to the given wire."""
        assert validate(wire, Wire)
        self._validate_wires([wire])
        return self.add_outer_ports([wire])[0]

    def add_outer_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        """Adds new outer ports, connected the given wires."""
        assert validate(wires, Sequence[Wire])
        self._validate_wires(wires)
        return self._add_outer_ports(wires)

    def _add_outer_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        outer_shape, wire_types = self.__outer_shape, self.__wire_types
        len_before = len(outer_shape)
        outer_shape.extend(wire_types[wire] for wire in wires)
        self.__outer_mapping.extend(wires)
        return tuple(range(len_before, len(outer_shape)))

    def add_slot(self, wires: Sequence[Wire] = ()) -> Slot:
        """Adds a new slot."""
        assert validate(wires, Sequence[Wire])
        slot_shapes = self.__slot_shapes
        k = len(slot_shapes)
        slot_shapes.append([])
        if wires:
            try:
                self._validate_wires(wires)
                self._add_slot_ports(k, wires)
            except ValueError:
                slot_shapes.pop()  # undo the slot addition
                raise
        return k

    def add_slot_port(self, slot: Slot, wire: Wire) -> Port:
        """Adds a new port for the given slot, connected the given wire."""
        assert validate(wire, Wire)
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        return self.add_slot_ports(slot, [wire])[0]

    def add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> tuple[Port, ...]:
        """Adds new ports for the given slot, connected the given wires."""
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        self._validate_wires(wires)
        return self._add_slot_ports(slot, wires)

    def _add_slot_ports(
        self, slot: Slot, wires: Sequence[Wire]
    ) -> tuple[Port, ...]:
        slot_shape, wire_types = self.__slot_shapes[slot], self.__wire_types
        len_before = len(slot_shape)
        slot_shape.extend(wire_types[w] for w in wires)
        self.__slot_mappings[slot].extend(wires)
        return tuple(range(len_before, len(slot_shape)))
