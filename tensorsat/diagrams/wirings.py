"""
Implementation of wirings and their builders for the :mod:`tensorsat.diagrams` module.
"""

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
from collections.abc import Mapping, Sequence
from itertools import accumulate
from typing import (
    ClassVar,
    Generic,
    Self,
    Type as SubclassOf,
    TypeAlias,
    TypedDict,
    final,
)
from hashcons import InstanceStore

if __debug__:
    from typing_validation import validate

from .types import Type, Shape, TypeT_co


Slot: TypeAlias = int
"""Type alias for (the index of) a slot in a diagram."""

Port: TypeAlias = int
"""Type alias for (the index of) a port in a diagram."""

Wire: TypeAlias = int
"""
Type alias for (the index of) a wire in a diagram.
Each port is connected to exactly one wire, but a wire can connect any number of ports.
"""


class WiringData(Generic[TypeT_co], TypedDict, total=True):
    """Data for a wiring."""

    num_slot_ports: Sequence[int]
    """Number of ports for each slot."""

    num_outer_ports: int
    """Number of outer ports."""

    wire_types: Sequence[TypeT_co]
    """Wire types."""

    slot_mappings: Sequence[Sequence[Wire]]
    """Assignment of a wire to each port of each slot."""

    outer_mapping: Sequence[Wire]
    """Assignment of a wire to each outer port."""


class Shaped(Generic[TypeT_co], ABC):
    """Interface and mixin properties for objects with a shape."""

    @staticmethod
    def wrap_shape[T: Type](shape: Shape[T]) -> Shaped[T]:
        """Wraps a shape into an anonymous :class:`Shaped` instance."""
        assert validate(shape, Shape)
        cls: SubclassOf[Shaped[T]] = final(
            type.__new__(
                type,
                "<anon shaped class>",
                (Shaped,),
                {
                    "shape": property(lambda self: shape),
                    "__repr__": lambda self: "<anon shaped>",
                },
            )
        )
        return cls()

    __slots__ = ()

    @property
    @abstractmethod
    def shape(self) -> Shape[TypeT_co]:
        """Shape of the object."""

    @final
    @property
    def num_ports(self) -> int:
        """Number of ports in the object, aka the length of its shape."""
        return len(self.shape)

    @final
    @property
    def ports(self) -> Sequence[Port]:
        """Sequence of (the indices of) ports in the object."""
        return range(self.num_ports)


class Slotted(Generic[TypeT_co], ABC):
    """Interface and mixin properties/methods for objects with shaped slots."""

    @staticmethod
    def wrap_slot_shapes[T: Type](slot_shapes: tuple[Shape[T], ...]) -> Slotted[T]:
        """Wraps a tuple of shapes into an anonymous :class:`Slotted` instance."""
        assert validate(slot_shapes, tuple[Shape[Type], ...])
        cls: SubclassOf[Slotted[T]] = final(
            type.__new__(
                type,
                "<anon slotted class>",
                (Slotted,),
                {
                    "shape": property(lambda self: slot_shapes),
                    "__repr__": lambda self: "<anon slotted>",
                },
            )
        )
        return cls()

    __slots__ = ()

    @property
    @abstractmethod
    def slot_shapes(self) -> tuple[Shape[TypeT_co], ...]:
        """Shapes for the slots."""

    @final
    @property
    def num_slots(self) -> int:
        """Number of slots."""
        return len(self.slot_shapes)

    @final
    @property
    def slots(self) -> Sequence[Slot]:
        """Sequence of (the indices of) slots."""
        return range(self.num_slots)

    @final
    def num_slot_ports(self, slot: Slot) -> int:
        """Number of ports for the given slot."""
        return len(self.slot_shapes[slot])

    @final
    def slot_ports(self, slot: Slot) -> Sequence[Port]:
        """Sequence of (the indices of) ports for the given slot."""
        return range(self.num_slot_ports(slot))

    @final
    def validate_slot_data(self, data: Mapping[Slot, Shaped[TypeT_co]], /) -> None:
        """Validates the shapes for given slot data."""
        assert validate(data, Mapping[Slot, Shaped[Type]])
        slots, slot_shapes = self.slots, self.slot_shapes
        for slot, shaped in data.items():
            if slot not in slots:
                raise ValueError(f"Invalid slot {slot}.")
            if shaped.shape != slot_shapes[slot]:
                raise ValueError(
                    f"Incompatible shape for data at slot {slot}:"
                    f" expected slot shape {slot_shapes[slot]}, "
                    f" got data with shape {shaped.shape}."
                )


class WiringBase(Shaped[TypeT_co], Slotted[TypeT_co], ABC):
    """Abstract base class for wiring and wiring builder."""

    __slots__ = ("__weakref__",)

    @property
    @abstractmethod
    def wire_types(self) -> Shape[TypeT_co]:
        """Wire types."""

    @property
    @abstractmethod
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        """Assignment of (the index of) a wire to each port of each slot."""

    @property
    @abstractmethod
    def outer_mapping(self) -> tuple[Wire, ...]:
        """Assignment of (the index of) a wire to each outer port."""

    @final
    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @final
    @property
    def wires(self) -> Sequence[Wire]:
        """Sequence of (the indices of) wires."""
        return range(self.num_wires)


@final
class Wiring(WiringBase[TypeT_co]):
    """An immutable wiring."""

    _store: ClassVar[InstanceStore] = InstanceStore()

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape[TypeT_co], ...],
        shape: Shape[TypeT_co],
        wire_types: Shape[TypeT_co],
        slot_mappings: tuple[tuple[Wire, ...], ...],
        outer_mapping: tuple[Wire, ...],
    ) -> Self:
        """Protected constructor."""
        instance_key = (
            slot_shapes,
            shape,
            wire_types,
            slot_mappings,
            outer_mapping,
        )
        with Wiring._store.instance(cls, instance_key) as self:
            if self is None:
                self = super().__new__(cls)
                self.__slot_shapes = slot_shapes
                self.__shape = shape
                self.__wire_types = wire_types
                self.__slot_mappings = slot_mappings
                self.__outer_mapping = outer_mapping
                Wiring._store.register(self)
            return self

    __slot_shapes: tuple[Shape[TypeT_co], ...]
    __shape: Shape[TypeT_co]
    __wire_types: Shape[TypeT_co]
    __slot_mappings: tuple[tuple[Wire, ...], ...]
    __outer_mapping: tuple[Wire, ...]

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_mappings",
        "__outer_mapping",
    )

    def __new__(cls, data: WiringData[TypeT_co]) -> Self:
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
        shape = Shape(wire_types[o] for o in range(num_outer_ports))
        return cls._new(slot_shapes, shape, wire_types, slot_mappings, outer_mapping)

    @property
    def slot_shapes(self) -> tuple[Shape[TypeT_co], ...]:
        return self.__slot_shapes

    @property
    def shape(self) -> Shape[TypeT_co]:
        return self.__shape

    @property
    def wire_types(self) -> Shape[TypeT_co]:
        return self.__wire_types

    @property
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        return self.__slot_mappings

    @property
    def outer_mapping(self) -> tuple[Wire, ...]:
        return self.__outer_mapping

    def compose(self, wirings: Mapping[Slot, Wiring[TypeT_co]]) -> Wiring[TypeT_co]:
        """Composes this wiring with the given wirings for (some of) its slots."""
        assert validate(wirings, Mapping[Slot, Wiring[Type]])
        slots, slot_shapes = self.slots, self.slot_shapes
        for slot, wiring in wirings.items():
            if slot not in slots:
                raise ValueError(f"Invalid slot {slot}.")
            if wiring is not None and wiring.shape != slot_shapes[slot]:
                raise ValueError(
                    f"Incompatible shape in wiring composition for slot {slot}:"
                    f" expected slot shape {self.slot_shapes[slot]},"
                    f" got a wiring of shape {wiring.shape}."
                )
        return self._compose(wirings)

    def _compose(self, wirings: Mapping[Slot, Wiring[TypeT_co]]) -> Wiring[TypeT_co]:
        slots, num_wires = self.slots, self.num_wires
        _wire_start_idx = list(
            accumulate(
                [
                    num_wires,
                    *(
                        wirings[slot].num_wires if slot in wirings else 0
                        for slot in slots
                    ),
                ]
            )
        )
        wiring_remappings = [
            range(start, end)
            for start, end in zip(_wire_start_idx[:-1], _wire_start_idx[1:])
        ]
        new_wire_types = self.wire_types * Shape._prod(
            tuple(wirings[slot].wire_types for slot in slots if slot in wirings)
        )
        new_slots_data: list[tuple[Wiring[TypeT_co], Slot, range]] = []
        for slot in slots:
            if slot in wirings:
                wiring = wirings[slot]
                new_slots_data.extend(
                    (wiring, wiring_slot, wiring_remappings[slot])
                    for wiring_slot in wiring.slots
                )
            else:
                new_slots_data.append((self, slot, range(num_wires)))
        new_slot_shapes = tuple(
            wiring.slot_shapes[wiring_slot] for wiring, wiring_slot, _ in new_slots_data
        )
        new_slot_mappings = tuple(
            tuple(remapping[w] for w in wiring.slot_mappings[wiring_slot])
            for wiring, wiring_slot, remapping in new_slots_data
        )
        return Wiring._new(
            new_slot_shapes,
            self.shape,
            new_wire_types,
            new_slot_mappings,
            self.outer_mapping,
        )

    def __repr__(self) -> str:
        num_wires = self.num_wires
        num_slots = self.num_slots
        num_out_ports = len(self.outer_mapping)
        attrs: list[str] = []
        if num_wires > 0:
            attrs.append(f"{num_wires} wires")
        if num_slots > 0:
            attrs.append(f"{num_slots} slots")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out ports")
        return f"<Wiring {id(self):#x}: {", ".join(attrs)}>"


@final
class WiringBuilder[T: Type](WiringBase[T]):
    """Utility class to build wirings."""

    __slot_shapes: list[list[T]]
    __shape: list[T]
    __wire_types: list[T]
    __slot_mappings: list[list[Wire]]
    __outer_mapping: list[Wire]

    __slot_shapes_cache: tuple[Shape[T], ...] | None
    __shape_cache: Shape[T] | None
    __wire_types_cache: Shape[T] | None
    __slot_mappings_cache: tuple[tuple[Wire, ...], ...] | None
    __outer_mapping_cache: tuple[Wire, ...] | None

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_mappings",
        "__outer_mapping",
        "__slot_shapes_cache",
        "__shape_cache",
        "__wire_types_cache",
        "__slot_mappings_cache",
        "__outer_mapping_cache",
    )

    def __new__(cls) -> Self:
        """Constructs a blank wiring builder."""
        self = super().__new__(cls)
        self.__slot_shapes = []
        self.__shape = []
        self.__wire_types = []
        self.__slot_mappings = []
        self.__outer_mapping = []
        return self

    @property
    def slot_shapes(self) -> tuple[Shape[T], ...]:
        slot_shapes = self.__slot_shapes_cache
        if slot_shapes is None:
            self.__slot_shapes_cache = slot_shapes = tuple(
                Shape._new(tuple(s)) for s in self.__slot_shapes
            )
        return slot_shapes

    @property
    def shape(self) -> Shape[T]:
        shape = self.__shape_cache
        if shape is None:
            self.__shape_cache = shape = Shape._new(tuple(self.__shape))
        return shape

    @property
    def wire_types(self) -> Shape[T]:
        wire_types = self.__wire_types_cache
        if wire_types is None:
            self.__wire_types_cache = wire_types = Shape._new(tuple(self.__wire_types))
        return wire_types

    @property
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        slot_mappings = self.__slot_mappings_cache
        if slot_mappings is None:
            self.__slot_mappings_cache = slot_mappings = tuple(
                map(tuple, self.__slot_mappings)
            )
        return slot_mappings

    @property
    def outer_mapping(self) -> tuple[Wire, ...]:
        outer_mapping = self.__outer_mapping_cache
        if outer_mapping is None:
            self.__outer_mapping_cache = outer_mapping = tuple(self.__outer_mapping)
        return outer_mapping

    @property
    def wiring(self) -> Wiring[T]:
        """The wiring built thus far."""
        return Wiring._new(
            self.slot_shapes,
            self.shape,
            self.wire_types,
            self.slot_mappings,
            self.outer_mapping,
        )

    def copy(self) -> WiringBuilder[T]:
        """Returns a deep copy of this wiring builder."""
        clone: WiringBuilder[T] = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__shape = self.__shape.copy()
        clone.__wire_types = self.__wire_types.copy()
        clone.__slot_mappings = [m.copy() for m in self.__slot_mappings]
        clone.__outer_mapping = self.__outer_mapping.copy()
        clone.__slot_shapes_cache = self.__slot_shapes_cache
        clone.__shape_cache = self.__shape_cache
        clone.__wire_types_cache = self.__wire_types_cache
        clone.__slot_mappings_cache = self.__slot_mappings_cache
        clone.__outer_mapping_cache = self.__outer_mapping_cache
        return clone

    def add_wire(self, t: T) -> Wire:
        """Adds a new wire with the given type."""
        assert validate(t, Type)
        return self._add_wires([t])[0]

    def add_wires(self, ts: Sequence[T]) -> tuple[Wire, ...]:
        """Adds new wires with the given types."""
        assert validate(ts, Sequence[Type])
        return self._add_wires(ts)

    def _add_wires(self, ts: Sequence[T]) -> tuple[Wire, ...]:
        self.__wire_types_cache = None
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
        self.__shape_cache = None
        self.__outer_mapping_cache = None
        shape, wire_types = self.__shape, self.__wire_types
        len_before = len(shape)
        shape.extend(wire_types[wire] for wire in wires)
        self.__outer_mapping.extend(wires)
        return tuple(range(len_before, len(shape)))

    def add_slot(self) -> Slot:
        """Adds a new slot."""
        self.__slot_shapes_cache = None
        slot_shapes = self.__slot_shapes
        k = len(slot_shapes)
        slot_shapes.append([])
        self.__slot_mappings.append([])
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

    def _add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> tuple[Port, ...]:
        self.__slot_shapes_cache = None
        self.__slot_mappings_cache = None
        slot_shape, wire_types = self.__slot_shapes[slot], self.__wire_types
        len_before = len(slot_shape)
        slot_shape.extend(wire_types[w] for w in wires)
        self.__slot_mappings[slot].extend(wires)
        return tuple(range(len_before, len(slot_shape)))

    def __repr__(self) -> str:
        num_wires = self.num_wires
        num_slots = self.num_slots
        num_out_ports = len(self.__outer_mapping)
        attrs: list[str] = []
        if num_wires > 0:
            attrs.append(f"{num_wires} wires")
        if num_slots > 0:
            attrs.append(f"{num_slots} slots")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out ports")
        return f"<WiringBuilder {id(self):#x}: {", ".join(attrs)}>"
