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
from collections import deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import (
    ClassVar,
    Generic,
    Self,
    Type as SubclassOf,
    TypeAlias,
    TypedDict,
    Unpack,
    final,
    override,
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

    wire_types: Sequence[TypeT_co]
    """Wire types."""

    slot_wires_list: Sequence[Sequence[Wire]]
    """Assignment of a wire to each port of each slot."""

    out_wires: Sequence[Wire]
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
    def slot_wires_list(self) -> tuple[tuple[Wire, ...], ...]:
        """Assignment of (the index of) a wire to each port of each slot."""

    @property
    @abstractmethod
    def out_wires(self) -> tuple[Wire, ...]:
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

    def slot_wires(self, slot: Slot) -> tuple[Wire, ...]:
        """Sequence of (the indices of) wires for the given slot."""
        assert validate(slot, Slot)
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        return self.slot_wires_list[slot]

    @final
    @property
    def wired_slot_ports(self) -> Mapping[Wire, tuple[tuple[Slot, Port], ...]]:
        """
        Computes and returns a mapping of wires to the collection of ``(slot, port)``
        pairs connected by that wire.
        """
        wired_slot_ports: dict[Wire, list[tuple[Slot, Port]]] = {}
        for slot, wires in enumerate(self.slot_wires_list):
            for port, wire in enumerate(wires):
                wired_slot_ports.setdefault(wire, []).append((slot, port))
        return MappingProxyType(
            {w: tuple(w_slot_ports) for w, w_slot_ports in wired_slot_ports.items()}
        )

    @final
    @property
    def wired_slots(self) -> Mapping[Wire, tuple[Slot, ...]]:
        """
        Computes and returns a mapping of wires to the collection of slot pairs
        connected by that wire.
        """
        wired_slots: dict[Wire, list[Slot]] = {}
        for slot, wires in enumerate(self.slot_wires_list):
            for wire in wires:
                wired_slots.setdefault(wire, []).append(slot)
        return MappingProxyType(
            {w: tuple(w_slots) for w, w_slots in wired_slots.items()}
        )


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
        slot_wires_list: tuple[tuple[Wire, ...], ...],
        out_wires: tuple[Wire, ...],
    ) -> Self:
        """Protected constructor."""
        instance_key = (
            slot_shapes,
            shape,
            wire_types,
            slot_wires_list,
            out_wires,
        )
        with Wiring._store.instance(cls, instance_key) as self:
            if self is None:
                self = super().__new__(cls)
                self.__slot_shapes = slot_shapes
                self.__shape = shape
                self.__wire_types = wire_types
                self.__slot_wires_list = slot_wires_list
                self.__out_wires = out_wires
                Wiring._store.register(self)
            return self

    __slot_shapes: tuple[Shape[TypeT_co], ...]
    __shape: Shape[TypeT_co]
    __wire_types: Shape[TypeT_co]
    __slot_wires_list: tuple[tuple[Wire, ...], ...]
    __out_wires: tuple[Wire, ...]

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_wires_list",
        "__out_wires",
    )

    def __new__(cls, **data: Unpack[WiringData[TypeT_co]]) -> Self:
        """Constructs a wiring from the given data."""
        assert validate(data, WiringData)
        # Destructure the data:
        wire_types = Shape(data["wire_types"])
        slot_wires_list = tuple(map(tuple, data["slot_wires_list"]))
        out_wires = tuple(data["out_wires"])
        # Validate the data:
        num_slots = len(slot_wires_list)
        num_wires = len(wire_types)
        for slot in range(num_slots):
            for wire in slot_wires_list[slot]:
                if wire not in range(num_wires):
                    raise ValueError(
                        f"Invalid wire index {wire} in slot mapping for slot {slot}."
                    )
        for wire in out_wires:
            if wire not in range(num_wires):
                raise ValueError(f"Invalid wire index {wire} in outer mapping.")
        slot_shapes = tuple(
            Shape(wire_types[i] for i in slot_wires) for slot_wires in slot_wires_list
        )
        shape = Shape(wire_types[o] for o in out_wires)
        return cls._new(slot_shapes, shape, wire_types, slot_wires_list, out_wires)

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
    def slot_wires_list(self) -> tuple[tuple[Wire, ...], ...]:
        return self.__slot_wires_list

    @property
    def out_wires(self) -> tuple[Wire, ...]:
        return self.__out_wires

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
        # 1. Build bipartite graph connecting slot wires of the outer wiring
        #    to outer wires of the wirings plugged into the slots:
        slot_wires_list = self.slot_wires_list
        fwd_mapping: dict[Wire, list[tuple[Slot, Wire]]] = {}
        bwd_mapping: dict[tuple[Slot, Wire], list[Wire]] = {}
        for slot, wiring in wirings.items():
            for self_w, wiring_w in zip(slot_wires_list[slot], wiring.out_wires):
                fwd_mapping.setdefault(self_w, []).append((slot, wiring_w))
                bwd_mapping.setdefault((slot, wiring_w), []).append(self_w)
        # 2. Compute connected component representatives for the bipartite graph,
        #    selecting as representatives the lowest index wire from the outer wiring
        #    appearing in the connected component:
        fwd_cc_repr: dict[Wire, Wire] = {}
        bwd_cc_repr: dict[tuple[Slot, Wire], Wire] = {}
        _wire_q = deque(sorted(fwd_mapping.keys()))
        while _wire_q:
            cc_repr = _wire_q.popleft()
            if cc_repr in fwd_cc_repr:
                continue
            fwd_cc_q: deque[Wire] = deque([cc_repr])
            bwd_cc_q: deque[tuple[Slot, Wire]] = deque([])
            while fwd_cc_q:
                while fwd_cc_q:
                    w = fwd_cc_q.popleft()
                    if w in fwd_cc_repr:
                        continue
                    fwd_cc_repr[w] = cc_repr
                    bwd_cc_q.extend(
                        sw for sw in fwd_mapping[w] if sw not in bwd_cc_repr
                    )
                while bwd_cc_q:
                    sw = bwd_cc_q.popleft()
                    if sw in bwd_cc_repr:
                        continue
                    bwd_cc_repr[sw] = cc_repr
                    fwd_cc_q.extend(w for w in bwd_mapping[sw] if w not in fwd_cc_repr)
        # 3. Remap wire indices after fusion (and store new wire types at the same time):
        wire_remap: dict[Wire, Wire] = {}
        slot_wire_remap: dict[tuple[Slot, Wire], Wire] = {}
        wire_types: list[TypeT_co] = []
        self_wire_types = self.wire_types
        for w in self.wires:
            if w in fwd_cc_repr and w != (w_repr := fwd_cc_repr[w]):
                wire_remap[w] = wire_remap[w_repr]
            else:
                wire_remap[w] = len(wire_types)
                wire_types.append(self_wire_types[w])
        for slot, wiring in wirings.items():
            wiring_wire_types = wiring.wire_types
            for w in wiring.wires:
                if (sw := (slot, w)) in bwd_cc_repr:
                    slot_wire_remap[sw] = wire_remap[bwd_cc_repr[sw]]
                else:
                    slot_wire_remap[sw] = len(wire_types)
                    wire_types.append(wiring_wire_types[w])
        # 4(a). Compute new slot wires for slots left open in outer wiring:
        slot_wires_list = tuple(
            tuple(wire_remap[w] for w in slot_wires)
            for slot, slot_wires in enumerate(self.slot_wires_list)
            if slot not in wirings
        )
        # 4(b). Compute new slot wires for slots of inner wirings:
        for slot, wiring in wirings.items():
            slot_wires_list += tuple(
                tuple(slot_wire_remap[(slot, w)] for w in _slot_wires)
                for _slot_wires in wiring.slot_wires_list
            )
        # 5. Compute new outer wires and return new wiring
        out_wires = tuple(wire_remap[w] for w in self.out_wires)
        return Wiring(
            wire_types=wire_types, slot_wires_list=slot_wires_list, out_wires=out_wires
        )

    def __repr__(self) -> str:
        num_wires = self.num_wires
        num_slots = self.num_slots
        num_out_ports = len(self.out_wires)
        attrs: list[str] = []
        if num_wires > 0:
            attrs.append(f"{num_wires} wire{'s' if num_wires!=1 else ''}")
        if num_slots > 0:
            attrs.append(f"{num_slots} slot{'s' if num_slots!=1 else ''}")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out port{'s' if num_out_ports!=1 else ''}")
        return f"<Wiring {id(self):#x}: {", ".join(attrs)}>"


@final
class WiringBuilder[T: Type](WiringBase[T]):
    """Utility class to build wirings."""

    __slot_shapes: list[list[T]]
    __shape: list[T]
    __wire_types: list[T]
    __slot_wires_list: list[list[Wire]]
    __out_wires: list[Wire]

    __slot_shapes_cache: tuple[Shape[T], ...] | None
    __shape_cache: Shape[T] | None
    __wire_types_cache: Shape[T] | None
    __slot_wires_list_cache: tuple[tuple[Wire, ...], ...] | None
    __out_wires_cache: tuple[Wire, ...] | None

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_wires_list",
        "__out_wires",
        "__slot_shapes_cache",
        "__shape_cache",
        "__wire_types_cache",
        "__slot_wires_list_cache",
        "__out_wires_cache",
    )

    def __new__(cls) -> Self:
        """Constructs a blank wiring builder."""
        self = super().__new__(cls)
        self.__slot_shapes = []
        self.__shape = []
        self.__wire_types = []
        self.__slot_wires_list = []
        self.__out_wires = []
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
    def slot_wires_list(self) -> tuple[tuple[Wire, ...], ...]:
        slot_wires_list = self.__slot_wires_list_cache
        if slot_wires_list is None:
            self.__slot_wires_list_cache = slot_wires_list = tuple(
                map(tuple, self.__slot_wires_list)
            )
        return slot_wires_list

    @property
    def out_wires(self) -> tuple[Wire, ...]:
        out_wires = self.__out_wires_cache
        if out_wires is None:
            self.__out_wires_cache = out_wires = tuple(self.__out_wires)
        return out_wires

    @property
    def wiring(self) -> Wiring[T]:
        """The wiring built thus far."""
        return Wiring._new(
            self.slot_shapes,
            self.shape,
            self.wire_types,
            self.slot_wires_list,
            self.out_wires,
        )

    @override
    def slot_wires(self, slot: Slot) -> tuple[Wire, ...]:
        # Overridden for more efficient implementation.
        assert validate(slot, Slot)
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        return tuple(self.__slot_wires_list[slot])

    def copy(self) -> WiringBuilder[T]:
        """Returns a deep copy of this wiring builder."""
        clone: WiringBuilder[T] = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__shape = self.__shape.copy()
        clone.__wire_types = self.__wire_types.copy()
        clone.__slot_wires_list = [m.copy() for m in self.__slot_wires_list]
        clone.__out_wires = self.__out_wires.copy()
        clone.__slot_shapes_cache = self.__slot_shapes_cache
        clone.__shape_cache = self.__shape_cache
        clone.__wire_types_cache = self.__wire_types_cache
        clone.__slot_wires_list_cache = self.__slot_wires_list_cache
        clone.__out_wires_cache = self.__out_wires_cache
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

    def add_out_port(self, wire: Wire) -> Port:
        """Adds a new outer port, connected to the given wire."""
        assert validate(wire, Wire)
        self._validate_wires([wire])
        return self.add_out_ports([wire])[0]

    def add_out_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        """Adds new outer ports, connected the given wires."""
        assert validate(wires, Sequence[Wire])
        self._validate_wires(wires)
        return self._add_out_ports(wires)

    def _add_out_ports(self, wires: Sequence[Wire]) -> tuple[Port, ...]:
        self.__shape_cache = None
        self.__out_wires_cache = None
        shape, wire_types = self.__shape, self.__wire_types
        len_before = len(shape)
        shape.extend(wire_types[wire] for wire in wires)
        self.__out_wires.extend(wires)
        return tuple(range(len_before, len(shape)))

    def add_slot(self) -> Slot:
        """Adds a new slot."""
        self.__slot_shapes_cache = None
        slot_shapes = self.__slot_shapes
        k = len(slot_shapes)
        slot_shapes.append([])
        self.__slot_wires_list.append([])
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
        self.__slot_wires_list_cache = None
        slot_shape, wire_types = self.__slot_shapes[slot], self.__wire_types
        len_before = len(slot_shape)
        slot_shape.extend(wire_types[w] for w in wires)
        self.__slot_wires_list[slot].extend(wires)
        return tuple(range(len_before, len(slot_shape)))

    def __repr__(self) -> str:
        num_wires = self.num_wires
        num_slots = self.num_slots
        num_out_ports = len(self.__out_wires)
        attrs: list[str] = []
        if num_wires > 0:
            attrs.append(f"{num_wires} wire{'s' if num_wires!=1 else ''}")
        if num_slots > 0:
            attrs.append(f"{num_slots} slot{'s' if num_slots!=1 else ''}")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out port{'s' if num_out_ports!=1 else ''}")
        return f"<WiringBuilder {id(self):#x}: {", ".join(attrs)}>"
