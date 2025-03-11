"""
Implementation of core diagrammatic data structures.
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
from abc import abstractmethod
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    Self,
    Type as SubclassOf,
    TypeAlias,
    TypeVar,
    TypedDict,
    Unpack,
    cast,
    final,
)
from weakref import WeakValueDictionary
from hashcons import InstanceStore

from ._utils.meta import TensorSatMeta, InheritanceForestMeta, cached_property

if TYPE_CHECKING:
    from .contractions import Contraction
else:
    Contraction = Any

if __debug__:
    from typing_validation import validate


class TypeMeta(TensorSatMeta):
    """Metaclass for typeclasses."""


class Type(metaclass=TypeMeta):
    """
    Abstract base class for types in diagrams.

    Types are used to signal compatibility between boxes, by requiring that ports wired
    together in a diagram all have the same type.
    By sharing common types, boxes from multiple languages can be wired together in the
    same diagram.
    """

    def __new__(cls) -> Self:
        """
        Constructs a new type.

        :meta public:
        """
        return super().__new__(cls)


class TensorLikeType(Type):
    """
    Abstract base class for tensor-like types, endowed with a dimension which can be
    used to estimate contraction cost and memory requirements for boxes.
    See :class:`TensorLikeBox`.
    """

    @property
    @abstractmethod
    def tensor_dim(self) -> int:
        """The tensor-like dimension for this object."""


TypeClass: TypeAlias = SubclassOf[Type]
"""A type class, i.e. a subclass of :class:`Type`."""

Shape: TypeAlias = tuple[Type, ...]
"""A shape, as a tuple of types."""

Slot: TypeAlias = int
"""Type alias for (the index of) a slot in a wiring/diagram."""

Port: TypeAlias = int
"""Type alias for (the index of) a port in a wiring/diagram."""

Wire: TypeAlias = int
"""
Type alias for (the index of) a wire in a wiring/diagram.

Each port is connected to exactly one wire, but a wire can connect any number of ports.
"""

Slots: TypeAlias = tuple[Slot, ...]
"""Type alias for a fixed sequence of slots in a wiring/diagram."""

Ports: TypeAlias = tuple[Port, ...]
"""Type alias for a fixed sequence of ports in a wiring/diagram."""

Wires: TypeAlias = tuple[Wire, ...]
"""Type alias for a fixed sequence of wires in a wiring/diagram."""


class WiringData(TypedDict, total=True):
    """Data for a wiring."""

    wire_types: Sequence[Type]
    """Wire types."""

    slot_wires_list: Sequence[Sequence[Wire]]
    """Assignment of a wire to each port of each slot."""

    out_wires: Sequence[Wire]
    """Assignment of a wire to each outer port."""


class Shaped(metaclass=TensorSatMeta):
    """Interface and mixin properties for objects with a shape."""

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Shape of the object."""

    @property
    def num_ports(self) -> int:
        """Number of ports in the object, aka the length of its shape."""
        return len(self.shape)

    @property
    def ports(self) -> Sequence[Port]:
        """Sequence of (the indices of) ports in the object."""
        return range(self.num_ports)


@final
class Wiring(Shaped, metaclass=TensorSatMeta):
    """An immutable wiring."""

    __store: ClassVar[InstanceStore] = InstanceStore()

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape, ...],
        wire_types: Shape,
        slot_wires_list: tuple[Wires, ...],
        out_wires: Wires,
    ) -> Self:
        """Protected constructor."""
        instance_key = (
            slot_shapes,
            wire_types,
            slot_wires_list,
            out_wires,
        )
        with Wiring.__store.instance(cls, instance_key) as self:
            if self is None:
                self = super().__new__(cls)
                self.__slot_shapes = slot_shapes
                self.__wire_types = wire_types
                self.__slot_wires_list = slot_wires_list
                self.__out_wires = out_wires
                Wiring.__store.register(self)
            return self

    __slot_shapes: tuple[Shape, ...]
    __wire_types: Shape
    __slot_wires_list: tuple[Wires, ...]
    __out_wires: Wires

    def __new__(cls, **data: Unpack[WiringData]) -> Self:
        """
        Constructs a wiring from the given data.

        :meta public:
        """
        assert validate(data, WiringData)
        # Destructure the data:
        wire_types = tuple(data["wire_types"])
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
            tuple(wire_types[i] for i in slot_wires) for slot_wires in slot_wires_list
        )
        return cls._new(slot_shapes, wire_types, slot_wires_list, out_wires)

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        """Shapes for the slots of the wiring."""
        return self.__slot_shapes

    @property
    def wire_types(self) -> Shape:
        """Wire types."""
        return self.__wire_types

    @property
    def slot_wires_list(self) -> tuple[Wires, ...]:
        """Assignment of (the index of) a wire to each port of each slot."""
        return self.__slot_wires_list

    @property
    def out_wires(self) -> Wires:
        """Assignment of (the index of) a wire to each outer port."""
        return self.__out_wires

    @cached_property
    def shape(self) -> Shape:  # type: ignore[override]
        """Shape of the wiring, i.e. types of its outer ports."""
        return tuple(self.wire_types[o] for o in self.out_wires)

    @property
    def num_slots(self) -> int:
        """Number of slots."""
        return len(self.slot_shapes)

    @property
    def slots(self) -> Sequence[Slot]:
        """Sequence of (the indices of) slots."""
        return range(self.num_slots)

    def num_slot_ports(self, slot: Slot) -> int:
        """Number of ports for the given slot."""
        return len(self.slot_shapes[slot])

    def slot_ports(self, slot: Slot) -> Sequence[Port]:
        """Sequence of (the indices of) ports for the given slot."""
        return range(self.num_slot_ports(slot))

    def validate_slot_data(self, data: Mapping[Slot, Shaped], /) -> None:
        """Validates the shapes for given slot data."""
        assert validate(data, Mapping[Slot, Shaped])
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

    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @property
    def wires(self) -> Sequence[Wire]:
        """Sequence of (the indices of) wires."""
        return range(self.num_wires)

    def slot_wires(self, slot: Slot) -> Wires:
        """Sequence of (the indices of) wires for the given slot."""
        assert validate(slot, Slot)
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        return self.slot_wires_list[slot]

    @property
    def dangling_wires(self) -> frozenset[Wire]:
        """The set of "dangling" wires, wires not connected to any slot ports."""
        return frozenset(self.wires) - frozenset(
            w for ws in self.slot_wires_list for w in ws
        )

    @property
    def discarded_wires(self) -> frozenset[Wire]:
        """The set of "discarded" wires, wires not connected to any out ports."""
        return frozenset(self.wires) - set(self.out_wires)

    @property
    def scalar_wires(self) -> frozenset[Wire]:
        """The set of "scalar" wires, wires not connected to any ports."""
        return frozenset(self.wires) - set(self.out_wires) - set(self.dangling_wires)

    @property
    def wired_slot_ports(self) -> Mapping[Wire, tuple[tuple[Slot, Port], ...]]:
        """
        Computes and returns a mapping of wires to the collection of ``(slot, port)``
        pairs connected by that wire.
        Wires not appearing as keys in the mapping are "dangling", i.e. they don't
        connect to any slot ports.
        """
        wired_slot_ports: dict[Wire, list[tuple[Slot, Port]]] = {}
        for slot, wires in enumerate(self.slot_wires_list):
            for port, wire in enumerate(wires):
                wired_slot_ports.setdefault(wire, []).append((slot, port))
        return MappingProxyType(
            {w: tuple(w_slot_ports) for w, w_slot_ports in wired_slot_ports.items()}
        )

    @property
    def wired_slots(self) -> Mapping[Wire, Slots]:
        """
        Computes and returns a mapping of wires to the collection of slots
        connected by that wire.
        Wires not appearing as keys in the mapping are "dangling", i.e. they don't
        connect to any slots.
        """
        wired_slots: dict[Wire, list[Slot]] = {}
        for slot, wires in enumerate(self.slot_wires_list):
            for wire in wires:
                wired_slots.setdefault(wire, []).append(slot)
        return MappingProxyType(
            {w: tuple(w_slots) for w, w_slots in wired_slots.items()}
        )

    @property
    def wired_out_ports(self) -> Mapping[Wire, Ports]:
        """
        Computes and returns a mapping of wires to the collection of out ports
        connected by that wire.
        Wires not appearing as keys in the mapping are "discarded", i.e. they don't
        connect to any out ports.
        """
        wired_out_ports: dict[Wire, list[Port]] = {}
        for port, wire in enumerate(self.out_wires):
            wired_out_ports.setdefault(wire, []).append(port)
        return MappingProxyType(
            {w: tuple(w_out_ports) for w, w_out_ports in wired_out_ports.items()}
        )

    @property
    def wired_ports(self) -> Mapping[Wire, tuple[Port | tuple[Slot, Port], ...]]:
        """
        Computes and returns a mapping of wires to the collection of ports connected by
        that wire: ``(slot, port)`` pairs for slot ports and individual out ports.
        Wires not appearing as keys in the mapping are "scalar", i.e. they don't
        connect to any ports.
        """
        wired_ports: dict[Wire, list[Port | tuple[Slot, Port]]] = {}
        for slot, wires in enumerate(self.slot_wires_list):
            for port, wire in enumerate(wires):
                wired_ports.setdefault(wire, []).append((slot, port))
        for port, wire in enumerate(self.out_wires):
            wired_ports.setdefault(wire, []).append(port)
        return MappingProxyType(
            {w: tuple(w_out_ports) for w, w_out_ports in wired_ports.items()}
        )

    def compose(self, wirings: Mapping[Slot, Wiring]) -> Wiring:
        """Composes this wiring with the given wirings for (some of) its slots."""
        assert validate(wirings, Mapping[Slot, Wiring])
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

    def _compose(self, wirings: Mapping[Slot, Wiring]) -> Wiring:
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
        wire_types: list[Type] = []
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
        # 4. Compute new slot wires:
        new_slot_wires_list: list[Wires] = []
        for slot in self.slots:
            if slot in wirings:
                new_slot_wires_list.extend(
                    tuple(slot_wire_remap[(slot, w)] for w in _slot_wires)
                    for _slot_wires in wirings[slot].slot_wires_list
                )
            else:
                new_slot_wires_list.append(
                    tuple(wire_remap[w] for w in self.slot_wires_list[slot])
                )
        # 5. Compute new outer wires and return new wiring
        out_wires = tuple(wire_remap[w] for w in self.out_wires)
        return Wiring(
            wire_types=wire_types,
            slot_wires_list=new_slot_wires_list,
            out_wires=out_wires,
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
class WiringBuilder(Shaped):
    """Utility class to build wirings."""

    __slot_shapes: list[list[Type]]
    __shape: list[Type]
    __wire_types: list[Type]
    __slot_wires_list: list[list[Wire]]
    __out_wires: list[Wire]

    def __new__(cls) -> Self:
        """
        Constructs a blank wiring builder.

        :meta public:
        """
        self = super().__new__(cls)
        self.__slot_shapes = []
        self.__shape = []
        self.__wire_types = []
        self.__slot_wires_list = []
        self.__out_wires = []
        return self

    @property
    def num_slots(self) -> int:
        """Number of slots."""
        return len(self.slot_shapes)

    @property
    def slots(self) -> Sequence[Slot]:
        """Sequence of (the indices of) slots."""
        return range(self.num_slots)

    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @property
    def wires(self) -> Sequence[Wire]:
        """Sequence of (the indices of) wires."""
        return range(self.num_wires)

    @cached_property
    def slot_shapes(self) -> tuple[Shape, ...]:
        return tuple(tuple(s) for s in self.__slot_shapes)

    @cached_property
    def shape(self) -> Shape:  # type: ignore[override]
        return tuple(self.__shape)

    @cached_property
    def wire_types(self) -> Shape:
        """Wire types."""
        return tuple(self.__wire_types)

    @cached_property
    def slot_wires_list(self) -> tuple[Wires, ...]:
        """Assignment of (the index of) a wire to each port of each slot."""
        return tuple(map(tuple, self.__slot_wires_list))

    @cached_property
    def out_wires(self) -> Wires:
        """Assignment of (the index of) a wire to each outer port."""
        return tuple(self.__out_wires)

    @property
    def wiring(self) -> Wiring:
        """The wiring built thus far."""
        return Wiring._new(
            self.slot_shapes,
            self.wire_types,
            self.slot_wires_list,
            self.out_wires,
        )

    def slot_wires(self, slot: Slot) -> Wires:
        """Sequence of (the indices of) wires for the given slot."""
        assert validate(slot, Slot)
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        return tuple(self.__slot_wires_list[slot])

    def copy(self) -> WiringBuilder:
        """Returns a deep copy of this wiring builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__shape = self.__shape.copy()
        clone.__wire_types = self.__wire_types.copy()
        clone.__slot_wires_list = [m.copy() for m in self.__slot_wires_list]
        clone.__out_wires = self.__out_wires.copy()
        return clone

    def add_wire(self, t: Type) -> Wire:
        """Adds a new wire with the given type."""
        assert validate(t, Type)
        return self._add_wires([t])[0]

    def add_wires(self, ts: Sequence[Type]) -> Wires:
        """Adds new wires with the given types."""
        assert validate(ts, Sequence[Type])
        return self._add_wires(ts)

    def _add_wires(self, ts: Sequence[Type]) -> Wires:
        del self.wire_types
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

    def add_out_ports(self, wires: Sequence[Wire]) -> Ports:
        """Adds new outer ports, connected the given wires."""
        assert validate(wires, Sequence[Wire])
        self._validate_wires(wires)
        return self._add_out_ports(wires)

    def _add_out_ports(self, wires: Sequence[Wire]) -> Ports:
        del self.shape
        del self.out_wires
        shape, wire_types = self.__shape, self.__wire_types
        len_before = len(shape)
        shape.extend(wire_types[wire] for wire in wires)
        self.__out_wires.extend(wires)
        return tuple(range(len_before, len(shape)))

    def add_slot(self) -> Slot:
        """Adds a new slot."""
        del self.slot_shapes
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

    def add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> Ports:
        """Adds new ports for the given slot, connected the given wires."""
        if slot not in range(self.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        self._validate_wires(wires)
        return self._add_slot_ports(slot, wires)

    def _add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> Ports:
        del self.slot_shapes
        del self.slot_wires_list
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


class BoxMeta(InheritanceForestMeta, TensorSatMeta):
    """
    Metaclass for box classes, forcing box classes to form an inheritance tree
    with :class:`Box` as the root.
    This guarantees that the class join of box classes, returned by the static method
    :func:`Box.class_join`, is always well-defined.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        cls = super().__new__(mcs, name, bases, namespace)
        if not getattr(cls, "__abstractmethods__", None):
            try:
                import autoray  # type: ignore[import-untyped]

                autoray.register_backend(cls, "tensorsat._autoray")
            except ModuleNotFoundError:
                pass
        return cls


class Box(Shaped, metaclass=BoxMeta):
    """
    Abstract base class for boxes in diagrams.
    """

    @staticmethod
    def class_join(bases: Iterable[BoxClass]) -> BoxClass:
        """Returns the join of the given box classes."""
        join = Box._subclass_join(bases)
        assert join is not None
        return join

    @classmethod
    def can_be_contracted(cls) -> bool:
        """
        Whether this box class can be contracted,
        i.e. whether it provides implementations for the (protected versions of)
        :func:`Box.contract2`, :meth:`Box.rewire` and :func:`Box.spider`.
        """
        abstract_methods = cls.__abstractmethods__
        return (
            "_contract2" not in abstract_methods
            and "_rewire" not in abstract_methods
            and "_spider" not in abstract_methods
        )

    @final
    @classmethod
    def prod(cls, boxes: Iterable[Self]) -> Self:
        """
        Takes the product of two or more boxes of this class.
        The resulting relation has as its ports the ports of the boxes given,
        in the order they were given.
        """
        boxes = tuple(boxes)
        if __debug__:
            for box in boxes:
                if not isinstance(box, cls):
                    raise TypeError(
                        f"Boxes must be instances of box class {cls},"
                        f" found instance of box class {type(box)} instead."
                    )
        if not boxes:
            raise ValueError("Products must involve at least one box.")
        if len(boxes) == 1:
            return boxes[0]
        res = boxes[0]
        for box in boxes[1:]:
            lhs, rhs = res, box
            lhs_len, rhs_len = len(lhs.shape), len(rhs.shape)
            lhs_wires, rhs_wires = range(lhs_len), range(lhs_len, lhs_len + rhs_len)
            out_wires = range(lhs_len + rhs_len)
            res = cls._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)
        return res

    @final
    @classmethod
    def from_wiring(cls, wiring: Wiring) -> Self:
        """
        Creates a box of this class for the given wiring.
        The ports of the box are the slot ports for the wiring, in slot order,
        followed by the out ports for the wiring.
        """
        wires = wiring.wires
        wire_types = wiring.wire_types
        wired_ports = wiring.wired_ports
        wires_by_spider_args: dict[tuple[Type, int], list[Wire]] = {}
        for w, w_ports in wired_ports.items():
            w_spider_args = (wire_types[w], len(w_ports))
            wires_by_spider_args.setdefault(w_spider_args, []).append(w)
        spiders = {(t, n): cls.spider(t, n) for t, n in wires_by_spider_args}
        wire_spiders = {
            w: spiders[spider_args]
            for spider_args, ws in wires_by_spider_args.items()
            for w in ws
        }
        res = cls.prod(wire_spiders[w] for w in wires)
        res_port_labels = [port for w in wires for port in wired_ports[w]]
        out_port_labels = [
            (slot, port) for slot in wiring.slots for port in wiring.slot_ports(slot)
        ] + list(wiring.ports)
        out_ports = [res_port_labels.index(p) for p in out_port_labels]
        return res._rewire(out_ports)

    @final
    @classmethod
    def spider(cls, t: Type, num_ports: int) -> Self:
        """
        The box corresponding to a single wire connected to the given number of ports,
        all ports being of this type.
        """
        validate(num_ports, int)
        if num_ports <= 0:
            raise ValueError("Number of ports must be strictly positive.")
        return cls._spider(t, num_ports)

    @classmethod
    @abstractmethod
    def _spider(cls, t: Type, num_ports: int) -> Self:
        """
        Protected version of :func:`Box.spider`, to be implemented by subclasses.
        It is guaranteed that ``num_ports`` is strictly positive.
        """

    @final
    @classmethod
    def scalar(cls, scalar: bool, /) -> Self:
        """The box corresponding to the given Boolean scalar."""
        box = cls._scalar(scalar)
        assert box.shape == ()
        assert box._as_scalar() == scalar
        return box

    @classmethod
    @abstractmethod
    def _scalar(cls, scalar: bool, /) -> Self:
        """Protected version of :func:`Box.scalar`, to be implemented by subclasses."""

    @final
    @classmethod
    def contract2(
        cls,
        lhs: Self,
        lhs_wires: Sequence[Wire],
        rhs: Self,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire] | None = None,
    ) -> Self:
        assert validate(lhs, cls)
        assert validate(lhs_wires, Sequence[Wire])
        assert validate(rhs, cls)
        assert validate(rhs_wires, Sequence[Wire])
        assert validate(out_wires, Sequence[Wire] | None)
        if len(lhs_wires) != len(lhs.shape):
            raise ValueError(
                f"Number of wires in lhs ({len(lhs_wires)}) does not match"
                f" the number of ports in lhs shape ({len(lhs.shape)})."
            )
        if len(rhs_wires) != len(rhs.shape):
            raise ValueError(
                f"Number of wires in rhs ({len(rhs_wires)}) does not match"
                f" the number of ports in rhs shape ({len(rhs.shape)})."
            )
        if out_wires is None:
            out_wires = sorted(set(lhs_wires).symmetric_difference(rhs_wires))
        else:
            out_wires_set = set(out_wires)
            out_wires_set.difference_update(lhs_wires)
            out_wires_set.difference_update(rhs_wires)
            if out_wires_set:
                raise ValueError("Every output wire must appear in LHR or RHS.")
        return cls._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)

    @classmethod
    @abstractmethod
    def _contract2(
        cls,
        lhs: Self,
        lhs_wires: Sequence[Wire],
        rhs: Self,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire],
    ) -> Self:
        """
        Protected version of :func:`Box.contract2`, to be implemented by subclasses.
        It is guaranteed that:

        - The length of ``lhs_wires`` matches the length of ``lhs.shape``
        - The length of ``rhs_wires`` matches the length of ``rhs.shape``
        - Every index in ``out_wires`` appears in ``lhs_wires`` or ``rhs_wires``

        It is possible for wires from ``lhs_wires`` or ``rhs_wires`` to appear multiple
        times in ``out_wires``, or not at all.
        """

    def __new__(cls, name: str | None = None) -> Self:
        """
        Constructs a new box.

        :meta public:
        """
        if not cls.__final__:
            raise TypeError("Only final subclasses of Box can be instantiated.")
        self = super().__new__(cls)
        self.__name = name
        return self

    __name: str | None

    @property
    def name(self) -> str | None:
        """An optional name for the box."""
        return self.__name

    @final
    def rewire(self, out_ports: Sequence[Port]) -> Self:
        """Permutes, duplicates and/or drops output ports from this box."""
        assert validate(out_ports, Sequence[Port])
        ports = self.ports
        if not all(port in ports for port in out_ports):
            raise ValueError("Output ports must be valid for this box.")
        return self._rewire(out_ports)

    @abstractmethod
    def _rewire(self, out_ports: Sequence[Port]) -> Self:
        """
        Protected version of :meth:`Box.rewire`, to be implemented by subclasses.

        It is guaranteed that the output ports are valid for this box.
        It is possible for ports to appear multiple times in ``out_ports``,
        or not at all.
        """

    @abstractmethod
    def _as_scalar(self) -> bool:
        """
        Conversion of box to a Boolean scalar, to be implemented by subclasses.

        It is guaranteed that this method is invoked only when the box's shape is empty.
        """

    @final
    def __bool__(self) -> bool:
        """
        Converts the box to a Boolean scalar.

        :raises ValueError: if the box's shape is not empty.
        """
        if self.shape:
            raise ValueError("Box is not a scalar.")
        return self._as_scalar()

    def __getitem__(self, ports: Port | Sequence[Port]) -> SelectedBlockPorts:
        """Selects the given ports from this box."""
        return SelectedBlockPorts(self, ports)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        num_ports = len(self.shape)
        box_name = self.name
        if box_name is None:
            return f"<{cls_name} {id(self):#x}: {num_ports} ports>"
        return f"<{cls_name} {id(self):#x}: {num_ports} ports, named {box_name!r}>"


BoxClass: TypeAlias = SubclassOf[Box]
"""A box class, i.e. a subclass of :class:`Box`."""


class TensorLikeBox(Box):
    """
    Abstract base classes for tensor-like boxes, where an integral shape can be
    used to estimate contraction costs and memory requirements.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[TensorLikeType, ...]: ...

    @cached_property
    def tensor_shape(self) -> tuple[int, ...]:
        """
        The tensor-like shape for this box.
        Guaranteed to have the
        """
        shape = self.shape
        assert all(isinstance(t, TensorLikeType) for t in shape)
        return tuple(t.tensor_dim for t in shape)


BoxT_inv = TypeVar("BoxT_inv", bound=Box, default=Box)
"""Invariant type variable for boxes."""

TensorLikeBoxT_inv = TypeVar("TensorLikeBoxT_inv", bound=TensorLikeBox)
"""Invariant type variable for tensor-like boxes."""


Block: TypeAlias = "Box | Diagram"
"""
Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

"""

RecipeParams = ParamSpec("RecipeParams")
"""Parameter specification variable for the parameters of a recipe."""


@final
class DiagramRecipe(Generic[RecipeParams], metaclass=TensorSatMeta):
    """A Recipe to produce diagrams from given perameters."""

    __WRAPPER_ASSIGNMENTS: ClassVar[tuple[str, ...]] = (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
    )

    __module__: str
    __name__: str
    __qualname__: str
    __doc__: str
    __annotations__: dict[str, Any]

    __diagrams: WeakValueDictionary[Any, Diagram]
    __wrapped__: Callable[Concatenate[DiagramBuilder, RecipeParams], None]

    def __new__(
        cls,
        recipe: Callable[Concatenate[DiagramBuilder, RecipeParams], None],
    ) -> Self:
        self = super().__new__(cls)
        # Adapted from functools.update_wrapper:
        for attr in self.__WRAPPER_ASSIGNMENTS:
            try:
                value = getattr(recipe, attr)
                setattr(self, attr, value)
            except AttributeError:
                pass
        self.__wrapped__ = recipe
        self.__diagrams = WeakValueDictionary()
        return self

    @property
    def name(self) -> str:
        """The name of this recipe."""
        return self.__wrapped__.__name__

    def __call__(
        self, *args: RecipeParams.args, **kwargs: RecipeParams.kwargs
    ) -> Diagram:
        """
        Returns the diagram constructed by the recipe on given arguments.

        :meta public:
        """
        key = (args, frozenset(kwargs.items()))
        if key in self.__diagrams:
            return self.__diagrams[key]
        builder: DiagramBuilder = DiagramBuilder()
        self.__wrapped__(builder, *args, **kwargs)
        diagram = builder.diagram()
        diagram._Diagram__recipe_used = self  # type: ignore[attr-defined]
        self.__diagrams[key] = diagram
        return diagram

    def __repr__(self) -> str:
        """Representation of the recipe."""
        recipe = self.__wrapped__
        mod = recipe.__module__
        name = recipe.__name__
        return f"Diagram.recipe({mod}.{name})"


@final
class PortOrderStructure(metaclass=TensorSatMeta):
    """A port order structure for a diagram."""

    @classmethod
    def _new(
        cls,
        wiring: Wiring,
        input_ports: frozenset[Port],
        slot_input_ports_list: tuple[frozenset[Port], ...],
        *,
        __consolidate: bool = True,
    ) -> Self:
        """Protected constructor."""
        if __consolidate:
            representative = {fset: fset for fset in slot_input_ports_list}
            slot_input_ports_list = tuple(
                representative[fset] for fset in slot_input_ports_list
            )
        self = super().__new__(cls)
        self.__wiring = wiring
        self.__input_ports = input_ports
        self.__slot_input_ports_list = slot_input_ports_list
        return self

    __wiring: Wiring
    __input_ports: frozenset[Port]
    __slot_input_ports_list: tuple[frozenset[Port], ...]

    def __new__(
        cls,
        wiring: Wiring,
        input_ports: Iterable[Port],
        slot_input_ports_list: Iterable[Iterable[Port]],
    ) -> Self:
        """
        Constructs a new diagram port order structure.

        :meta public:
        """
        assert validate(wiring, Wiring)
        input_ports = frozenset(input_ports)
        _slot_input_ports_list = tuple(map(frozenset, slot_input_ports_list))
        assert validate(input_ports, frozenset[Port])
        assert validate(_slot_input_ports_list, tuple[frozenset[Port], ...])
        for port in input_ports:
            if port not in wiring.ports:
                raise ValueError(f"Invalid input port {port}.")
        if len(_slot_input_ports_list) != wiring.num_slots:
            raise ValueError(
                f"Expected {wiring.num_slots} slot input port sets,"
                f" got {len(_slot_input_ports_list)}."
            )
        for slot, ports in enumerate(_slot_input_ports_list):
            for port in ports:
                if port not in wiring.slot_ports(slot):
                    raise ValueError(f"Invalid input port {port} for slot {slot}.")
        return cls._new(wiring, input_ports, _slot_input_ports_list)

    @property
    def wiring(self) -> Wiring:
        """The wiring for this port order structure."""
        return self.__wiring

    @property
    def input_ports(self) -> frozenset[Port]:
        """The input ports for the wiring."""
        return self.__input_ports

    @property
    def slot_input_ports_list(self) -> tuple[frozenset[Port], ...]:
        """The input ports for each slot in the wiring."""
        return self.__slot_input_ports_list

    @property
    def output_ports(self) -> frozenset[Port]:
        """The output ports for the wiring."""
        input_ports = self.input_ports
        return frozenset(port for port in self.wiring.ports if port not in input_ports)

    @property
    def slot_output_ports_list(self) -> tuple[frozenset[Port], ...]:
        """The output ports for each slot in the wiring."""
        wiring, slot_input_ports_list = self.wiring, self.slot_input_ports_list
        return tuple(
            frozenset(p for p in wiring.slot_ports(slot) if p not in slot_inputs)
            for slot, slot_inputs in enumerate(slot_input_ports_list)
        )

    def __repr__(self) -> str:
        return f"<DiagramPortOrderStructure {id(self):#x}>"


@final
class Diagram(Shaped, metaclass=TensorSatMeta):
    """
    A diagram, consisting of a :class:`Wiring` together with :obj:`Block` associated
    to (a subset of) the wiring's slots.
    """

    @staticmethod
    def from_recipe(
        recipe: Callable[[DiagramBuilder], None],
    ) -> Diagram:
        """
        A function decorator to create a diagram from a diagram-building recipe.

        For example, the snippet below creates the :class:`Diagram` instance
        ``full_adder`` for a full-adder circuit:

        .. code-block:: python

            from tensorsat.lang.fin_rel import FinSet
            from tensorsat.lib.bincirc import bit, and_, or_, xor_

            @Diagram.from_recipe
            def full_adder(diag: DiagramBuilder[FinSet]) -> None:
                a, b, c_in = diag.add_inputs()
                x1, = xor_ @ diag[a, b]
                x2, = and_ @ diag[a, b]
                x3, = and_ @ diag[x1, c_in]
                s, = xor_ @ diag[x1, x3]
                c_out, = or_ @ diag[x2, x3]
                diag.add_outputs(s, c_out)

        """
        builder: DiagramBuilder = DiagramBuilder()
        recipe(builder)
        diagram = builder.diagram()
        diagram._Diagram__recipe_used = recipe  # type: ignore[attr-defined]
        return diagram

    @staticmethod
    def recipe(
        recipe: Callable[Concatenate[DiagramBuilder, RecipeParams], None],
    ) -> Callable[RecipeParams, Diagram]:
        """
        A function decorator to create a parametric diagram factory from a
        diagram-building recipe taking additional parameters.

        For example, the snippet below creates a function returning a ripple-carry adder
        diagram given the number ``n`` of bits for each of its two arguments.

        .. code-block:: python

            from tensorsat.lang.fin_rel import FinSet
            from tensorsat.lib.bincirc import bit, and_, or_, xor_

            @Diagram.recipe
            def rc_adder(diag: DiagramBuilder[FinSet], num_bits: int) -> None:
                inputs = diag.add_inputs(bit**(2*num_bits+1))
                outputs: list[Wire] = []
                c = inputs[0]
                for i in range(num_bits):
                    a, b = inputs[2 * i + 1 : 2 * i + 3]
                    s, c = full_adder @ diag[c, a, b]
                    outputs.append(s)
                outputs.append(c)
                diag.add_outputs(outputs)

        Note that the results of calls to recipes are automatically cached,
        and that the parameters are expected to be hashable.
        """
        return DiagramRecipe(recipe)

    @classmethod
    def _new(cls, wiring: Wiring, blocks: tuple[Block | None, ...]) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__wiring = wiring
        self.__blocks = blocks
        self.__recipe_used = None
        self.__port_order_struct = None
        return self

    __wiring: Wiring
    __blocks: tuple[Box | Diagram | None, ...]
    __recipe_used: Callable[Concatenate[DiagramBuilder, ...], Diagram] | None
    __port_order_struct: PortOrderStructure | None

    # Attributes only set in certain conditions:
    __hash_cache: int
    __seq_blocks: tuple[Block, ...]

    def __new__(cls, wiring: Wiring, blocks: Mapping[Slot, Block]) -> Self:
        """
        Constructs a new diagram from a wiring and blocks for (some of) its slots.

        :meta public:
        """
        assert validate(wiring, Wiring)
        wiring.validate_slot_data(blocks)
        _blocks = tuple(map(blocks.get, wiring.slots))
        return cls._new(wiring, _blocks)

    @property
    def wiring(self) -> Wiring:
        """Wiring for the diagram."""
        return self.__wiring

    @property
    def blocks(self) -> tuple[Box | Diagram | None, ...]:
        """
        Sequence of blocks associated to the slots in the diagram's wiring,
        or :obj:`None` to indicate that a slot is open.
        """
        return self.__blocks

    @property
    def shape(self) -> Shape:
        """Shape of the diagram."""
        return self.wiring.shape

    @property
    def open_slots(self) -> Slots:
        """Slots of the diagram wiring which are open in the diagram."""
        return tuple(slot for slot, block in enumerate(self.blocks) if block is None)

    @property
    def num_open_slots(self) -> int:
        """Number of open slots in the diagram."""
        return self.blocks.count(None)

    @property
    def subdiagram_slots(self) -> Slots:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Diagram)
        )

    @property
    def subdiagrams(self) -> tuple[Diagram, ...]:
        """Diagrams associated to the slots in :attr:`subdiagram_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Diagram))

    @property
    def box_slots(self) -> Slots:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Box)
        )

    @property
    def boxes(self) -> tuple[Box, ...]:
        """Boxes associated to the slots in :attr:`box_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Box))

    @property
    def is_flat(self) -> bool:
        """Whether the diagram is flat, i.e., it has no sub-diagrams."""
        return not any(isinstance(block, Diagram) for block in self.blocks)

    @property
    def depth(self) -> int:
        """Nesting depth of the diagram."""
        subdiagrams = self.subdiagrams
        if not subdiagrams:
            return 0
        return 1 + max(diag.depth for diag in subdiagrams)

    @property
    def recipe_used(self) -> Callable[Concatenate[DiagramBuilder, ...], Diagram] | None:
        """The recipe used to construct this diagram, if any."""
        return self.__recipe_used

    @property
    def port_order_struct(self) -> PortOrderStructure | None:
        """The port order structure associated to this diagram, if any."""
        return self.__port_order_struct

    def with_port_order_struct(self, struct: PortOrderStructure | None, /) -> Diagram:
        """This diagram, with a different port order structure."""
        if struct is not None and self.wiring != struct.wiring:
            raise ValueError(
                "Diagram wiring must coincide with that for port order structure."
            )
        clone = type(self)._new(self.__wiring, self.__blocks)
        clone.__port_order_struct = struct
        return clone

    @cached_property
    def box_class(self) -> BoxClass:
        """
        The most specific common box class for the boxes in this diagram and its
        subdiagrams. See :meth:`Box.box_class_join`.
        """
        return Box.class_join(self.box_classes)

    @cached_property
    def box_classes(self) -> frozenset[BoxClass]:
        """The set of box classes appearing in this diagram and its subdiagrams."""
        box_classes: set[BoxClass] = {type(box) for box in self.boxes}
        for diag in self.subdiagrams:
            box_classes.update(diag.box_classes)
        return frozenset(box_classes)

    def compose(self, new_blocks: Mapping[Slot, Block | Wiring]) -> Diagram:
        """
        Composes this wiring with the given boxes, diagrams and/or wirings
        for (some of) its slots.
        """
        assert validate(new_blocks, Mapping[Slot, Block | Wiring])
        curr_wiring = self.wiring
        curr_wiring.validate_slot_data(new_blocks)
        curr_blocks = self.blocks
        for slot in new_blocks.keys():
            if curr_blocks[slot] is not None:
                raise ValueError(f"Slot {slot} is not open.")
        merged_wiring = curr_wiring.compose(
            {
                slot: block
                for slot, block in new_blocks.items()
                if isinstance(block, Wiring)
            }
        )
        merged_blocks: list[Block | None] = []
        for slot, curr_block in enumerate(curr_blocks):
            if curr_block is not None:
                merged_blocks.append(curr_block)
            elif (new_block := new_blocks[slot]) is not None:
                if isinstance(new_block, (Box, Diagram)):
                    merged_blocks.append(new_block)
                else:
                    merged_blocks.extend([None] * new_block.num_slots)
            else:
                merged_blocks.append(None)
        diagram = Diagram._new(merged_wiring, tuple(merged_blocks))
        return diagram

    def flatten(self, *, cache: bool = True) -> Diagram:
        """
        Returns a flat diagram, obtained by recursively flattening all
        sub-diagrams, composing their wirings into the current wiring, and taking
        all blocks (of this diagrams and its sub-diagrams) as the blocks of the result.
        """
        assert validate(cache, bool)
        return self._flatten({} if cache else None)

    def _flatten(self, cache: dict[Diagram, Diagram] | None) -> Diagram:
        if cache is not None and self in cache:
            return cache[self]
        flat_subdiagrams = [
            subdiagram._flatten(cache) for subdiagram in self.subdiagrams
        ]
        subwirings = [subdiag.wiring for subdiag in flat_subdiagrams]
        flat_wiring = self.wiring.compose(dict(zip(self.subdiagram_slots, subwirings)))
        flat_blocks: list[Box | None] = []
        subdiagram_slots = {slot: idx for idx, slot in enumerate(self.subdiagram_slots)}
        for slot, block in enumerate(self.blocks):
            if (idx := subdiagram_slots.get(slot)) is not None:
                flat_blocks.extend(
                    cast(tuple[Box | None, ...], flat_subdiagrams[idx].blocks)
                )
            else:
                flat_blocks.append(cast(Box | None, block))
        flat_diagram = Diagram._new(flat_wiring, tuple(flat_blocks))
        if cache is not None:
            cache[self] = flat_diagram
        return flat_diagram

    def __getitem__(self, ports: Port | Sequence[Port]) -> SelectedBlockPorts:
        """Selects the given ports from this diagram."""
        return SelectedBlockPorts(self, ports)

    def __rshift__(self, other: Block | SelectedBlockPorts) -> Diagram:
        """
        Returns the sequential composition of this diagram with another diagram/box.

        :meta public:
        """
        return self[self.ports] >> other

    def __rrshift__(self, other: Box) -> Diagram:
        """
        Returns the sequential composition of another box with this diagram.

        :meta public:
        """
        return other[other.ports] >> self

    def __repr__(self) -> str:
        attrs: list[str] = []
        num_wires = self.wiring.num_wires
        num_open_slots = self.num_open_slots
        num_blocks = len(self.blocks)
        depth = self.depth
        num_ports = len(self.wiring.out_wires)
        recipe = self.recipe_used
        if num_wires > 0:
            attrs.append(f"{num_wires} wires")
        if num_open_slots > 0:
            attrs.append(f"{num_open_slots} open slots")
        if num_blocks > 0:
            attrs.append(f"{num_blocks} blocks")
        if depth > 0:
            attrs.append(f"depth {depth}")
        if num_ports > 0:
            attrs.append(f"{num_ports} ports")
        if recipe is not None:
            attrs.append(f"from recipe {recipe.__name__!r}")
        return f"<Diagram {id(self):#x}: {", ".join(attrs)}>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Diagram):
            return NotImplemented
        if self is other:
            return True
        return self.wiring == other.wiring and self.blocks == other.blocks

    def __hash__(self) -> int:
        try:
            return self.__hash_cache
        except AttributeError:
            self.__hash_cache = h = hash((Diagram, self.wiring, self.blocks))
            return h


@final
class SelectedBlockPorts(metaclass=TensorSatMeta):
    """
    Utility class wrapping a selection of ports in a given box or diagram,
    to be used for the purposes of composition.

    Supports usage of the ``@`` operator with selected wires from a diagram builder
    on the rhs, enabling special syntax for addition of blocks to diagrams.

    Supports usage of the ``>>`` operator with a block or block port selection on the
    rhs, enabling special syntax for sequential composition of diagrams.

    See :meth:`DiagramBuilder.__getitem__`.
    """

    @classmethod
    def _new(cls, block: Block, ports: Ports) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__block = block
        self.__ports = ports
        return self

    __block: Block
    __ports: Ports

    def __new__(cls, block: Block, ports: Port | Sequence[Port]) -> Self:
        assert validate(block, Box | Diagram)
        if isinstance(ports, Port):
            ports = (ports,)
        else:
            assert validate(ports, Sequence[Port])
            ports = tuple(ports)
        block_ports = block.ports
        for port in ports:
            if port not in block_ports:
                raise ValueError(f"Invalid port {port} selected for {block}")
        return cls._new(block, ports)

    @property
    def block(self) -> Block:
        """The block to which the selected ports belong."""
        return self.__block

    @property
    def ports(self) -> Ports:
        """The selected ports."""
        return self.__ports

    def __rshift__(self, other: Block | SelectedBlockPorts) -> Diagram:
        """
        Sequentially composes the block (from which ports were selected)
        with another block.

        :meta public:
        """
        self_ports = self.ports
        if isinstance(other, (Box, Diagram)):
            if other.num_ports < len(self_ports):
                raise ValueError(
                    f"Cannot sequentially compose: {len(self.ports)} selected on LHS,"
                    f" {other.num_ports} ports available on RHS."
                )
            other = SelectedBlockPorts(other, range(len(self_ports)))
        elif isinstance(other, SelectedBlockPorts):
            if len(other.ports) != len(self_ports):
                raise ValueError(
                    f"Cannot sequentially compose: {len(self.ports)} selected on LHS,"
                    f" {len(other.ports)} ports selected on RHS."
                )
        else:
            return NotImplemented
        blocks: list[Block] = []
        if isinstance(self.block, Diagram) and hasattr(
            self.block, "_Diagram__seq_blocks"
        ):
            blocks.extend(self.block._Diagram__seq_blocks)
        else:
            blocks.append(self.block)
        if isinstance(other.block, Diagram) and hasattr(
            other.block, "_Diagram__seq_blocks"
        ):
            blocks.extend(other.block._Diagram__seq_blocks)
        else:
            blocks.append(other.block)
        fst_block_in_ports = tuple(p for p in blocks[0].ports if p not in self.ports)
        fst_block_shape = blocks[0].shape
        builder = DiagramBuilder()
        wires = builder.add_inputs(fst_block_shape[i] for i in fst_block_in_ports)
        for block in blocks:
            wires = block @ builder[wires]
        builder.add_outputs(wires)
        diagram = builder.diagram()
        diagram._Diagram__seq_blocks = tuple(blocks)  # type: ignore
        return diagram

    def __repr__(self) -> str:
        if isinstance(self.block, Box):
            return f"<Box {id(self.block):#x}>[{self.ports}]"
        return f"<Diagram {id(self.block):#x}>[{self.ports}]"


@final
class DiagramBuilder(metaclass=TensorSatMeta):
    """Utility class to build diagrams."""

    __wiring_builder: WiringBuilder
    __blocks: dict[Slot, Block]
    __input_ports: set[Port]
    __output_ports: set[Port]
    __slot_input_ports_list: list[set[Port]]

    def __new__(cls) -> DiagramBuilder:
        """
        Creates a blank diagram builder.

        :meta public:
        """
        self = super().__new__(cls)
        self.__wiring_builder = WiringBuilder()
        self.__blocks = {}
        self.__input_ports = set()
        self.__output_ports = set()
        self.__slot_input_ports_list = []
        return self

    def copy(self) -> DiagramBuilder:
        """Returns a deep copy of this diagram builder."""
        clone: DiagramBuilder = DiagramBuilder.__new__(DiagramBuilder)
        clone.__wiring_builder = self.__wiring_builder.copy()
        clone.__blocks = self.__blocks.copy()
        clone.__input_ports = self.__input_ports.copy()
        clone.__output_ports = self.__output_ports.copy()
        clone.__slot_input_ports_list = self.__slot_input_ports_list.copy()
        return clone

    @property
    def wiring(self) -> WiringBuilder:
        """The wiring builder for the diagram."""
        return self.__wiring_builder

    @property
    def blocks(self) -> Mapping[Slot, Block]:
        """Blocks in the diagram."""
        return MappingProxyType(self.__blocks)

    def diagram(self, *, port_order: bool = True) -> Diagram:
        """The diagram built thus far."""
        wiring = self.__wiring_builder.wiring
        blocks = self.__blocks
        _blocks = tuple(blocks.get(slot) for slot in wiring.slots)
        diagram = Diagram._new(wiring, _blocks)
        if port_order:
            diagram._Diagram__port_order_struct = PortOrderStructure._new(  # type: ignore
                diagram.wiring,
                frozenset(self.__input_ports),
                tuple(map(frozenset, self.__slot_input_ports_list)),
            )
        return diagram

    def set_block(self, slot: Slot, block: Block) -> None:
        """Sets a block for an existing open slot."""
        assert validate(block, Box | Diagram)
        blocks = self.__blocks
        if slot not in range(self.wiring.num_slots):
            raise ValueError(f"Invalid slot {slot}.")
        if slot in blocks:
            raise ValueError(f"Slot {slot} is already occupied.")
        if self.wiring.slot_shapes[slot] != block.shape:
            raise ValueError(
                f"Incompatible shape for block at slot {slot}:"
                f" expected {self.wiring.slot_shapes[slot]}, got {block.shape}."
            )
        self.__blocks[slot] = block

    def add_block(
        self, block: Block, inputs: Mapping[Port, Wire] = MappingProxyType({})
    ) -> Wires:
        """
        Adds a new slot to the diagram with the given block assigned to it.
        Specifically:

        1. Adds a new slot to the wiring.
        2. For each block port not having a wire associated to it by ``inputs``, creates
           a new wire (in port order).
        3. Adds a new port to the slot for each port in the block: those appearing in
           ``inputs`` are connected to the specified wire, while the others are connected
           to the newly created wires.
        4. Sets the block for the slot.
        5. Returns the newly created wires (in port order).

        By default, no inputs are passed, so a new wire is created for each port of the
        given block.
        """
        wire_types = self.__wiring_builder.wire_types
        assert validate(block, Box | Diagram)
        assert validate(inputs, Mapping[Port, Wire])
        block_shape = block.shape
        for port, wire in inputs.items():
            try:
                port_type = block_shape[port]
            except IndexError:
                raise ValueError(f"Invalid port {port} for block.")
            try:
                if port_type != wire_types[wire]:
                    raise ValueError(
                        f"Incompatible wire type for port {port}:"
                        f" port has type {block_shape[port]}, "
                        f"wire has type {wire_types[wire]}."
                    )
            except IndexError:
                raise ValueError(f"Invalid wire index {wire}.") from None
        return self._add_block(block, inputs)

    def _add_block(self, block: Block, inputs: Mapping[Port, Wire]) -> Wires:
        wiring_builder = self.__wiring_builder
        block_ports, block_shape = block.ports, block.shape
        output_ports = tuple(port for port in block_ports if port not in inputs)
        output_port_ts = tuple(block_shape[port] for port in output_ports)
        output_wires = wiring_builder.add_wires(output_port_ts)
        port_wire_mapping = {**inputs, **dict(zip(output_ports, output_wires))}
        slot = wiring_builder.add_slot()
        wiring_builder.add_slot_ports(
            slot, [port_wire_mapping[port] for port in block_ports]
        )
        self.__blocks[slot] = block
        self.__slot_input_ports_list.append(set(inputs.keys()))
        return output_wires

    def add_input(self, t: Type) -> Wire:
        """
        Creates a new wires of the given type,
        then adds a port connected to that wire.
        """
        assert validate(t, Type)
        return self._add_inputs((t,))[0]

    def add_inputs(self, ts: Iterable[Type]) -> Wires:
        """
        Creates new wires of the given types,
        then adds ports connected to those wires.
        """
        ts = tuple(ts)
        assert validate(ts, tuple[Type, ...])
        return self._add_inputs(ts)

    def _add_inputs(self, ts: tuple[Type, ...]) -> Wires:
        wiring = self.wiring
        wires = wiring._add_wires(ts)
        ports = wiring._add_out_ports(wires)
        self.__input_ports.update(ports)
        return wires

    def add_output(self, wire: Wire) -> Port:
        """Adds a port connected to the given wire."""
        assert validate(wire, Wire)
        if wire not in self.wiring.wires:
            raise ValueError(f"Invalid wire index {wire}.")
        return self._add_outputs((wire,))[0]

    def add_outputs(self, wires: Iterable[Wire]) -> Ports:
        """Adds ports connected to the given wires."""
        wires = tuple(wires)
        assert validate(wires, Wires)
        diag_wires = self.wiring.wires
        for wire in wires:
            if wire not in diag_wires:
                raise ValueError(f"Invalid wire index {wire}.")
        return self._add_outputs(wires)

    def _add_outputs(self, wires: Wires) -> Ports:
        ports = self.wiring._add_out_ports(wires)
        self.__output_ports.update(ports)
        return ports

    def __getitem__(self, wires: Wire | Sequence[Wire]) -> SelectedBuilderWires:
        """
        Enables special syntax for addition of blocks to the diagram:

        .. code-block:: python

            from tensorsat.lang.rel import bit
            from tensorsat.lib.bincirc import and_, or_, xor_
            circ = DiagramBuilder()
            a, b, c_in = circ.add_inputs(bit*3)
            x1, = xor_ @ circ[a, b]
            x2, = and_ @ circ[a, b]
            x3, = and_ @ circ[x1, c_in]
            s, = xor_ @ circ[x1, x3]
            c_out, = or_ @ circ[x2, x3]
            circ.add_outputs((s, c_out))

        This is achieved by this method returning an object which encodes the
        association of ports to wires, and supports the application of the ``@``
        operator with a block as the lhs and the object as the rhs.

        :meta public:
        """
        return SelectedBuilderWires(self, wires)

    def __rmatmul__(self, block: Block) -> Wires:
        """
        An alias of ``self.add_block(block)``.

        :meta public:
        """
        return self.add_block(block)

    def __repr__(self) -> str:
        attrs: list[str] = []
        num_wires = self.wiring.num_wires
        num_blocks = len(self.__blocks)
        num_open_slots = self.wiring.num_slots - num_blocks
        num_out_ports = len(self.wiring.out_wires)
        if num_wires > 0:
            attrs.append(f"{num_wires} wires")
        if num_open_slots > 0:
            attrs.append(f"{num_open_slots} open slots")
        if num_blocks > 0:
            attrs.append(f"{num_blocks} blocks")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out ports")
        return f"<DiagramBuilder {id(self):#x}: {", ".join(attrs)}>"


@final
class SelectedBuilderWires(metaclass=TensorSatMeta):
    """
    Utility class wrapping a selection of wires in a given diagram builder,
    to be used for the purposes of adding blocks to the builder.

    Supports usage of the ``@`` operator with a block on the lhs,
    enabling special syntax for addition of blocks to diagrams.
    See :meth:`DiagramBuilder.__getitem__`.
    """

    @classmethod
    def _new(
        cls,
        builder: DiagramBuilder,
        wires: Wires,
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__builder = builder
        self.__wires = wires
        return self

    __builder: DiagramBuilder
    __wires: Wires

    def __new__(
        cls,
        builder: DiagramBuilder,
        wires: Wire | Sequence[Wire],
    ) -> Self:
        assert validate(builder, DiagramBuilder)

        if isinstance(wires, Wire):
            wires = (wires,)
        else:
            assert validate(wires, Sequence[Port])
            wires = tuple(wires)
        builder_wires = builder.wiring.wires
        for wire in wires:
            if wire not in builder_wires:
                raise ValueError(f"Invalid wire {wire} selected for {builder}")
        return cls._new(builder, wires)

    @property
    def builder(self) -> DiagramBuilder:
        """The builder to which the selected wires belong."""
        return self.__builder

    @property
    def wires(self) -> Wires:
        """The selected wires."""
        return self.__wires

    def __rmatmul__(self, other: Block | SelectedBlockPorts) -> Wires:
        """
        Adds the given block to the diagram, applied to the selected wires.

        :meta public:
        """
        wires = self.wires
        if isinstance(other, (Box, Diagram)):
            if len(other.ports) < len(wires):
                raise ValueError(
                    f"Cannot apply block: {len(wires)} wires selected,"
                    f" {other.num_ports} ports available on RHS."
                )
            other = SelectedBlockPorts(other, range(len(wires)))
        elif isinstance(other, SelectedBlockPorts):
            if len(other.ports) != len(wires):
                raise ValueError(
                    f"Cannot apply block: {len(wires)} wires selected,"
                    f" {len(other.ports)} ports selected on block."
                )
        else:
            return NotImplemented
        return self.builder.add_block(other.block, dict(zip(other.ports, wires)))

    def __repr__(self) -> str:
        return f"<DiagramBuilder {id(self.builder):#x}>[{self.wires}]"
