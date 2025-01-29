"""
Diagrams for compact-closed languages.

Diagrams (cf. :class:`Diagram`) consist of boxes (cf. :class:`Box`) and/or sub-diagrams
wired together (cf. :class:`Wiring`) in such a way as to respect the types
(cf. :class:`Type`) declared by boxes for their ports.
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import accumulate, chain
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    Self,
    Type as SubclassOf,
    TypeVar,
    TypedDict,
    cast,
    final,
    overload,
)
from hashcons import InstanceStore

if __debug__:
    from typing_validation import validate


class Type:
    """
    Abstract base class for types in diagrams.

    Types are used to signal compatibility between boxes, by requiring that ports wired
    together in a diagram all have the same type.
    By sharing common types, boxes from multiple languages can be wired together in the
    same diagram.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """Constructs a new type."""
        if cls is Type:
            raise TypeError("Cannot instantiate abstract class Type.")
        return super().__new__(cls)

    @final
    def spider(self, num_ports: int) -> Box[Self]:
        """
        The box corresponding to a single wire connected to the given number of ports,
        all ports being of this type.
        """
        validate(num_ports, int)
        if num_ports <= 0:
            raise ValueError("Number of ports must be strictly positive.")
        return self._spider(num_ports)

    @abstractmethod
    def _spider(self, num_ports: int) -> Box[Self]:
        """
        Protected version of :meth:`Type.spider`, to be implemented by subclasses.
        It is guaranteed that ``num_ports`` is strictly positive.
        """

    @final
    def __mul__[T: Self, _T: Type](self: T, other: _T | Shape[_T]) -> Shape[T | _T]:
        """Takes the product of this type with another type or shape."""
        if isinstance(other, Shape):
            return Shape([self, *other])
        return Shape([self, other])

    @final
    def __pow__[T: Self](self: T, rhs: int, /) -> Shape[T]:
        """Repeats a type a given number of times."""
        assert validate(rhs, int)
        return Shape([self] * rhs)


TypeT_co = TypeVar("TypeT_co", bound=Type, covariant=True)
"""Covariant type variable for a type."""

TypeT_inv = TypeVar("TypeT_inv", bound=Type)
"""Invariant type variable for a type."""


@final
class Shape(Generic[TypeT_co]):
    """A Shape, as a finite tuple of types."""

    _store: ClassVar[InstanceStore] = InstanceStore()

    @classmethod
    def prod(cls, shapes: Iterable[Shape[TypeT_co]], /) -> Shape[TypeT_co]:
        """Takes the product of multiple shapes, i.e. concatenates their types."""
        shapes = tuple(shapes)
        assert validate(shapes, tuple[Shape[Type], ...])
        return Shape._prod(shapes)

    @classmethod
    def _prod(cls, shapes: tuple[Shape[TypeT_co], ...], /) -> Shape[TypeT_co]:
        return cls._new(sum((shape.__components for shape in shapes), ()))

    __components: tuple[TypeT_co, ...]

    __slots__ = ("__weakref__", "__components")

    @classmethod
    def _new(cls, components: tuple[TypeT_co, ...]) -> Self:
        """Protected constructor."""
        with Shape._store.instance(cls, components) as self:
            if self is None:
                self = super().__new__(cls)
                self.__components = components
                Shape._store.register(self)
        return self

    def __new__(cls, components: Iterable[TypeT_co]) -> Self:
        """
        Constructs a new shape with given component types.
        If iterables of types are passed, their types are extracted and inserted
        into the shape at the selected point.
        """
        components = tuple(components)
        assert validate(components, tuple[Type, ...])
        return cls._new(components)

    def __mul__[T: Type](self, rhs: Shape[T], /) -> Shape[TypeT_co | T]:
        """Takes the product of two shapes (i.e. concatenates their types)."""
        assert validate(rhs, Shape)
        return Shape([*self, *rhs])

    def __pow__(self, rhs: int, /) -> Shape[TypeT_co]:
        """Repeats a shape a given number of times."""
        assert validate(rhs, int)
        return Shape(tuple(self) * rhs)

    def __iter__(self) -> Iterator[TypeT_co]:
        """Iterates over the components of the shape."""
        return iter(self.__components)

    def __len__(self) -> int:
        """Returns the number of components in the shape."""
        return len(self.__components)

    @overload
    def __getitem__(self, index: int, /) -> TypeT_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Shape[TypeT_co]: ...
    def __getitem__(self, index: int | slice, /) -> TypeT_co | Shape[TypeT_co]:
        """Returns the component(s) at the given index(es)."""
        if isinstance(index, slice):
            return Shape(self.__components[index])
        assert validate(index, int)
        return self.__components[index]

    def __repr__(self) -> str:
        return f"Shape({self.__components})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.__components == other.__components

    def __hash__(self) -> int:
        return hash((Shape, self.__components))


type Slot = int
"""Type alias for (the index of) a slot in a diagram."""

type Port = int
"""Type alias for (the index of) a port in a diagram."""

type Wire = int
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


class Box(Shaped[TypeT_co], ABC):
    """
    Abstract base class for boxes in diagrams.
    """

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
            excluded = set(lhs_wires) & set(rhs_wires)
            out_wires = []
            for w in chain(lhs_wires, rhs_wires):
                if w not in excluded:
                    out_wires.append(w)
                    excluded.add(w)
        else:
            out_wires_set = set(out_wires)
            if len(out_wires) != len(out_wires_set):
                raise NotImplementedError("Output wires cannot be repeated.")
                # TODO: This is not ordinarily handled by einsum,
                #       but it is pretty natural in the context of the wirings we use,
                #       so we might wish to add support for it in the future.
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
        Protected version of :meth:`Box.contract2`, to be implemented by subclasses.
        It is guaranteed that:
        - The length of ``lhs_wires`` matches the length of ``lhs.shape``
        - The length of ``rhs_wires`` matches the length of ``rhs.shape``
        - Indices in ``out_wires`` are not repeated
        - Every index in ``out_wires`` appears in ``lhs_wires`` or ``rhs_wires``
        """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """Constructs a new box."""
        if cls is Box:
            raise TypeError("Cannot instantiate abstract class Box.")
        return super().__new__(cls)

    @final
    def transpose(self, perm: Sequence[Port]) -> Self:
        """Transposes the output ports into the given order."""
        assert validate(perm, Sequence[Port])
        if len(perm) != self.num_ports or set(perm) != set(self.ports):
            raise ValueError(
                "Input to transpose method must be a permutation of the box ports."
            )
        return self._transpose(perm)

    @abstractmethod
    def _transpose(self, perm: Sequence[Port]) -> Self:
        """
        Protected version of :meth:`Box.transpose`, to be implemented by subclasses.
        It is guaranteed that ``perm`` is a permutation of ``range(self.ports)``.
        """

    def __mul__(self, other: Self) -> Self:
        """
        Takes the product of this relation with another relation of the same class.
        The resulting relation has as its ports the ports of this relation followed
        by the ports of the other relation, and is of the same class of both.
        """
        lhs, rhs = self, other
        lhs_len, rhs_len = len(lhs.shape), len(rhs.shape)
        lhs_wires, rhs_wires = range(lhs_len), range(lhs_len, lhs_len + rhs_len)
        out_wires = range(lhs_len + rhs_len)
        return type(self)._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)


type Block[T: Type] = Box[T] | Diagram[T]
"""
Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

"""


@final
class Diagram(Shaped[TypeT_co]):
    """
    A diagram, consisting of a :class:`Wiring` together with :obj:`Block`s associated
    to (a subset of) the wiring's slots.
    """

    _store: ClassVar[InstanceStore] = InstanceStore()

    @staticmethod
    def from_recipe[
        T: Type
    ](
        input_types: Sequence[T],
    ) -> Callable[
        [Callable[[DiagramBuilder[T], tuple[Wire, ...]], Sequence[Wire]]], Diagram[T]
    ]:
        """
        Given an input shape, returns a function decorator which makes it possible
        to define a diagram by providing a building recipe.
        For example, the snippet below creates the :class:`Diagram` instance ``hadd``
        for a half-adder circuit, by wrapping a recipe using a diagram builder
        internally:

        .. code-block:: python

            from typing import reveal_type
            from quetz.langs.rel import bit
            from quetz.libs.bincirc import and_, or_, xor_

            @Diagram.from_recipe(bit*3)
            def hadd(circ: DiagramBuilder, inputs: Sequence[Wire]) -> Sequence[Wire]:
                a, b, c_in = inputs
                x1, = xor_ @ circ[a, b]
                x2, = and_ @ circ[a, b]
                x3, = and_ @ circ[x1, c_in]
                s, = xor_ @ circ[x1, x3]
                c_out, = or_ @ circ[x2, x3]
                return s, c_out

            reveal_type(hadd) # Diagram

        The diagram creation process is as follows:

        1. A blank diagram builder is created.
        2. Inputs of the given types are added to the builder.
        3. The recipe is called on the builder and input wires.
        4. The recipe returns the output wires.
        5. The outputs are added to the builder.
        6. The diagram is created from the builder and returned.

        """
        return lambda recipe: DiagramRecipe(recipe)(input_types)

    @staticmethod
    def recipe(
        recipe: Callable[[DiagramBuilder[TypeT_co], tuple[Wire, ...]], Sequence[Wire]]
    ) -> DiagramRecipe[TypeT_co]:
        """
        Returns a function decorator which makes it possible to define diagrams
        by providing a building recipe:

        .. code-block:: python

            @Diagram.recipe
            def ripply_carry_adder(
                circ: CircuitBuilder,
                inputs: Sequence[Wire]
            ) -> Sequence[Wire]:
                if len(inputs) % 2 != 1:
                    raise ValueError("Ripple carry adder expects odd number of inputs.")
                num_bits = len(inputs) // 2
                outputs: list[int] = []
                c = inputs[0]
                for i in range(num_bits):
                    a, b = inputs[2 * i + 1 : 2 * i + 3]
                    s, c = full_adder @ circ[c, a, b]
                    outputs.append(s)
                outputs.append(c)
                return tuple(outputs)

            # ...later on, applying a 2-bit RCA at some point of some circuit...
            s0, s1, c_out = ripply_carry_adder @ some_circuit[c_in, a0, b0, a1, b1]

        Unlike the :func:`Diagram.from_recipe`, where decoration resulted in a
        fixed :class:`Diagram` instance, this decorator returns a
        :class:`DiagramRecipe` object, which creates an actual diagram just-in-time
        at the point of application to selected diagram builder wires.

        """

        return DiagramRecipe(recipe)

    @classmethod
    def _new(
        cls, wiring: Wiring[TypeT_co], blocks: tuple[Block[TypeT_co] | None, ...]
    ) -> Self:
        """Protected constructor."""
        with Diagram._store.instance(cls, (wiring, blocks)) as self:
            if self is None:
                self = super().__new__(cls)
                self.__wiring = wiring
                self.__blocks = blocks
                Diagram._store.register(self)
            return self

    __wiring: Wiring[TypeT_co]
    __blocks: tuple[Box[TypeT_co] | Diagram[TypeT_co] | None, ...]

    __slots__ = ("__weakref__", "__wiring", "__blocks")

    def __new__(
        cls, wiring: Wiring[TypeT_co], blocks: Mapping[Slot, Block[TypeT_co]]
    ) -> Self:
        """Constructs a new diagram from a wiring and blocks for (some of) its slots."""
        assert validate(wiring, Wiring)
        wiring.validate_slot_data(blocks)
        _blocks = tuple(map(blocks.get, wiring.slots))
        return cls._new(wiring, _blocks)

    @property
    def wiring(self) -> Wiring[TypeT_co]:
        """Wiring for the diagram."""
        return self.__wiring

    @property
    def blocks(self) -> tuple[Block[TypeT_co] | None, ...]:
        """
        Sequence of blocks associated to the slots in the diagram's wiring,
        or :obj:`None` to indicate that a slot is open.
        """
        return self.__blocks

    @property
    def shape(self) -> Shape[TypeT_co]:
        """Shape of the diagram."""
        return self.wiring.shape

    @property
    def open_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which are open in the diagram."""
        return tuple(slot for slot, block in enumerate(self.blocks) if block is None)

    @property
    def num_open_slots(self) -> int:
        """Number of open slots in the diagram."""
        return self.blocks.count(None)

    @property
    def subdiagram_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Diagram)
        )

    @property
    def subdiagrams(self) -> tuple[Diagram[TypeT_co], ...]:
        """Diagrams associated to the slots in :prop:`subdiagram_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Diagram))

    @property
    def box_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Box)
        )

    @property
    def boxes(self) -> tuple[Box[TypeT_co], ...]:
        """Boxes associated to the slots in :prop:`box_slots`."""
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

    def compose(
        self, new_blocks: Mapping[Slot, Block[TypeT_co] | Wiring[TypeT_co]]
    ) -> Diagram[TypeT_co]:
        """
        Composes this wiring with the given boxes, diagrams and/or wirings
        for (some of) its slots.
        """
        assert validate(new_blocks, Mapping[Slot, Block[TypeT_co] | Wiring[TypeT_co]])
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
        merged_blocks: list[Block[TypeT_co] | None] = []
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
        return Diagram._new(merged_wiring, tuple(merged_blocks))

    def flatten(self, *, cache: bool = True) -> Diagram[TypeT_co]:
        """
        Returns a recursively diagram, obtained by recursively flattening all
        sub-diagrams, composing their wirings into the current wiring, and taking
        all blocks (of this diagrams and its sub-diagrams) as the blocks of the result.
        """
        assert validate(cache, bool)
        return self._flatten({} if cache else None)

    def _flatten(
        self, cache: dict[Diagram[TypeT_co], Diagram[TypeT_co]] | None
    ) -> Diagram[TypeT_co]:
        if cache is not None and self in cache:
            return cache[self]
        flat_subdiagrams = [
            subdiagram._flatten(cache) for subdiagram in self.subdiagrams
        ]
        subwirings = [subdiag.wiring for subdiag in flat_subdiagrams]
        flat_wiring = self.wiring.compose(dict(zip(self.subdiagram_slots, subwirings)))
        flat_blocks: list[Box[TypeT_co] | None] = []
        subdiagram_slots = {slot: idx for idx, slot in enumerate(self.subdiagram_slots)}
        for slot, block in enumerate(self.blocks):
            if (idx := subdiagram_slots.get(slot)) is not None:
                flat_blocks.extend(
                    cast(tuple[Box[TypeT_co] | None, ...], flat_subdiagrams[idx].blocks)
                )
            else:
                flat_blocks.append(cast(Box[TypeT_co] | None, block))
        flat_diagram = Diagram._new(flat_wiring, tuple(flat_blocks))
        if cache is not None:
            cache[self] = flat_diagram
        return self


@final
class DiagramBuilder(Generic[TypeT_inv]):
    """Utility class to build diagrams."""

    __wiring_builder: WiringBuilder[TypeT_inv]
    __blocks: dict[Slot, Block[TypeT_inv]]

    __slots__ = ("__weakref__", "__wiring_builder", "__blocks")

    def __new__(cls) -> Self:
        """Creates a blank diagram builder."""
        self = super().__new__(cls)
        self.__wiring_builder = WiringBuilder()
        self.__blocks = {}
        return self

    def copy(self) -> DiagramBuilder[TypeT_inv]:
        """Returns a deep copy of this diagram builder."""
        clone: DiagramBuilder[TypeT_inv] = DiagramBuilder.__new__(DiagramBuilder)
        clone.__wiring_builder = self.__wiring_builder.copy()
        clone.__blocks = self.__blocks.copy()
        return clone

    @property
    def wiring(self) -> WiringBuilder[TypeT_inv]:
        """The wiring builder for the diagram."""
        return self.__wiring_builder

    @property
    def blocks(self) -> MappingProxyType[Slot, Block[TypeT_inv]]:
        """Blocks in the diagram."""
        return MappingProxyType(self.__blocks)

    @property
    def diagram(self) -> Diagram[TypeT_inv]:
        """The diagram built thus far."""
        wiring = self.__wiring_builder.wiring
        blocks = self.__blocks
        _blocks = tuple(blocks.get(slot) for slot in wiring.slots)
        return Diagram._new(wiring, _blocks)

    def set_block(self, slot: Slot, block: Block[TypeT_inv]) -> None:
        """Sets a block for an existing open slot."""
        assert validate(block, Block)
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
        self, block: Block[TypeT_inv], inputs: Mapping[Port, Wire]
    ) -> tuple[Wire, ...]:
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
        4. Returns the newly created wires (in port order).
        """
        wire_types = self.__wiring_builder.__wire_types
        assert validate(block, Block)
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

    def _add_block(
        self, block: Block[TypeT_inv], inputs: Mapping[Port, Wire]
    ) -> tuple[Wire, ...]:
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
        return output_wires

    def add_inputs(self, ts: Sequence[TypeT_inv]) -> tuple[Wire, ...]:
        """
        Creates new wires of the given types,
        then adds ports connected to those wires.
        """
        assert validate(ts, Sequence[TypeT_inv])
        return self._add_inputs(ts)

    def _add_inputs(self, ts: Sequence[TypeT_inv]) -> tuple[Wire, ...]:
        wiring = self.wiring
        wires = wiring._add_wires(ts)
        wiring._add_outer_ports(wires)
        return wires

    def add_outputs(self, wires: Sequence[Wire]) -> None:
        """Adds ports connected to the given wires."""
        assert validate(wires, Sequence[Wire])
        diag_wires = self.wiring.wires
        for wire in wires:
            if wire not in diag_wires:
                raise ValueError(f"Invalid wire index {wire}.")
        self._add_outputs(wires)

    def _add_outputs(self, wires: Sequence[Wire]) -> None:
        self.wiring._add_outer_ports(wires)

    def __getitem__(
        self, wires: Sequence[Wire] | Mapping[Port, Wire]
    ) -> SelectedInputWires[TypeT_inv]:
        """
        Enables special syntax for addition of blocks to the diagram:

        .. code-block:: python
            from quetz.langs.rel import bit
            from quetz.libs.bincirc import and_, or_, xor_

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
        """
        return SelectedInputWires(self, wires)


@final
class SelectedInputWires(Generic[TypeT_inv]):
    """
    Utility class wrapping a selection of input wires in a given diagram builder,
    to be used for the purposes of adding blocks to the builder.

    Supports usage of the ``@`` operator with a block on the lhs,
    enabling special syntax for addition of blocks to diagrams.
    See :meth:`DiagramBuilder.__getitem__`.
    """

    @classmethod
    def _new(
        cls,
        builder: DiagramBuilder[TypeT_inv],
        wires: MappingProxyType[Port, Wire] | tuple[Wire, ...],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__builder = builder
        self.__wires = wires
        return self

    __builder: DiagramBuilder[TypeT_inv]
    __wires: MappingProxyType[Port, Wire] | tuple[Wire, ...]

    __slots__ = ("__weakref__", "__builder", "__wires")

    def __new__(
        cls,
        builder: DiagramBuilder[TypeT_inv],
        wires: Sequence[Wire] | Mapping[Port, Wire],
    ) -> Self:
        assert validate(builder, DiagramBuilder)
        _wires: MappingProxyType[Port, Wire] | tuple[Wire, ...]
        if isinstance(wires, Mapping):
            assert validate(wires, Mapping[Port, Wire])
            _wires = MappingProxyType({**wires})
        else:
            assert validate(wires, Sequence[Wire])
            _wires = tuple(wires)
        builder_wires = builder.wiring.wires
        for wire in _wires if isinstance(_wires, tuple) else _wires.values():
            if wire not in builder_wires:
                raise ValueError(f"Invalid wire index {wire}.")
        return cls._new(builder, _wires)

    @property
    def builder(self) -> DiagramBuilder[TypeT_inv]:
        """The builder to which the selected input wires belong."""
        return self.__builder

    @property
    def wires(self) -> MappingProxyType[Port, Wire] | tuple[Wire, ...]:
        """
        The selected input wires, as either:

        - a tuple of wires, implying contiguous port selection starting at index 0
        - a mapping of ports to wires

        """
        return self.__wires

    def __rmatmul__(self, block: Block[TypeT_inv]) -> tuple[Wire, ...]:
        """Adds the given block to the diagram, applied to the selected input wires."""
        if not isinstance(block, (Box, Diagram)):
            return NotImplemented
        if isinstance(self.__wires, MappingProxyType):
            wires = self.__wires
        else:
            wires = MappingProxyType(dict(enumerate(self.__wires)))
        return self.__builder.add_block(block, wires)


@final
class DiagramRecipe(Generic[TypeT_inv]):
    """
    Utility class wrapping diagram building logic, which can be executed on
    demand for given input types.

    Supports usage of the ``@`` operator with selected input wires on the rhs,
    analogously to the special block addition syntax for diagram builders.

    See the :func:`Diagram.from_recipe` and :func:`Diagram.recipe` decorators for
    examples of how this works.
    """

    __recipe: Callable[[DiagramBuilder[TypeT_inv], tuple[Wire, ...]], Sequence[Wire]]

    __slots__ = ("__weakref__", "__recipe")

    def __new__(
        cls,
        recipe: Callable[[DiagramBuilder[TypeT_inv], tuple[Wire, ...]], Sequence[Wire]],
    ) -> Self:
        """Wraps the given diagram building logic."""
        self = super().__new__(cls)
        self.__recipe = recipe
        return self

    def __call__(self, input_types: Sequence[TypeT_inv]) -> Diagram[TypeT_inv]:
        """Executes the recipe for the given input types, returning the diagram."""
        builder: DiagramBuilder[TypeT_inv] = DiagramBuilder()
        inputs = builder._add_inputs(input_types)
        outputs = self.__recipe(builder, inputs)
        builder._add_outputs(outputs)
        return builder.diagram

    def __matmul__(self, selected: SelectedInputWires[TypeT_inv]) -> tuple[Wire, ...]:
        """
        Executes the recipe for the input types specified by the selected input wires,
        then adds the diagram resulting from the recipe as a block in the the broader
        diagram being built, connected to the given input wires, and returns the
        resulting output wires.
        """
        selected_wires = selected.wires
        num_ports = len(selected_wires)
        if set(selected_wires) != set(range(num_ports)):
            raise ValueError("Selected ports must form a contiguous zero-based range.")
        wire_types = selected.builder.wiring.wire_types
        input_types = tuple(
            wire_types[selected_wires[port]] for port in range(num_ports)
        )
        diagram = self(input_types)
        return diagram @ selected
