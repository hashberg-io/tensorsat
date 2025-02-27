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
from abc import ABC, ABCMeta, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    overload,
    override,
)
from weakref import WeakValueDictionary

from hashcons import InstanceStore

if TYPE_CHECKING:
    from .contraction import Contraction
else:
    Contraction = Any

if __debug__:
    from typing_validation import validate


# TODO: Create TypeMeta to track types.
#       Automate registration of concrete Type subclasses into their language (module).
#       Make it possible to subclass concrete Type classes, to allow overlapping langs.
#       It makes sense to consider alternative parametrisations for types in diff langs.


class Type(ABC):
    """
    Abstract base class for types in diagrams.

    Types are used to signal compatibility between boxes, by requiring that ports wired
    together in a diagram all have the same type.
    By sharing common types, boxes from multiple languages can be wired together in the
    same diagram.
    """

    __final__: ClassVar[bool] = False

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """
        Constructs a new type.

        :meta public:
        """
        if not cls.__final__:
            raise TypeError("Only final subclasses of Type can be instantiated.")
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
    def __mul__[_S: Self, _T: Type](self: _S, other: _T | Shape[_T]) -> Shape[_S | _T]:
        """
        Takes the product of this type with another type or shape.

        :meta public:
        """
        if isinstance(other, Shape):
            return Shape([self, *other])
        return Shape([self, other])

    @final
    def __pow__[_S: Self](self: _S, rhs: int, /) -> Shape[_S]:
        """
        Repeats a type a given number of times.

        :meta public:
        """
        assert validate(rhs, int)
        return Shape([self] * rhs)


TypeT_co = TypeVar("TypeT_co", bound=Type, covariant=True, default=Type)
"""Covariant type variable for a type."""

TypeT_inv = TypeVar("TypeT_inv", bound=Type, default=Type)
"""Invariant type variable for a type."""


class Shape(Sequence[TypeT_co]):
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

        :meta public:
        """
        components = tuple(components)
        assert validate(components, tuple[Type, ...])
        return cls._new(components)

    def __mul__[_T: Type](self, rhs: _T | Shape[_T], /) -> Shape[TypeT_co | _T]:
        """
        Takes the product of two shapes (i.e. concatenates their types).

        :meta public:
        """
        if isinstance(rhs, Type):
            return Shape([*self, cast(_T, rhs)])
        assert validate(rhs, Shape)
        return Shape([*self, *rhs])

    def __pow__(self, rhs: int, /) -> Shape[TypeT_co]:
        """
        Repeats a shape a given number of times.

        :meta public:
        """
        assert validate(rhs, int)
        return Shape(tuple(self) * rhs)

    def __iter__(self) -> Iterator[TypeT_co]:
        """
        Iterates over the components of the shape.

        :meta public:
        """
        return iter(self.__components)

    def __len__(self) -> int:
        """
        Returns the number of components in the shape.

        :meta public:
        """
        return len(self.__components)

    @overload
    def __getitem__(self, index: int, /) -> TypeT_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Shape[TypeT_co]: ...
    @overload
    def __getitem__(self, index: Iterable[int], /) -> Shape[TypeT_co]: ...
    def __getitem__(
        self, index: int | slice | Iterable[int], /
    ) -> TypeT_co | Shape[TypeT_co]:
        """
        Returns the component(s) at the given index(es).

        :meta public:
        """
        if isinstance(index, slice):
            return Shape._new(self.__components[index])
        if isinstance(index, int):
            return self.__components[index]
        index = list(index)
        assert validate(index, list[int])
        components = self.__components
        return Shape._new(tuple(components[i] for i in index))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.__components == other.__components

    def __hash__(self) -> int:
        return hash((Shape, self.__components))

    def __repr__(self) -> str:
        return f"<Shape {id(self):#x}: {len(self)} components>"


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
    def wrap_shape[_T: Type](shape: Shape[_T]) -> Shaped[_T]:
        """Wraps a shape into an anonymous :class:`Shaped` instance."""
        assert validate(shape, Shape)
        cls: SubclassOf[Shaped[_T]] = final(
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
    def wrap_slot_shapes[_T: Type](slot_shapes: tuple[Shape[_T], ...]) -> Slotted[_T]:
        """Wraps a tuple of shapes into an anonymous :class:`Slotted` instance."""
        assert validate(slot_shapes, tuple[Shape[Type], ...])
        cls: SubclassOf[Slotted[_T]] = final(
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
        """
        Constructs a wiring from the given data.

        :meta public:
        """
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
        # 4. Compute new slot wires:
        new_slot_wires_list: list[tuple[Wire, ...]] = []
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
class WiringBuilder[_T: Type](WiringBase[_T]):
    """Utility class to build wirings."""

    __slot_shapes: list[list[_T]]
    __shape: list[_T]
    __wire_types: list[_T]
    __slot_wires_list: list[list[Wire]]
    __out_wires: list[Wire]

    __slot_shapes_cache: tuple[Shape[_T], ...] | None
    __shape_cache: Shape[_T] | None
    __wire_types_cache: Shape[_T] | None
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
    def slot_shapes(self) -> tuple[Shape[_T], ...]:
        slot_shapes = self.__slot_shapes_cache
        if slot_shapes is None:
            self.__slot_shapes_cache = slot_shapes = tuple(
                Shape._new(tuple(s)) for s in self.__slot_shapes
            )
        return slot_shapes

    @property
    def shape(self) -> Shape[_T]:
        shape = self.__shape_cache
        if shape is None:
            self.__shape_cache = shape = Shape._new(tuple(self.__shape))
        return shape

    @property
    def wire_types(self) -> Shape[_T]:
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
    def wiring(self) -> Wiring[_T]:
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

    def copy(self) -> WiringBuilder[_T]:
        """Returns a deep copy of this wiring builder."""
        clone: WiringBuilder[_T] = WiringBuilder.__new__(WiringBuilder)
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

    def add_wire(self, t: _T) -> Wire:
        """Adds a new wire with the given type."""
        assert validate(t, Type)
        return self._add_wires([t])[0]

    def add_wires(self, ts: Sequence[_T]) -> tuple[Wire, ...]:
        """Adds new wires with the given types."""
        assert validate(ts, Sequence[Type])
        return self._add_wires(ts)

    def _add_wires(self, ts: Sequence[_T]) -> tuple[Wire, ...]:
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


# TODO: Improve BoxMeta to track boxes.
#       Automate registration of concrete Box subclasses into their language (module).
#       Make it possible to subclass concrete Box classes, to allow overlapping langs.
#       It makes sense to consider alternative parametrisations for boxes in diff langs.

# TODO: consider introducing box labels, for builtin boxes


class BoxMeta(ABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> BoxMeta:
        cls = super().__new__(mcs, name, bases, namespace)
        if not cls.__abstractmethods__:
            try:
                import autoray  # type: ignore

                autoray.register_backend(cls, "tensorsat._autoray")
            except ModuleNotFoundError:
                pass
        return cls


class Box(Shaped[TypeT_co], metaclass=BoxMeta):
    """
    Abstract base class for boxes in diagrams.
    """

    __final__: ClassVar[bool] = False

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

    # @final
    # @staticmethod
    # def recipe(
    #     recipe: Callable[[Shape[TypeT_inv]], BoxT_inv],
    # ) -> BoxRecipe[TypeT_inv, BoxT_inv]:
    #     """Wraps """
    #     return BoxRecipe(recipe)

    # __recipe_used: BoxRecipe[TypeT_co, Self] | None

    __slots__ = (
        "__weakref__",
        # "__recipe_used"
    )

    def __new__(cls) -> Self:
        """
        Constructs a new box.

        :meta public:
        """
        if not cls.__final__:
            raise TypeError("Only final subclasses of Box can be instantiated.")
        self = super().__new__(cls)
        # self.__recipe_used = None
        return self

    # @final
    # @property
    # def recipe_used(self) -> BoxRecipe[TypeT_co, Self] | None:
    #     """The recipe used to create the box, if any."""
    #     return self.__recipe_used

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

        :meta public:
        """
        lhs, rhs = self, other
        lhs_len, rhs_len = len(lhs.shape), len(rhs.shape)
        lhs_wires, rhs_wires = range(lhs_len), range(lhs_len, lhs_len + rhs_len)
        out_wires = range(lhs_len + rhs_len)
        return type(self)._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        num_ports = len(self.shape)
        return f"<{cls_name} {id(self):#x}: {num_ports} ports>"

BoxT_inv = TypeVar("BoxT_inv", bound=Box, default=Box)
"""Invariant type variable for boxes."""


type Block[T: Type] = Box[T] | Diagram[T]
"""
Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

"""

# TODO: Consider making diagrams parametric in BoxT_co as well,
#       even though we cannot (yet?) make BoxT_co bound to on Box[TypeT_co]

RecipeParams = ParamSpec("RecipeParams")
"""Parameter specification variable for the parameters of a recipe."""


class DiagramRecipe(Generic[RecipeParams, TypeT_inv]):
    """A Recipe to produce diagrams from given perameters."""

    __diagrams: WeakValueDictionary[Any, Diagram[TypeT_inv]]
    __recipe: Callable[Concatenate[DiagramBuilder[TypeT_inv], RecipeParams], None]

    __slots__ = ("__weakref__", "__diagrams", "__recipe")

    def __new__(
        cls,
        recipe: Callable[Concatenate[DiagramBuilder[TypeT_inv], RecipeParams], None],
    ) -> Self:
        self = super().__new__(cls)
        self.__recipe = recipe
        self.__diagrams = WeakValueDictionary()
        return self

    @property
    def name(self) -> str:
        """The name of this recipe."""
        return self.__recipe.__name__

    def __call__(
        self, *args: RecipeParams.args, **kwargs: RecipeParams.kwargs
    ) -> Diagram[TypeT_inv]:
        """
        Returns the diagram constructed by the recipe on given arguments.

        :meta public:
        """
        key = (args, frozenset(kwargs.items()))
        if key in self.__diagrams:
            return self.__diagrams[key]
        builder: DiagramBuilder[TypeT_inv] = DiagramBuilder()
        self.__recipe(builder, *args, **kwargs)
        diagram = builder.diagram
        diagram._Diagram__recipe_used = self  # type: ignore[attr-defined]
        self.__diagrams[key] = diagram
        return diagram

    def __repr__(self) -> str:
        """Representation of the recipe."""
        recipe = self.__recipe
        mod = recipe.__module__
        name = recipe.__name__
        return f"Diagram.recipe({mod}.{name})"


@final
class Diagram(Shaped[TypeT_co]):
    """
    A diagram, consisting of a :class:`Wiring` together with :obj:`Block` associated
    to (a subset of) the wiring's slots.
    """

    @staticmethod
    def from_recipe(
        recipe: Callable[[DiagramBuilder[TypeT_inv]], None],
    ) -> Diagram[TypeT_inv]:
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
        builder: DiagramBuilder[TypeT_inv] = DiagramBuilder()
        recipe(builder)
        diagram = builder.diagram
        # TODO: store recipe name, if available
        return diagram

    @staticmethod
    def recipe(
        recipe: Callable[Concatenate[DiagramBuilder[TypeT_inv], RecipeParams], None],
    ) -> Callable[RecipeParams, Diagram[TypeT_inv]]:
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
    def _new(
        cls, wiring: Wiring[TypeT_co], blocks: tuple[Block[TypeT_co] | None, ...]
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__wiring = wiring
        self.__blocks = blocks
        # self.__recipe_used = None
        return self

    __wiring: Wiring[TypeT_co]
    __blocks: tuple[Box[TypeT_co] | Diagram[TypeT_co] | None, ...]
    __recipe_used: DiagramRecipe[Any, TypeT_co] | None
    __hash_cache: int

    __slots__ = ("__weakref__", "__wiring", "__blocks", "__recipe_used", "__hash_cache")

    def __new__(
        cls, wiring: Wiring[TypeT_co], blocks: Mapping[Slot, Block[TypeT_co]]
    ) -> Self:
        """
        Constructs a new diagram from a wiring and blocks for (some of) its slots.

        :meta public:
        """
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
        """Diagrams associated to the slots in :attr:`subdiagram_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Diagram))

    @property
    def box_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Box)
        )

    @property
    def boxes(self) -> tuple[Box[TypeT_co], ...]:
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
    def recipe_used(self) -> DiagramRecipe[Any, TypeT_co] | None:
        """The recipe used to construct this diagram, if any."""
        return self.__recipe_used

    def contract(self, contraction: Contraction) -> Box[TypeT_co]:
        """Contracts the diagram using the given contraction."""
        return contraction.contract(self)

    # TODO: implement partial contraction, with wiring update logic.

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
        Returns a flat diagram, obtained by recursively flattening all
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
        return flat_diagram

    # TODO: implement diagrammatic contraction
    # ContractionPath = Any # dummy
    # def contract(self, path: ContractionPath | None = None) -> Diagram[TypeT_co]:
    #     raise NotImplementedError()

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
        if recipe:
            attrs.append(f"from recipe {recipe.name!r}")
        return f"<Diagram {id(self):#x}: {", ".join(attrs)}>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Diagram):
            return NotImplemented
        if self is other:
            return True
        return self.__wiring == other.__wiring and self.__blocks == other.__blocks

    def __hash__(self) -> int:
        try:
            return self.__hash_cache
        except AttributeError:
            self.__hash_cache = h = hash((Diagram, self.__wiring, self.__blocks))
            return h


@final
class DiagramBuilder(Generic[TypeT_inv]):
    """Utility class to build diagrams."""

    __wiring_builder: WiringBuilder[TypeT_inv]
    __blocks: dict[Slot, Block[TypeT_inv]]

    __slots__ = ("__weakref__", "__wiring_builder", "__blocks")

    def __new__(cls) -> Self:
        """
        Creates a blank diagram builder.

        :meta public:
        """
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
    def blocks(self) -> Mapping[Slot, Block[TypeT_inv]]:
        """Blocks in the diagram."""
        return MappingProxyType(self.__blocks)

    @property
    def diagram(self) -> Diagram[TypeT_inv]:
        """The diagram built thus far."""
        wiring = self.__wiring_builder.wiring
        blocks = self.__blocks
        _blocks = tuple(blocks.get(slot) for slot in wiring.slots)
        diagram = Diagram._new(wiring, _blocks)
        return diagram

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
        5. Returns the newly created wires (in port order).

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
        self.__blocks[slot] = block
        return output_wires

    def add_inputs(self, ts: Iterable[TypeT_inv]) -> tuple[Wire, ...]:
        """
        Creates new wires of the given types,
        then adds ports connected to those wires.
        """
        ts = tuple(ts)
        assert validate(ts, tuple[TypeT_inv, ...])
        return self._add_inputs(ts)

    def _add_inputs(self, ts: tuple[TypeT_inv, ...]) -> tuple[Wire, ...]:
        wiring = self.wiring
        wires = wiring._add_wires(ts)
        wiring._add_out_ports(wires)
        return wires

    def add_outputs(self, wires: Iterable[Wire]) -> None:
        """Adds ports connected to the given wires."""
        wires = tuple(wires)
        assert validate(wires, tuple[Wire, ...])
        diag_wires = self.wiring.wires
        for wire in wires:
            if wire not in diag_wires:
                raise ValueError(f"Invalid wire index {wire}.")
        self._add_outputs(wires)

    def _add_outputs(self, wires: tuple[Wire, ...]) -> None:
        self.wiring._add_out_ports(wires)

    def __getitem__(
        self, wires: Wire | Sequence[Wire] | Mapping[Port, Wire]
    ) -> SelectedInputWires[TypeT_inv]:
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
        return SelectedInputWires(self, wires)

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
        wires: Wire | Sequence[Wire] | Mapping[Port, Wire],
    ) -> Self:
        assert validate(builder, DiagramBuilder)
        _wires: MappingProxyType[Port, Wire] | tuple[Wire, ...]
        if isinstance(wires, Wire):
            _wires = (wires,)
        elif isinstance(wires, Mapping):
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
    def wires(self) -> Mapping[Port, Wire] | tuple[Wire, ...]:
        """
        The selected input wires, as either:

        - a tuple of wires, implying contiguous port selection starting at index 0
        - a mapping of ports to wires

        """
        return self.__wires

    def __rmatmul__(self, block: Block[TypeT_inv]) -> tuple[Wire, ...]:
        """
        Adds the given block to the diagram, applied to the selected input wires.

        :meta public:
        """
        if not isinstance(block, (Box, Diagram)):
            return NotImplemented
        if isinstance(self.__wires, MappingProxyType):
            wires = self.__wires
        else:
            wires = MappingProxyType(dict(enumerate(self.__wires)))
        return self.__builder.add_block(block, wires)

    def __repr__(self) -> str:
        return f"<DiagramBuilder {id(self.__builder):#x}>[{self.__wires}]"
