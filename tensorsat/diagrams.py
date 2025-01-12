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
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import accumulate
from typing import (
    Any,
    Protocol,
    Self,
    Type as SubclassOf,
    TypeAlias,
    TypeVar,
    TypedDict,
    cast,
    final,
    overload,
)

if __debug__:
    from typing_validation import validate


class Type(ABC):
    """
    Abstract base class for types in diagrams.

    Types are primarily used to signal compatibility of contractions, but they can also
    carry useful additional information, such as the cardinality of a set or the
    dimensionality of a vector space.
    """

    @staticmethod
    def unique() -> Type:
        """Returns a unique anonymous type."""
        cls: SubclassOf[Type] = final(
            type.__new__(
                type,
                "<anon type class>",
                (Type,),
                {
                    "__repr__": lambda self: "<anon type>",
                },
            )
        )
        return cls()

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

    @classmethod
    def concat(cls, shapes: Iterable[Shape], /) -> Self:
        """Concatenates multiple shapes."""
        shapes = tuple(shapes)
        assert validate(shapes, tuple[Shape, ...])
        return Shape._concat(shapes)

    @classmethod
    def _concat(cls, shapes: tuple[Shape, ...], /) -> Self:
        return cls._new(sum((shape.__components for shape in shapes), ()))

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.__components == other.__components

    def __hash__(self) -> int:
        return hash((Shape, self.__components))


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


class Shaped(Protocol):
    """Protocol for objects with a shape."""

    __slots__ = ()

    @property
    def shape(self) -> Shape:
        """Shape of the object."""


class WiringBase(Shaped, ABC):
    """Abstract base class for wiring and wiring builder."""

    __slots__ = ("__weakref__",)

    @property
    @abstractmethod
    def slot_shapes(self) -> tuple[Shape, ...]:
        """Shapes for the slots."""

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Outer shape for the wiring."""

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
        return len(self.shape)

    @property
    def num_wires(self) -> int:
        """Number of wires."""
        return len(self.wire_types)

    @property
    def wires(self) -> tuple[Wire, ...]:
        """Sequence of (the indices of) wires."""
        return tuple(range(self.num_wires))


SlotContentT = TypeVar("SlotContentT")
"""Type variable for the type of content in a slot."""

SlotContents: TypeAlias = Sequence[SlotContentT | None] | Mapping[Slot, SlotContentT]
"""Type alias for allowed ways to specify slot contents."""


def _slot_contents_to_tuple(
    num_slots: int, contents: SlotContents[SlotContentT]
) -> tuple[SlotContentT | None, ...]:
    if isinstance(contents, Mapping):
        return tuple(contents.get(k, None) for k in range(num_slots))
    if len(contents) != num_slots:
        raise ValueError(
            "Incorrect number of slot contents:"
            f" expected {num_slots}, got {len(contents)}."
        )
    return tuple(contents)


@final
class Wiring(WiringBase):
    """An immutable wiring."""

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape, ...],
        shape: Shape,
        wire_types: Shape,
        slot_mappings: tuple[tuple[Wire, ...], ...],
        outer_mapping: tuple[Wire, ...],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__slot_shapes = slot_shapes
        self.__shape = shape
        self.__wire_types = wire_types
        self.__slot_mappings = slot_mappings
        self.__outer_mapping = outer_mapping
        return self

    __slot_shapes: tuple[Shape, ...]
    __shape: Shape
    __wire_types: Shape
    __slot_mappings: tuple[tuple[Wire, ...], ...]
    __outer_mapping: tuple[Wire, ...]
    __hash_cache: int

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_mappings",
        "__outer_mapping",
        "__hash_cache",
    )

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
        shape = Shape(wire_types[o] for o in range(num_outer_ports))
        return cls._new(slot_shapes, shape, wire_types, slot_mappings, outer_mapping)

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        return self.__slot_shapes

    @property
    def shape(self) -> Shape:
        return self.__shape

    @property
    def wire_types(self) -> Shape:
        return self.__wire_types

    @property
    def slot_mappings(self) -> tuple[tuple[Wire, ...], ...]:
        return self.__slot_mappings

    @property
    def outer_mapping(self) -> tuple[Wire, ...]:
        return self.__outer_mapping

    def compose(self, wirings: SlotContents[Wiring]) -> Wiring:
        """Composes this wiring with the given wirings for (some of) its slots."""
        assert validate(wirings, SlotContents[Wiring])  # type: ignore[misc]
        wirings = _slot_contents_to_tuple(self.num_slots, wirings)
        return self._compose(wirings)

    def _compose(self, wirings: tuple[Wiring | None, ...]) -> Wiring:
        slot_shapes = self.slot_shapes
        num_wires = self.num_wires
        for slot, (slot_shape, wiring) in enumerate(zip(slot_shapes, wirings)):
            if wiring is not None and wiring.shape != slot_shape:
                raise ValueError(
                    f"Incompatible shape in composition for slot {slot}:"
                    f" expected slot shape {slot_shape},"
                    f" got wiring shape {wiring.shape}."
                )
        _wire_start_idx = list(
            accumulate(
                [
                    num_wires,
                    *(
                        wiring.num_wires if wiring is not None else 0
                        for wiring in wirings
                    ),
                ]
            )
        )
        wiring_remappings = [
            range(start, end)
            for start, end in zip(_wire_start_idx[:-1], _wire_start_idx[1:])
        ]
        new_wire_types = self.wire_types + Shape._concat(
            tuple(wiring.wire_types for wiring in wirings if wiring is not None)
        )
        new_slots_data: list[tuple[Wiring, Slot, range]] = []
        for slot, wiring in enumerate(wirings):
            if wiring is None:
                new_slots_data.append((self, slot, range(num_wires)))
            else:
                new_slots_data.extend(
                    (wiring, wiring_slot, wiring_remappings[slot])
                    for wiring_slot in wiring.slots
                )
        new_slot_shapes = tuple(
            wiring.slot_shapes[wiring_slot]
            for slot, (wiring, wiring_slot, remapping) in enumerate(new_slots_data)
        )
        new_slot_mappings = tuple(
            tuple(remapping[w] for w in wiring.slot_mappings[wiring_slot])
            for slot, (wiring, wiring_slot, remapping) in enumerate(new_slots_data)
        )
        return Wiring._new(
            new_slot_shapes,
            self.shape,
            new_wire_types,
            new_slot_mappings,
            self.outer_mapping,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Wiring):
            return NotImplemented
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in (
                "slot_shapes",
                "shape",
                "wire_types",
                "slot_mappings",
                "outer_mapping",
            )
        )

    def __hash__(self) -> int:
        try:
            return self.__hash_cache
        except AttributeError:
            self.__hash_cache = h = hash(
                (
                    Wiring,
                    self.slot_shapes,
                    self.shape,
                    self.wire_types,
                    self.slot_mappings,
                    self.outer_mapping,
                )
            )
            return h


@final
class WiringBuilder(WiringBase):
    """A wiring builder."""

    __slot_shapes: list[list[Type]]
    __shape: list[Type]
    __wire_types: list[Type]
    __slot_mappings: list[list[Wire]]
    __outer_mapping: list[Wire]

    __slots__ = (
        "__slot_shapes",
        "__shape",
        "__wire_types",
        "__slot_mappings",
        "__outer_mapping",
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
    def slot_shapes(self) -> tuple[Shape, ...]:
        return tuple(Shape(shape) for shape in self.__slot_shapes)

    @property
    def shape(self) -> Shape:
        return Shape(self.__shape)

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
            self.shape,
            self.wire_types,
            self.slot_mappings,
            self.outer_mapping,
        )

    def copy(self) -> WiringBuilder:
        """Returns a deep copy of this wiring builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone.__slot_shapes = [s.copy() for s in self.__slot_shapes]
        clone.__shape = self.__shape.copy()
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
        shape, wire_types = self.__shape, self.__wire_types
        len_before = len(shape)
        shape.extend(wire_types[wire] for wire in wires)
        self.__outer_mapping.extend(wires)
        return tuple(range(len_before, len(shape)))

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

    def _add_slot_ports(self, slot: Slot, wires: Sequence[Wire]) -> tuple[Port, ...]:
        slot_shape, wire_types = self.__slot_shapes[slot], self.__wire_types
        len_before = len(slot_shape)
        slot_shape.extend(wire_types[w] for w in wires)
        self.__slot_mappings[slot].extend(wires)
        return tuple(range(len_before, len(slot_shape)))


class Box(Shaped, ABC):
    """
    Abstract base class for boxes in diagrams.
    """

    @staticmethod
    def unique(shape: Shape) -> Box:
        """Returns a unique anonymous box."""
        cls: SubclassOf[Box] = final(
            type.__new__(
                type,
                "<anon box class>",
                (Box,),
                {
                    "shape": property(lambda self: shape),
                    "__repr__": lambda self: "<anon box>",
                },
            )
        )
        return cls()

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """Constructs a new box."""
        if cls is Box:
            raise TypeError("Cannot instantiate abstract class Box.")
        return super().__new__(cls)

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """
        The shape of the box.
        """


Block: TypeAlias = Box | "Diagram"
"""
Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

"""


@final
class Diagram(Shaped):
    """
    A diagram, consisting of a :class:`Wiring` together with :obj:`Block`s associated
    to (a subset of) the wiring's slots.
    """

    @staticmethod
    def _validate_slot_content_shapes(
        wiring: Wiring, contents: tuple[Shaped | None, ...]
    ) -> None:
        for slot, (shape, content) in enumerate(zip(wiring.slot_shapes, contents)):
            if content is not None and content.shape != shape:
                raise ValueError(
                    f"Incompatible shape for block at slot {slot}:"
                    f" expected slot shape {shape}, got content shape {content.shape}."
                )

    @classmethod
    def _new(cls, wiring: Wiring, blocks: tuple[Block | None, ...]) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__wiring = wiring
        self.__blocks = blocks
        return self

    __wiring: Wiring
    __blocks: tuple[Box | Diagram | None, ...]

    __hash_cache: int

    __slots__ = ("__weakref__", "__wiring", "__blocks", "__hash_cache")

    def __new__(cls, wiring: Wiring, blocks: SlotContents[Block]) -> Self:
        """Constructs a new diagram from a wiring and blocks for (some of) its slots."""
        assert validate(wiring, Wiring)
        assert validate(blocks, SlotContents[Block])  # type: ignore[misc]
        blocks = _slot_contents_to_tuple(wiring.num_slots, blocks)
        Diagram._validate_slot_content_shapes(wiring, blocks)
        return cls._new(wiring, blocks)

    @property
    def wiring(self) -> Wiring:
        """Wiring for the diagram."""
        return self.__wiring

    @property
    def blocks(self) -> tuple[Block | None, ...]:
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
    def open_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which are open in the diagram."""
        return tuple(slot for slot, block in enumerate(self.blocks) if block is None)

    @property
    def subdiagram_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Diagram)
        )

    @property
    def subdiagrams(self) -> tuple[Diagram, ...]:
        """Diagrams associated to the slots in :prop:`subdiagram_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Diagram))

    @property
    def box_slots(self) -> tuple[Slot, ...]:
        """Slots of the diagram wiring which have a diagram as a block."""
        return tuple(
            slot for slot, block in enumerate(self.blocks) if isinstance(block, Box)
        )

    @property
    def boxes(self) -> tuple[Box, ...]:
        """Boxes associated to the slots in :prop:`box_slots`."""
        return tuple(block for block in self.blocks if isinstance(block, Box))

    def compose(self, new_blocks: SlotContents[Block | Wiring]) -> Diagram:
        """
        Composes this wiring with the given boxes, diagrams and/or wirings
        for (some of) its slots.
        """
        assert validate(new_blocks, SlotContents[Block | Wiring])  # type: ignore[misc]
        curr_wiring = self.wiring
        new_blocks = _slot_contents_to_tuple(curr_wiring.num_slots, new_blocks)
        Diagram._validate_slot_content_shapes(curr_wiring, new_blocks)
        curr_blocks = self.blocks
        for slot, (curr_block, new_block) in enumerate(zip(curr_blocks, new_blocks)):
            if curr_block is not None and new_block is not None:
                if curr_block is not new_block:
                    raise ValueError(f"Slot {slot} is already occupied.")
        merged_wiring = curr_wiring.compose(
            tuple(block if isinstance(block, Wiring) else None for block in new_blocks)
        )
        merged_blocks = tuple(
            (
                curr_block
                if curr_block is not None
                else new_block if isinstance(new_block, (Box, Diagram)) else None
            )
            for curr_block, new_block in zip(curr_blocks, new_blocks)
        )
        return Diagram._new(merged_wiring, merged_blocks)

    def flatten(self) -> Diagram:
        """
        Returns a recursively diagram, obtained by recursively flattening all
        sub-diagrams, composing their wirings into the current wiring, and taking
        all blocks (of this diagrams and its sub-diagrams) as the blocks of the result.
        """
        return self._flatten({})

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
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Diagram):
            return NotImplemented
        return self.wiring == other.wiring and self.blocks == other.blocks

    def __hash__(self) -> int:
        try:
            return self.__hash_cache
        except AttributeError:
            self.__hash_cache = h = hash((Diagram, self.wiring, self.blocks))
            return h
