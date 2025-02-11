"""
Implementation of diagrams and their builders for the :mod:`tensorsat.diagrams` module.
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
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import (
    ClassVar,
    Generic,
    Self,
    cast,
    final,
)
from hashcons import InstanceStore

if __debug__:
    from typing_validation import validate


from .types import Type, Shape, TypeT_co, TypeT_inv
from .wirings import Port, Shaped, Slot, Wire, Wiring, WiringBuilder
from .boxes import Box

type Block[T: Type] = Box[T] | Diagram[T]
"""
Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

"""

# TODO: Consider making diagrams parametric in BoxT_co as well,
#       even though we cannot (yet?) make BoxT_co bound to on Box[TypeT_co]


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
        for a full-adder circuit, by wrapping a recipe using a diagram builder
        internally:

        .. code-block:: python

            from typing import reveal_type
            from tensorsat.lang.rel import bit
            from tensorsat.lib.bincirc import and_, or_, xor_

            @Diagram.from_recipe(bit**3)
            def adder(
                circ: DiagramBuilder[FinSet],
                inputs: Sequence[Wire]
            ) -> Sequence[Wire]:
                a, b, c_in = inputs
                x1, = xor_ @ circ[a, b]
                x2, = and_ @ circ[a, b]
                x3, = and_ @ circ[x1, c_in]
                s, = xor_ @ circ[x1, x3]
                c_out, = or_ @ circ[x2, x3]
                return s, c_out

            reveal_type(adder) # Diagram

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
            def ripple_carry_adder(
                circ: DiagramBuilder[FinSet],
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
                self.__recipe_used = None
                Diagram._store.register(self)
            return self

    __wiring: Wiring[TypeT_co]
    __blocks: tuple[Box[TypeT_co] | Diagram[TypeT_co] | None, ...]
    __recipe_used: DiagramRecipe[TypeT_co] | None

    __slots__ = ("__weakref__", "__wiring", "__blocks", "__recipe_used")

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

    @property
    def recipe_used(self) -> DiagramRecipe[TypeT_co] | None:
        """The recipe used to create the diagram, if any."""
        return self.__recipe_used

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

    def __repr__(self) -> str:
        attrs: list[str] = []
        num_wires = self.wiring.num_wires
        num_open_slots = self.num_open_slots
        num_blocks = len(self.blocks)
        depth = self.depth
        num_out_ports = len(self.wiring.out_wires)
        recipe_used = self.recipe_used
        if num_wires > 0:
            attrs.append(f"{num_wires} wires")
        if num_open_slots > 0:
            attrs.append(f"{num_open_slots} open slots")
        if num_blocks > 0:
            attrs.append(f"{num_blocks} blocks")
        if depth > 0:
            attrs.append(f"depth {depth}")
        if num_out_ports > 0:
            attrs.append(f"{num_out_ports} out ports")
        if recipe_used and recipe_used.name:
            attrs.append(f"from recipe {recipe_used.name!r}")
        return f"<Diagram {id(self):#x}: {", ".join(attrs)}>"


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
        wiring._add_out_ports(wires)
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

    def __repr__(self) -> str:
        return f"<DiagramBuilder {id(self.__builder):#x}>[{self.__wires}]"


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
    __name: str | None

    __slots__ = ("__weakref__", "__recipe", "__name")

    def __new__(
        cls,
        recipe: Callable[[DiagramBuilder[TypeT_inv], tuple[Wire, ...]], Sequence[Wire]],
    ) -> Self:
        """Wraps the given diagram building logic."""
        self = super().__new__(cls)
        self.__recipe = recipe
        self.__name = getattr(recipe, "__name__", None)
        return self

    @property
    def name(self) -> str | None:
        """Name of the recipe, if available."""
        return self.__name

    def __call__(self, input_types: Sequence[TypeT_inv]) -> Diagram[TypeT_inv]:
        """Executes the recipe for the given input types, returning the diagram."""
        builder: DiagramBuilder[TypeT_inv] = DiagramBuilder()
        inputs = builder._add_inputs(input_types)
        outputs = self.__recipe(builder, inputs)
        builder._add_outputs(outputs)
        diagram = builder.diagram
        diagram._Diagram__recipe_used = self  # type: ignore[attr-defined]
        return diagram

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

    def __repr__(self) -> str:
        name = self.__name
        if name is None:
            return f"<DiagramRecipe {id(self):#x}>"
        return f"<DiagramRecipe {id(self):#x} {name!r}>"
