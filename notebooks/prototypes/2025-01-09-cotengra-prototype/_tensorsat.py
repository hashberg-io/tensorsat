"""
A temporary minimal implementation of tensorsat, for prototyping purposes.
"""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import prod
from types import MappingProxyType
from typing import Any, Self, SupportsIndex, TypeAlias, TypedDict, overload
import numpy as np
import numpy.typing as npt

BoolTensor: TypeAlias = npt.NDArray[np.uint8]
"""Type alias for a boolean tensor, currently implemented as a NumPy uint8 ndarray."""

ComponentIndex: TypeAlias = int
"""Type alias for the index of a component in a relation."""

Dim: TypeAlias = int
"""Type alias for a dimension in a shape."""

El: TypeAlias = int
"""Type alias for an individual element in a component set of a relation."""


class Shape(tuple[Dim, ...]):
    """A shape, defined as a tuple of of positive dimensions."""

    def __new__(cls, dims: Iterable[Dim]) -> Self:
        self = super().__new__(cls, dims)
        if any(dim <= 0 for dim in self):
            raise ValueError("Dimensions in a shape must be positive.")
        return self

    @overload  # type: ignore [override]
    def __add__(self, value: Shape, /) -> Shape: ...
    @overload
    def __add__(self, value: tuple[Dim, ...], /) -> tuple[Dim, ...]: ...
    @overload
    def __add__[_T](self, value: tuple[_T, ...], /) -> tuple[Dim | _T, ...]: ...
    def __add__[_T](self, value: tuple[_T, ...], /) -> tuple[Dim | _T, ...]:
        if isinstance(value, Shape):
            return Shape(super().__add__(value))
        return super().__add__(value)

    def __mul__(self, value: SupportsIndex, /) -> Shape:
        return Shape(super().__mul__(value))

    def __rmul__(self, value: SupportsIndex, /) -> Shape:
        return Shape(super().__rmul__(value))

    def __repr__(self) -> str:
        return f"Shape({super().__repr__()})"


class Rel:
    """A relation, defined by a Boolean tensor."""

    @classmethod
    def from_subset(
        cls, shape: Sequence[Dim], subset: Iterable[El | tuple[El, ...]]
    ) -> Self:
        """Constructs a relation from a set of tuples."""
        shape = Shape(shape)
        tensor = np.zeros(shape, dtype=np.uint8)
        for idx in subset:
            if isinstance(idx, El):
                idx = (idx,)
            if len(idx) != len(shape):
                raise ValueError(f"Tuple length for {idx} invalid for shape {shape}.")
            if not all(0 <= i < s for i, s in zip(idx, shape)):
                raise ValueError(f"Tuple {idx} invalid for shape {shape}.")
            tensor[idx] = 1
        return cls._new(tensor)

    @classmethod
    def from_mapping(
        cls,
        input_shape: Sequence[Dim],
        output_shape: Sequence[Dim],
        mapping: Mapping[tuple[El, ...], El | tuple[El, ...]],
    ) -> Self:
        """
        Constructs a function graph from a mapping of tuples to tuples.
        The relation shape is given by ``input_shape + output_shape``.
        """
        input_shape = Shape(input_shape)
        output_shape = Shape(output_shape)
        _mapping: dict[tuple[El, ...], tuple[El, ...]] = {
            k: (v,) if isinstance(v, int) else v for k, v in mapping.items()
        }
        rel = cls.from_subset(
            input_shape + output_shape, (k + v for k, v in _mapping.items())
        )
        if len(_mapping) != prod(input_shape):
            raise ValueError("Mapping does not cover the entire input space.")
        return rel

    @classmethod
    def singleton(cls, shape: Sequence[Dim], value: El | tuple[El, ...]) -> Self:
        """Constructs a singleton relation with the given value."""
        shape = Shape(shape)
        return cls.from_mapping(Shape(()), shape, {(): value})

    @classmethod
    def from_callable(
        cls,
        input_shape: Sequence[Dim],
        output_shape: Sequence[Dim],
        func: Callable[[tuple[El, ...]], El | tuple[El, ...]],
    ) -> Self:
        """
        Constructs a function graph from a callable.
        The relation shape is given by ``input_shape + output_shape``.
        """
        input_shape = Shape(input_shape)
        output_shape = Shape(output_shape)
        mapping = {idx: func(idx) for idx in np.ndindex(input_shape)}
        return cls.from_mapping(input_shape, output_shape, mapping)

    @classmethod
    def _new(
        cls,
        tensor: BoolTensor,
    ) -> Self:
        """
        Protected constructor.
        Presumes that the tensor is already validated,
        and that the tensor is not going to be accessible anywhere else.
        """
        assert tensor.dtype == np.uint8
        tensor.setflags(write=False)
        tensor = tensor.view()
        self = super().__new__(cls)
        self.__tensor = tensor
        self.__shape = Shape(tensor.shape)
        return self

    __tensor: BoolTensor
    __shape: Shape
    __hash: int

    def __new__(cls, tensor: BoolTensor) -> Self:
        """Constructs a relation from a Boolean tensor."""
        tensor = np.sign(tensor, dtype=np.uint8)
        return cls._new(tensor)

    @property
    def tensor(self) -> BoolTensor:
        """The Boolean tensor defining the relation."""
        return self.__tensor

    @property
    def shape(self) -> Shape:
        """The shape of the relation."""
        return self.__shape

    def matrix(self, num_inputs: int, /) -> BoolTensor:
        """
        The matrix representation of the relation, with the given number of inputs.
        By convention, inputs are selected as the first components,
        so that input tuples index the rows of the matrix.
        """
        shape = self.shape
        if not 0 <= num_inputs < len(shape):
            raise ValueError("Invalid number of inputs.")
        input_dim = prod(shape[:num_inputs])
        output_dim = prod(shape[num_inputs:])
        return self.tensor.reshape(input_dim, output_dim)

    def is_function_graph(self, num_inputs: int, /) -> bool:
        """
        Whether the relation is a function graph with the given number of inputs.
        By convention, inputs for a function graph are listed as the first components.
        See :meth:`Rel.matrix`.
        """
        return bool(np.all(np.count_nonzero(self.matrix(num_inputs), axis=1) == 1))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Rel):
            return NotImplemented
        return (
            bool((self.__tensor == other.__tensor).all()) and self.shape == other.shape
        )

    def __hash__(self) -> int:
        try:
            return self.__hash
        except AttributeError:
            self.__hash = h = hash((Rel, bytes(self.__tensor)))
            return h


SlotIndex: TypeAlias = int
"""Type alias for the index of an input slot in a wiring diagram."""

InputIndex: TypeAlias = tuple[SlotIndex, int]
"""
Type alias for the index of an input component in a wiring diagram,
as a pair of the slot index and the input index within the slot.
"""

OutputIndex: TypeAlias = int
"""Type alias for the index of an output component in a wiring diagram."""

WiringNodeIndex: TypeAlias = int
"""Type alias for the index of a node in a wiring diagram."""


class WiringDiagramData(TypedDict, total=True):
    """Data for a wiring diagram."""

    slot_num_inputs: Sequence[int]
    """Number of inputs for each slot."""

    num_outputs: int
    """Number of outputs."""

    wiring_node_dims: Sequence[Dim]
    """Dimensions of the wiring nodes."""

    input_wiring: Mapping[InputIndex, WiringNodeIndex]
    """Mapping of input components to wiring nodes."""

    output_wiring: Mapping[OutputIndex, WiringNodeIndex]
    """Mapping of output components to wiring nodes."""


class Wiring:
    """A wiring diagram."""

    @classmethod
    def _new(
        cls,
        input_shapes: tuple[Shape, ...],
        output_shape: Shape,
        wiring_node_dims: Shape,
        input_wiring: MappingProxyType[InputIndex, WiringNodeIndex],
        output_wiring: MappingProxyType[OutputIndex, WiringNodeIndex],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__input_shapes = input_shapes
        self.__output_shape = output_shape
        self.__wiring_node_dims = wiring_node_dims
        self.__input_wiring = input_wiring
        self.__output_wiring = output_wiring
        return self

    __input_shapes: tuple[Shape, ...]
    __output_shape: Shape
    __wiring_node_dims: Shape
    __input_wiring: Mapping[InputIndex, WiringNodeIndex]
    __output_wiring: Mapping[OutputIndex, WiringNodeIndex]

    def __new__(cls, data: WiringDiagramData) -> Self:
        """Constructs a wiring diagram from the given data."""
        # Destructure the data:
        slot_num_inputs = tuple(data["slot_num_inputs"])
        num_outputs = data["num_outputs"]
        wiring_node_dims = Shape(data["wiring_node_dims"])
        input_wiring = MappingProxyType(data["input_wiring"])
        output_wiring = MappingProxyType(data["output_wiring"])
        # Validate the data:
        num_wiring_nodes = len(wiring_node_dims)
        if set(input_wiring.keys()) != {
            (k, i) for k, num_in in enumerate(slot_num_inputs) for i in range(num_in)
        }:
            raise ValueError("Incorrect domain for input wiring.")
        if not all(0 <= w < num_wiring_nodes for w in input_wiring.values()):
            raise ValueError("Incorrect image for input wiring.")
        if set(output_wiring.keys()) != set(range(num_outputs)):
            raise ValueError("Incorrect domain for output wiring.")
        if not all(0 <= w < num_wiring_nodes for w in output_wiring.values()):
            raise ValueError("Incorrect image for output wiring.")
        # Create and return the instance:
        input_shapes = tuple(
            Shape(wiring_node_dims[i] for i in range(num_in))
            for num_in in slot_num_inputs
        )
        output_shape = Shape(wiring_node_dims[o] for o in range(num_outputs))
        return cls._new(
            input_shapes, output_shape, wiring_node_dims, input_wiring, output_wiring
        )

    @property
    def input_shapes(self) -> tuple[Shape, ...]:
        """The input shapes for the wiring diagram."""
        return self.__input_shapes

    @property
    def output_shape(self) -> Shape:
        """The output shape for the wiring diagram."""
        return self.__output_shape

    @property
    def wiring_node_dims(self) -> Shape:
        """The dimensions of the wiring nodes."""
        return self.__wiring_node_dims

    @property
    def input_wiring(self) -> Mapping[InputIndex, WiringNodeIndex]:
        """The input wiring for the wiring diagram."""
        return self.__input_wiring

    @property
    def output_wiring(self) -> Mapping[OutputIndex, WiringNodeIndex]:
        """The output wiring for the wiring diagram."""
        return self.__output_wiring


class WiringBuilder:
    """A builder for wiring diagrams."""

    _input_shapes: list[list[Dim]]
    _output_shape: list[Dim]
    _wiring_node_dims: list[Dim]
    _input_wiring: dict[InputIndex, WiringNodeIndex]
    _output_wiring: dict[OutputIndex, WiringNodeIndex]

    def __new__(cls) -> Self:
        """Constructs a blank wiring diagram builder."""
        self = super().__new__(cls)
        self._input_shapes = []
        self._output_shape = []
        self._wiring_node_dims = []
        self._input_wiring = {}
        self._output_wiring = {}
        return self

    def clone(self) -> WiringBuilder:
        """Clones the wiring diagram builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone._input_shapes = [s.copy() for s in self._input_shapes]
        clone._output_shape = self._output_shape.copy()
        clone._wiring_node_dims = self._wiring_node_dims.copy()
        clone._input_wiring = self._input_wiring.copy()
        clone._output_wiring = self._output_wiring.copy()
        return clone

    @property
    def input_shapes(self) -> tuple[Shape, ...]:
        """The input shapes constructed by the builder thus far."""
        return tuple(Shape(dims) for dims in self._input_shapes)

    @property
    def num_slots(self) -> int:
        """The number of input slots constructed by the builder thus far."""
        return len(self._input_shapes)

    def num_inputs(self, slots: SlotIndex) -> int:
        """The number of inputs in the given slot constructed by the builder thus far."""
        return len(self._input_shapes[slots])

    @property
    def output_shape(self) -> Shape:
        """The output shape constructed by the builder thus far."""
        return Shape(self._output_shape)

    @property
    def num_outputs(self) -> int:
        """The number of outputs constructed by the builder thus far."""
        return len(self._output_shape)

    @property
    def wiring_node_dims(self) -> Shape:
        """The dimensions of the wiring nodes constructed by the builder thus far."""
        return Shape(self._wiring_node_dims)

    @property
    def num_wiring_nodes(self) -> int:
        """The number of wiring nodes constructed by the builder thus far."""
        return len(self._wiring_node_dims)

    @property
    def input_wiring(self) -> MappingProxyType[InputIndex, WiringNodeIndex]:
        """The input wiring constructed by the builder thus far."""
        return MappingProxyType(self._input_wiring)

    @property
    def output_wiring(self) -> MappingProxyType[OutputIndex, WiringNodeIndex]:
        """The output wiring constructed by the builder thus far."""
        return MappingProxyType(self._output_wiring)

    @property
    def wiring_diagram(self) -> Wiring:
        """The wiring diagram constructed by the builder thus far."""
        return Wiring._new(
            self.input_shapes,
            self.output_shape,
            self.wiring_node_dims,
            self.input_wiring,
            self.output_wiring,
        )

    def add_node(self, dim: Dim) -> WiringNodeIndex:
        """Adds a new wiring node with the given dimension."""
        if dim <= 0:
            raise ValueError("Wiring node dimension must be positive.")
        return self._add_nodes([dim])[0]

    def add_nodes(self, dims: Sequence[Dim]) -> tuple[WiringNodeIndex, ...]:
        """Adds new wiring nodes with the given dimensions."""
        if any(dim <= 0 for dim in dims):
            raise ValueError("Wiring node dimension must be positive.")
        return self._add_nodes(dims)

    def _add_nodes(self, dims: Sequence[Dim]) -> tuple[WiringNodeIndex, ...]:
        wiring_node_dims = self._wiring_node_dims
        len_before = len(wiring_node_dims)
        wiring_node_dims.extend(dims)
        return tuple(range(len_before, len(wiring_node_dims)))

    def add_output(self, wiring_node: WiringNodeIndex) -> OutputIndex:
        """Adds an output component for the given wiring node."""
        if not 0 <= wiring_node < len(self._wiring_node_dims):
            raise ValueError(f"Invalid wiring node index {wiring_node}.")
        return self._add_outputs([wiring_node])[0]

    def add_outputs(
        self, wiring_nodes: Sequence[WiringNodeIndex]
    ) -> tuple[OutputIndex, ...]:
        """Adds new outputs connected the given wiring nodes."""
        num_wiring_nodes = len(self._wiring_node_dims)
        if not all(0 <= w < num_wiring_nodes for w in wiring_nodes):
            raise ValueError("Invalid wiring node index.")
        return self._add_outputs(wiring_nodes)

    def _add_outputs(
        self, wiring_nodes: Sequence[WiringNodeIndex]
    ) -> tuple[OutputIndex, ...]:
        output_shape, wiring_node_dims = self._output_shape, self._wiring_node_dims
        len_before = len(output_shape)
        output_shape.extend(wiring_node_dims[w] for w in wiring_nodes)
        new_outputs = tuple(range(len_before, len(output_shape)))
        self._output_wiring.update(zip(new_outputs, wiring_nodes))
        return new_outputs

    @overload
    def add_slot(self) -> SlotIndex: ...
    @overload
    def add_slot(
        self, wiring_nodes: Sequence[WiringNodeIndex]
    ) -> tuple[SlotIndex, tuple[InputIndex, ...]]: ...
    def add_slot(
        self, wiring_nodes: Sequence[WiringNodeIndex] | None = None
    ) -> SlotIndex | tuple[SlotIndex, tuple[InputIndex, ...]]:
        """Adds a new input slot to the wiring diagram."""
        k = len(self._input_shapes)
        self._input_shapes.append([])
        if wiring_nodes is None:
            return k
        try:
            new_inputs = self.add_inputs(k, wiring_nodes)
        except ValueError:
            self._input_shapes.pop()
            raise
        return k, new_inputs

    def add_input(self, slot: SlotIndex, wired_to: WiringNodeIndex) -> InputIndex:
        """Adds a new input for the given slot, connected the given wiring node."""
        return self.add_inputs(slot, (wired_to,))[0]

    def add_inputs(
        self, slot: SlotIndex, wiring_nodes: Sequence[WiringNodeIndex]
    ) -> tuple[InputIndex, ...]:
        """Adds new inputs for the given slot, connected the given wiring nodes."""
        if not 0 <= slot < len(self._input_shapes):
            raise ValueError(f"Invalid input slot index {slot}.")
        if not all(0 <= w < len(self._wiring_node_dims) for w in wiring_nodes):
            raise ValueError("Invalid wiring node index.")
        return self._add_inputs(slot, wiring_nodes)

    def _add_inputs(
        self, slot: SlotIndex, wiring_nodes: Sequence[WiringNodeIndex]
    ) -> tuple[InputIndex, ...]:
        input_shape, wiring_node_dims = self._input_shapes[slot], self._wiring_node_dims
        len_before = len(input_shape)
        input_shape.extend(wiring_node_dims[w] for w in wiring_nodes)
        new_inputs = tuple((slot, i) for i in range(len_before, len(input_shape)))
        self._input_wiring.update(zip(new_inputs, wiring_nodes))
        return new_inputs


class RelNet:
    """A relation network."""

    __wiring: Wiring
    __rels: tuple[Rel, ...]

    def __new__(cls, wiring: Wiring, rels: Iterable[Rel]) -> Self:
        """Constructs a relation network from the given wiring and relations."""
        rels = tuple(rels)
        if tuple(wiring.input_shapes) != tuple(rel.shape for rel in rels):
            raise ValueError("Mismatch between relations and input shapes for wiring.")
        self = super().__new__(cls)
        self.__wiring = wiring
        self.__rels = tuple(rels)
        return self

    @property
    def wiring(self) -> Wiring:
        """The wiring diagram for the network."""
        return self.__wiring

    @property
    def rels(self) -> tuple[Rel, ...]:
        """The relations in the network."""
        return self.__rels


ComponentsToWiringNodes: TypeAlias = Sequence[int | None] | Mapping[int, int]
"""
Type alias for a mapping of components in a relation to wiring nodes.
Can be a sequence of indices, with None for missing components,
or a mapping of component indices to wiring node indices.
"""

GateData: TypeAlias = tuple[Rel, tuple[int | None, ...], tuple[int, ...]]
"""
Type alias for the data of a gate in a circuit, consisting of:

- a relationa
- a sequence of wiring node indices to which it was applied
- a sequence of wiring node indices for its outputs

"""


class CircuitBuilder:
    """A builder for relation networks using circuit-like construction syle."""

    _gates: list[GateData]
    _builder: WiringBuilder
    _inputs: list[OutputIndex]
    _outputs: list[OutputIndex]

    def __new__(cls, inputs: Sequence[Dim] | None = None) -> Self:
        """Constructs a blank circuit builder."""
        self = super().__new__(cls)
        self._gates = []
        self._builder = WiringBuilder()
        self._inputs = []
        self._outputs = []
        return self

    def clone(self) -> CircuitBuilder:
        """Clones the circuit builder."""
        clone = CircuitBuilder.__new__(CircuitBuilder)
        clone._gates = self._gates.copy()
        clone._builder = self._builder.clone()
        clone._inputs = self._inputs.copy()
        clone._outputs = self._outputs.copy()
        return clone

    @property
    def network(self) -> RelNet:
        """The relation network constructed by the builder thus far."""
        return RelNet(self._builder.wiring_diagram, [rel for rel, _, _ in self._gates])

    @property
    def gates(self) -> tuple[GateData, ...]:
        """Sequence of gates in the circuit, in order of addition."""
        return tuple(self._gates)

    @property
    def num_gates(self) -> int:
        """Number of gates in the circuit."""
        return len(self._gates)

    @property
    def inputs(self) -> tuple[WiringNodeIndex, ...]:
        """Wiring nodes corresponding to the inputs of the circuit."""
        output_wiring = self._builder._output_wiring
        return tuple(output_wiring[i] for i in self._inputs)

    @property
    def num_inputs(self) -> int:
        """Number of inputs for the circuit."""
        return len(self._inputs)

    @property
    def outputs(self) -> tuple[WiringNodeIndex, ...]:
        """Wiring nodes corresponding to the outputs of the circuit."""
        output_wiring = self._builder._output_wiring
        return tuple(output_wiring[o] for o in self._outputs)

    @property
    def num_outputs(self) -> int:
        """Number of outputs for the circuit."""
        return len(self._outputs)

    def add_gate(
        self, gate: Rel, wiring_nodes: ComponentsToWiringNodes
    ) -> tuple[WiringNodeIndex, ...]:
        """
        Adds the given relation as a gate in the circuit,
        by using the given wiring nodes as its inputs.
        Relation components are matched to wiring nodes in order,
        skipping components where None appears.

        Following a convention where functions graphs have inputs before outputs,
        this means that functions can be applied ordinarily by passing their inputs.
        """
        wiring_nodes = self._validate_wiring_nodes(wiring_nodes)
        return self._add_gate(gate, wiring_nodes)

    def _validate_wiring_nodes(
        self, wiring_nodes: ComponentsToWiringNodes
    ) -> tuple[WiringNodeIndex | None, ...]:
        if isinstance(wiring_nodes, Mapping):
            num_components = max(wiring_nodes) + 1
            _wiring_nodes = tuple(wiring_nodes.get(i) for i in range(num_components))
        else:
            _wiring_nodes = tuple(wiring_nodes)
        num_wiring_nodes = self._builder.num_wiring_nodes
        if not all(0 <= w < num_wiring_nodes for w in _wiring_nodes if w is not None):
            raise ValueError("Invalid wiring node index.")
        return _wiring_nodes

    def _add_gate(
        self, gate: Rel, wiring_nodes: tuple[WiringNodeIndex | None, ...]
    ) -> tuple[WiringNodeIndex, ...]:
        if (_residual := len(gate.shape) - len(wiring_nodes)) >= 0:
            wiring_nodes = wiring_nodes + (None,) * _residual
        else:
            raise ValueError("Too many wiring nodes specified.")
        all_wiring_node_dims = self._builder._wiring_node_dims
        for w, (j, d) in zip(wiring_nodes, enumerate(gate.shape)):
            if w is not None and all_wiring_node_dims[w] != d:
                raise ValueError(f"Incorrect dimension for component {j}.")
        builder = self._builder
        builder.add_slot([w for w in wiring_nodes if w is not None])
        output_nodes = builder.add_nodes(
            [d for w, d in zip(wiring_nodes, gate.shape) if w is None]
        )
        self._gates.append((gate, wiring_nodes, output_nodes))
        return output_nodes

    def add_output(self, wiring_node: WiringNodeIndex) -> None:
        """Adds the given wiring node as an output of the circuit."""
        self.add_outputs((wiring_node,))

    def add_outputs(self, wiring_nodes: Sequence[WiringNodeIndex]) -> None:
        """Adds the given wiring nodes as outputs of the circuit."""
        self._outputs.extend(self._builder._add_outputs(wiring_nodes))

    def add_input(self, dim: Dim) -> WiringNodeIndex:
        """
        Adds an input of the given dimension to the circuit.
        Returns the wiring node corresponding to the input.
        """
        return self.add_inputs((dim,))[0]

    def add_inputs(self, dims: Sequence[Dim]) -> tuple[WiringNodeIndex, ...]:
        """
        Adds inputs of the given dimensions to the circuit.
        Returns the tuple of wiring nodes corresponding to the inputs.
        """
        builder = self._builder
        wiring_nodes = builder._add_nodes(dims)
        self._inputs.extend(builder._add_outputs(wiring_nodes))
        return wiring_nodes

    # def describe(
    #     self,
    #     *,
    #     circ: str = "circ",
    #     gates: Mapping[Rel, str] = MappingProxyType({}),
    #     var_prefix: str = "x",
    # ) -> str:
    #     def gatename(idx: int) -> str:
    #         return gates.get(self._gates[idx][0], f"<gate {idx}>")
    #     def varname(w: WiringNodeIndex | None) -> str:
    #         return f"{var_prefix}{w}" if w is not None else "None"
    #     def varnames(ws: Sequence[WiringNodeIndex | None]) -> str:
    #         if len(ws) == 0:
    #             return "()"
    #         if ws[_cut_idx:=-1] is None:
    #             while ws[_cut_idx] is None:
    #                 _cut_idx -= 1
    #             ws = ws[:_cut_idx+1]
    #         return f"{", ".join(varname(w) for w in ws)}"+("," if len(ws) == 1 else "")
    #     lines: list[str] = []
    #     lines.append(f"{circ} = CircuitBuilder()")
    #     # TODO: inputs have to be added at the correct places in the circuit
    #     for idx, (gate, input_nodes, output_nodes) in enumerate(self._gates):
    #         if output_nodes:
    #             line = f"{varnames(output_nodes)} = "
    #         else:
    #             line = ""
    #         line += f"{gatename(idx)} @ {circ}[{varnames(input_nodes)}]"
    #         lines.append(line)
    #     # TODO: outputs can be added at the end of the circuit
    #     return "\n".join(lines)

    def __getitem__(
        self, wiring_nodes: tuple[WiringNodeIndex | None, ...]
    ) -> _SelectedWiringNodes:
        """
        Returns a wiring node selector for the given wiring nodes,
        to implement EDSL syntax for gate addition.
        """
        return _SelectedWiringNodes(self, wiring_nodes)


class _SelectedWiringNodes:
    """
    Utility class tracking selection of wiring nodes for a circuit builder,
    to implement EDSL syntax for gate addition.
    Example of construction of a half-adder circuit:

    .. code-block:: python

        hadd = CircuitBuilder()
        a, b, c_in = hadd.add_inputs((2,)*3)
        x1, = xor_ @ hadd[a, b]
        x2, = and_ @ hadd[a, b]
        x3, = and_ @ hadd[x1, c_in]
        s, = xor_ @ hadd[x1, x3]
        c_out, = or_ @ hadd[x2, x3]
        hadd.add_outputs((s, c_out))

    """

    _builder: CircuitBuilder
    _wiring_nodes: tuple[WiringNodeIndex | None, ...]

    def __new__(
        cls, builder: CircuitBuilder, wiring_nodes: ComponentsToWiringNodes
    ) -> Self:
        """Constructs a wiring node selector for the given circuit builder."""
        wiring_nodes = builder._validate_wiring_nodes(wiring_nodes)
        self = super().__new__(cls)
        self._builder = builder
        self._wiring_nodes = wiring_nodes
        return self

    def __rmatmul__(self, gate: Rel) -> tuple[WiringNodeIndex, ...]:
        """
        Adds the given relation as a gate in the circuit,
        using the selected wiring nodes as inputs.
        Returns the wiring nodes corresponding to the outputs of the gate.
        """
        return self._builder._add_gate(gate, self._wiring_nodes)
