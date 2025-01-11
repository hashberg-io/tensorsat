"""
A temporary minimal implementation of tensorsat, for prototyping purposes.
"""

from __future__ import annotations
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import prod
from types import MappingProxyType
from typing import Any, Protocol, Self, SupportsIndex, TypeAlias, TypedDict, cast, overload
import numpy as np
import numpy.typing as npt
import cotengra as ct # type: ignore[import-untyped]
from cotengra import ContractionTree, HyperGraph

BoolTensor: TypeAlias = npt.NDArray[np.uint8]
"""Type alias for a boolean tensor, currently implemented as a NumPy uint8 ndarray."""

Component: TypeAlias = int
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


Slot: TypeAlias = int
"""Type alias for the index of an inner slot in a wiring diagram."""

SlotPort: TypeAlias = tuple[Slot, int]
"""
Type alias for the index of an input component in a wiring diagram,
as a pair of the slot index and the input index within the slot.
"""

OuterPort: TypeAlias = int
"""Type alias for the index of an output component in a wiring diagram."""

Node: TypeAlias = int
"""Type alias for the index of a node in a wiring diagram."""


class WiringDiagramData(TypedDict, total=True):
    """Data for a wiring diagram."""

    num_slot_ports: Sequence[int]
    """Number of ports for each slot."""

    num_outer_ports: int
    """Number of outer ports."""

    node_dims: Sequence[Dim]
    """Dimensions of the wiring nodes."""

    slot_wiring: Mapping[SlotPort, Node]
    """Mapping of slot ports to wiring nodes."""

    outer_wiring: Mapping[OuterPort, Node]
    """Mapping of output ports to wiring nodes."""


class Wiring:
    """A wiring diagram."""

    @classmethod
    def _new(
        cls,
        slot_shapes: tuple[Shape, ...],
        outer_shape: Shape,
        node_dims: Shape,
        slot_wiring: MappingProxyType[SlotPort, Node],
        outer_wiring: MappingProxyType[OuterPort, Node],
    ) -> Self:
        """Protected constructor."""
        self = super().__new__(cls)
        self.__slot_shapes = slot_shapes
        self.__outer_shape = outer_shape
        self.__node_dims = node_dims
        self.__slot_wiring = slot_wiring
        self.__outer_wiring = outer_wiring
        return self

    __slot_shapes: tuple[Shape, ...]
    __outer_shape: Shape
    __node_dims: Shape
    __slot_wiring: Mapping[SlotPort, Node]
    __outer_wiring: Mapping[OuterPort, Node]

    def __new__(cls, data: WiringDiagramData) -> Self:
        """Constructs a wiring diagram from the given data."""
        # Destructure the data:
        slot_num_inputs = tuple(data["num_slot_ports"])
        num_outputs = data["num_outer_ports"]
        node_dims = Shape(data["node_dims"])
        slot_wiring = MappingProxyType(data["slot_wiring"])
        outer_wiring = MappingProxyType(data["outer_wiring"])
        # Validate the data:
        num_nodes = len(node_dims)
        if set(slot_wiring.keys()) != {
            (k, i) for k, num_in in enumerate(slot_num_inputs) for i in range(num_in)
        }:
            raise ValueError("Incorrect domain for input wiring.")
        if not all(0 <= w < num_nodes for w in slot_wiring.values()):
            raise ValueError("Incorrect image for input wiring.")
        if set(outer_wiring.keys()) != set(range(num_outputs)):
            raise ValueError("Incorrect domain for output wiring.")
        if not all(0 <= w < num_nodes for w in outer_wiring.values()):
            raise ValueError("Incorrect image for output wiring.")
        # Create and return the instance:
        input_shapes = tuple(
            Shape(node_dims[i] for i in range(num_in))
            for num_in in slot_num_inputs
        )
        output_shape = Shape(node_dims[o] for o in range(num_outputs))
        return cls._new(
            input_shapes, output_shape, node_dims, slot_wiring, outer_wiring
        )

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        """Shapes for the slots."""
        return self.__slot_shapes

    @property
    def num_slots(self) -> int:
        """Number of inner slots."""
        return len(self.__slot_shapes)

    @property
    def slots(self) -> tuple[Slot, ...]:
        """Indices of the inner slots."""
        return tuple(range(self.num_slots))

    def num_slot_ports(self, slot: Slot) -> int:
        """Number of ports for the given slot."""
        return len(self.__slot_shapes[slot])

    def slot_ports(self, slot: Slot) -> tuple[Node, ...]:
        """Tuple of wiring nodes to which ports for the given slot are connected."""
        num_inputs = self.num_slot_ports(slot)
        slot_wiring = self.__slot_wiring
        return tuple(slot_wiring[slot, i] for i in range(num_inputs))

    @property
    def outer_shape(self) -> Shape:
        """Outer shape."""
        return self.__outer_shape

    @property
    def num_outer_ports(self) -> int:
        """Number of outer ports."""
        return len(self.__outer_shape)

    @property
    def outer_ports(self) -> tuple[Node, ...]:
        """Tuple of wiring nodes to which outer ports are connected."""
        outer_wiring = self.__outer_wiring
        return tuple(outer_wiring[o] for o in range(self.num_outer_ports))

    @property
    def node_dims(self) -> Shape:
        """Dimensions of the wiring nodes."""
        return self.__node_dims

    @property
    def num_nodes(self) -> int:
        """Number of wiring nodes."""
        return len(self.__node_dims)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Indices of the wiring nodes."""
        return tuple(range(self.num_nodes))

    @property
    def slot_wiring(self) -> Mapping[SlotPort, Node]:
        """Wiring of slot ports to nodes."""
        return self.__slot_wiring

    @property
    def outer_wiring(self) -> Mapping[OuterPort, Node]:
        """Wiring of outer ports to nodes."""
        return self.__outer_wiring

    @property
    def contraction_tree(self) -> ContractionTree:
        """The cotengra contraction tree for this wiring."""
        return ct.array_contract_tree(
            [self.slot_ports(slot) for slot in self.slots],
            self.outer_ports,
            shapes=self.slot_shapes
        )

    @property
    def hypergraph(self) -> HyperGraph:
        return ct.hypergraph.get_hypergraph(
            [list(map(str, self.slot_ports(slot))) for slot in self.slots],
            list(map(str, self.outer_ports)),
            size_dict={str(w): dim for w, dim in enumerate(self.node_dims)},
        )


class WiringBuilder:
    """A builder for wiring diagrams."""

    _slot_shapes: list[list[Dim]]
    _outer_shape: list[Dim]
    _node_dims: list[Dim]
    _slot_wiring: dict[SlotPort, Node]
    _outer_wiring: dict[OuterPort, Node]

    def __new__(cls) -> Self:
        """Constructs a blank wiring diagram builder."""
        self = super().__new__(cls)
        self._slot_shapes = []
        self._outer_shape = []
        self._node_dims = []
        self._slot_wiring = {}
        self._outer_wiring = {}
        return self

    def clone(self) -> WiringBuilder:
        """Clones the wiring diagram builder."""
        clone = WiringBuilder.__new__(WiringBuilder)
        clone._slot_shapes = [s.copy() for s in self._slot_shapes]
        clone._outer_shape = self._outer_shape.copy()
        clone._node_dims = self._node_dims.copy()
        clone._slot_wiring = self._slot_wiring.copy()
        clone._outer_wiring = self._outer_wiring.copy()
        return clone

    @property
    def slot_shapes(self) -> tuple[Shape, ...]:
        """Shapes for the inner slots."""
        return tuple(Shape(dims) for dims in self._slot_shapes)

    @property
    def num_slots(self) -> int:
        """Number of inner slots."""
        return len(self._slot_shapes)

    @property
    def slots(self) -> tuple[Slot, ...]:
        """Indices of the inner slots."""
        return tuple(range(self.num_slots))

    def num_slot_ports(self, slot: Slot) -> int:
        """Number of ports for the given slot."""
        return len(self._slot_shapes[slot])

    def slot_ports(self, slot: Slot) -> tuple[Node, ...]:
        """Tuple of wiring nodes to which ports for the given slot are connected."""
        num_inputs = self.num_slot_ports(slot)
        slot_wiring = self._slot_wiring
        return tuple(slot_wiring[slot, i] for i in range(num_inputs))

    @property
    def outer_shape(self) -> Shape:
        """Outer shape of the wiring."""
        return Shape(self._outer_shape)

    @property
    def num_outer_ports(self) -> int:
        """Number of outer ports for the wiring result."""
        return len(self._outer_shape)

    @property
    def outer_ports(self) -> tuple[Node, ...]:
        """Tuple of wiring nodes to which outer ports of the wiring are connected."""
        return tuple(range(self.num_outer_ports))

    @property
    def node_dims(self) -> Shape:
        """Dimensions of the wiring nodes."""
        return Shape(self._node_dims)

    @property
    def num_nodes(self) -> int:
        """Number of wiring nodes."""
        return len(self._node_dims)

    @property
    def slot_wiring(self) -> MappingProxyType[SlotPort, Node]:
        """Wiring of slot ports to nodes."""
        return MappingProxyType(self._slot_wiring)

    @property
    def outer_wiring(self) -> MappingProxyType[OuterPort, Node]:
        """Wiring of outer ports to node."""
        return MappingProxyType(self._outer_wiring)

    @property
    def wiring(self) -> Wiring:
        """Wiring diagram constructed by the builder."""
        return Wiring._new(
            self.slot_shapes,
            self.outer_shape,
            self.node_dims,
            self.slot_wiring,
            self.outer_wiring,
        )

    def add_node(self, dim: Dim) -> Node:
        """Adds a new wiring node with the given dimension."""
        if dim <= 0:
            raise ValueError("Wiring node dimension must be positive.")
        return self._add_nodes([dim])[0]

    def add_nodes(self, dims: Sequence[Dim]) -> tuple[Node, ...]:
        """Adds new wiring nodes with the given dimensions."""
        if any(dim <= 0 for dim in dims):
            raise ValueError("Wiring node dimension must be positive.")
        return self._add_nodes(dims)

    def _add_nodes(self, dims: Sequence[Dim]) -> tuple[Node, ...]:
        node_dims = self._node_dims
        len_before = len(node_dims)
        node_dims.extend(dims)
        return tuple(range(len_before, len(node_dims)))

    def add_outer_port(self, node: Node) -> OuterPort:
        """Adds an outer ports component for the given wiring node."""
        if not 0 <= node < len(self._node_dims):
            raise ValueError(f"Invalid wiring node index {node}.")
        return self._add_outer_ports([node])[0]

    def add_outer_ports(
        self, nodes: Sequence[Node]
    ) -> tuple[OuterPort, ...]:
        """Adds new outer ports connected the given wiring nodes."""
        num_nodes = len(self._node_dims)
        if not all(0 <= w < num_nodes for w in nodes):
            raise ValueError("Invalid wiring node index.")
        return self._add_outer_ports(nodes)

    def _add_outer_ports(
        self, nodes: Sequence[Node]
    ) -> tuple[OuterPort, ...]:
        output_shape, node_dims = self._outer_shape, self._node_dims
        len_before = len(output_shape)
        output_shape.extend(node_dims[w] for w in nodes)
        new_outputs = tuple(range(len_before, len(output_shape)))
        self._outer_wiring.update(zip(new_outputs, nodes))
        return new_outputs

    @overload
    def add_slot(self) -> Slot: ...
    @overload
    def add_slot(
        self, nodes: Sequence[Node]
    ) -> tuple[Slot, tuple[SlotPort, ...]]: ...
    def add_slot(
        self, nodes: Sequence[Node] | None = None
    ) -> Slot | tuple[Slot, tuple[SlotPort, ...]]:
        """Adds a new inner slot to the wiring diagram."""
        k = len(self._slot_shapes)
        self._slot_shapes.append([])
        if nodes is None:
            return k
        try:
            new_inputs = self.add_slot_ports(k, nodes)
        except ValueError:
            self._slot_shapes.pop()
            raise
        return k, new_inputs

    def add_slot_port(self, slot: Slot, wired_to: Node) -> SlotPort:
        """Adds a new port for the given slot, connected the given wiring node."""
        return self.add_slot_ports(slot, (wired_to,))[0]

    def add_slot_ports(
        self, slot: Slot, nodes: Sequence[Node]
    ) -> tuple[SlotPort, ...]:
        """Adds new ports for the given slot, connected the given wiring nodes."""
        if not 0 <= slot < len(self._slot_shapes):
            raise ValueError(f"Invalid inner slot index {slot}.")
        if not all(0 <= w < len(self._node_dims) for w in nodes):
            raise ValueError("Invalid wiring node index.")
        return self._add_slot_ports(slot, nodes)

    def _add_slot_ports(
        self, slot: Slot, nodes: Sequence[Node]
    ) -> tuple[SlotPort, ...]:
        input_shape, node_dims = self._slot_shapes[slot], self._node_dims
        len_before = len(input_shape)
        input_shape.extend(node_dims[w] for w in nodes)
        new_inputs = tuple((slot, i) for i in range(len_before, len(input_shape)))
        self._slot_wiring.update(zip(new_inputs, nodes))
        return new_inputs


class RelNet:
    """A relation network."""

    __wiring: Wiring
    __rels: tuple[Rel, ...]

    def __new__(cls, wiring: Wiring, rels: Iterable[Rel]) -> Self:
        """Constructs a relation network from the given wiring and relations."""
        rels = tuple(rels)
        if tuple(wiring.slot_shapes) != tuple(rel.shape for rel in rels):
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

- a relation
- a sequence of wiring nodes to which it was applied
- a sequence of wiring nodes for its outputs

"""


class CircuitBuilder:
    """A builder for relation networks using circuit-like construction syle."""

    _gates: list[GateData]
    _builder: WiringBuilder
    _input_shape: Shape
    _outputs: list[OuterPort]

    def __new__(cls, input_shape: Sequence[Dim]) -> Self:
        """Constructs a blank circuit builder."""
        self = super().__new__(cls)
        self._gates = []
        self._builder = builder = WiringBuilder()
        self._input_shape = input_shape = Shape(input_shape)
        self._outputs = []
        builder._add_outer_ports(builder._add_nodes(input_shape))
        return self

    def clone(self) -> CircuitBuilder:
        """Clones the circuit builder."""
        clone = CircuitBuilder.__new__(CircuitBuilder, self._input_shape)
        clone._gates = self._gates.copy()
        clone._builder = self._builder.clone()
        clone._input_shape = self._input_shape
        clone._outputs = self._outputs
        return clone

    @property
    def gates(self) -> tuple[GateData, ...]:
        """Sequence of gates in the circuit."""
        return tuple(self._gates)

    @property
    def num_gates(self) -> int:
        """Number of gates in the circuit."""
        return len(self._gates)

    @property
    def inputs(self) -> tuple[Node, ...]:
        """Wiring nodes corresponding to the inputs of the circuit."""
        return tuple(range(len(self._input_shape)))

    @property
    def num_inputs(self) -> int:
        """Number of inputs for the circuit."""
        return len(self._input_shape)

    @property
    def outputs(self) -> tuple[Node, ...]:
        """Wiring nodes corresponding to the outputs of the circuit."""
        return self._builder.outer_ports[self.num_inputs:]

    @property
    def network(self) -> RelNet:
        """Relational network for the circuit."""
        return RelNet(self._builder.wiring, [rel for rel, _, _ in self._gates])

    def add_gate(
        self, gate: Rel, nodes: ComponentsToWiringNodes
    ) -> tuple[Node, ...]:
        """
        Adds the given relation as a gate in the circuit,
        by using the given wiring nodes as its inputs.
        Relation components are matched to wiring nodes in order,
        skipping components where None appears.

        Following a convention where functions graphs have inputs before outputs,
        this means that functions can be applied ordinarily by passing their inputs.
        """
        nodes = self._validate_nodes(nodes)
        return self._add_gate(gate, nodes)

    def _validate_nodes(
        self, nodes: ComponentsToWiringNodes
    ) -> tuple[Node | None, ...]:
        if isinstance(nodes, Mapping):
            num_components = max(nodes) + 1
            _nodes = tuple(nodes.get(i) for i in range(num_components))
        else:
            _nodes = tuple(nodes)
        num_nodes = self._builder.num_nodes
        if not all(0 <= w < num_nodes for w in _nodes if w is not None):
            raise ValueError("Invalid wiring node index.")
        return _nodes

    def _add_gate(
        self, gate: Rel, nodes: tuple[Node | None, ...]
    ) -> tuple[Node, ...]:
        if (_residual := len(gate.shape) - len(nodes)) >= 0:
            nodes = nodes + (None,) * _residual
        else:
            raise ValueError("Too many wiring nodes specified.")
        all_node_dims = self._builder._node_dims
        for w, (j, d) in zip(nodes, enumerate(gate.shape)):
            if w is not None and all_node_dims[w] != d:
                raise ValueError(f"Incorrect dimension for component {j}.")
        builder = self._builder
        output_nodes = builder.add_nodes(
            [d for w, d in zip(nodes, gate.shape) if w is None]
        )
        slot_notes: list[Node] = []
        nodes_q = deque(nodes)
        output_nodes_q = deque(output_nodes)
        while nodes_q:
            node = nodes_q.popleft()
            if node is None:
                slot_notes.append(output_nodes_q.popleft())
            else:
                slot_notes.append(node)
        builder.add_slot(slot_notes)
        self._gates.append((gate, nodes, output_nodes))
        return output_nodes

    def add_outputs(self, outputs: Sequence[Node]) -> None:
        """Adds the given wiring nodes as outputs of the circuit."""
        self._builder.add_outer_ports(outputs)

    def __getitem__(
        self, nodes: tuple[Node | None, ...]
    ) -> _SelectedWiringNodes:
        """
        Returns a wiring node selector for the given wiring nodes,
        to implement EDSL syntax for gate addition.
        """
        return _SelectedWiringNodes(self, nodes)


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
    _nodes: tuple[Node | None, ...]

    def __new__(
        cls, builder: CircuitBuilder, nodes: ComponentsToWiringNodes
    ) -> Self:
        """Constructs a wiring node selector for the given circuit builder."""
        nodes = builder._validate_nodes(nodes)
        self = super().__new__(cls)
        self._builder = builder
        self._nodes = nodes
        return self

    def __rmatmul__(self, gate: Rel) -> tuple[Node, ...]:
        """
        Adds the given relation as a gate in the circuit,
        using the selected wiring nodes as inputs.
        Returns the wiring nodes corresponding to the outputs of the gate.
        """
        return self._builder._add_gate(gate, self._nodes)


class CircuitApplicable(Protocol):
    """Structural type for objects which can be applied to selected wiring nodes."""

    def __matmul__(self, selected: _SelectedWiringNodes, /) -> tuple[int, ...]:
        """Logic executed when the object is applied to selected wiring nodes."""


class circuit_applicable(CircuitApplicable):
    """
    Function decorator, to define custom logic which is triggered upon application
    to selected wiring nodes in a circuit, analogously to relations.
    Can be used, for example, to dynamically extend a given circuit via Python code.
    """

    _apply: Callable[[CircuitBuilder, tuple[int, ...]], tuple[int, ...]]

    def __new__(
        cls, apply: Callable[[CircuitBuilder, tuple[int, ...]], tuple[int, ...]], /
    ) -> Self:
        """Constructor for the decorator."""
        self = super().__new__(cls)
        self._apply = apply
        return self

    def __matmul__(self, selected: _SelectedWiringNodes) -> tuple[int, ...]:
        """Executes the decorated function when applied to selected wiring nodes."""
        builder = selected._builder
        nodes = selected._nodes
        if any(w is None for w in nodes):
            raise ValueError("All inputs to a circuit extender must be passed.")
        return self._apply(builder, cast(tuple[int, ...], nodes))
