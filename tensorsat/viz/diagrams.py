"""
Visualisation utilities for diagrams.

.. warning::

    This module is subject to frequent breaking changes.

"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from collections import deque
from collections.abc import Sequence
from types import MappingProxyType
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    TypedDict,
    Unpack,
    cast,
    overload,
)
from numpy.typing import ArrayLike

from ..diagrams import PortOrderStructure, Slot, Box, Diagram, Wire, Port, DiagramRecipe
from .._utils.misc import (
    ValueSetter,
    apply_setter,
    dict_deep_copy,
    dict_deep_update,
)

try:
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.axes import Axes
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For diagram visualisation, 'matplotlib' must be installed."
    )

try:
    import networkx as nx  # type: ignore
    from networkx import MultiGraph
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For diagram visualisation, 'networkx' must be installed."
    )

if __debug__:
    from typing_validation import validate

DiagramGraphNodeKind: TypeAlias = Literal["box", "hole", "diagram", "port", "wire"]
"""Type alias for possible kinds of nodes in the NetworkX graph for a diagram."""

DIAGRAM_GRAPH_NODE_KIND: Final[tuple[DiagramGraphNodeKind, ...]] = (
    "box",
    "hole",
    "diagram",
    "port",
    "wire",
)
"""Possible kinds of nodes in the NetworkX graph for a diagram."""


DiagramGraphNode: TypeAlias = (
    tuple[Literal["box"], int, Box]  # ("box", slot, box)
    | tuple[Literal["hole"], int, None]  # ("hole", slot, None)
    | tuple[Literal["diagram"], int, Diagram]  # ("diagram", slot, diagram)
    | tuple[Literal["port"], int, None]  # ("port", port, None)
    | tuple[Literal["wire"], int, None]  # ("wire", wire, None)
)
"""
Type alias for a node in the NetworkX graph representing a diagram,
labelled by the triple ``(kind, index, data)``.
"""


def diagram_to_nx_graph(
    diagram: Diagram, *, simplify_wires: bool = False
) -> MultiGraph:
    """Utility function converting a diagram to a NetworkX graph."""
    assert validate(diagram, Diagram)
    box_slots = set(diagram.box_slots)
    open_slots = set(diagram.open_slots)
    subdiagram_slots = set(diagram.subdiagram_slots)
    blocks = diagram.blocks

    def slot_node(slot: int) -> DiagramGraphNode:
        """Utility function generating the node label for a given diagram slot."""
        if slot in box_slots:
            box = blocks[slot]
            assert isinstance(box, Box)
            return ("box", slot, box)
        if slot in open_slots:
            return ("hole", slot, None)
        if slot in subdiagram_slots:
            subdiagram = blocks[slot]
            assert isinstance(subdiagram, Diagram)
            return ("diagram", slot, subdiagram)
        assert False, "Slot must be open, filled with a box, or filled with a diagram."

    wiring = diagram.wiring
    wired_slots = wiring.wired_slots
    out_wires = wiring.out_wires
    if simplify_wires:
        slot_slot_wires = {
            w: w_slots
            for w, w_slots in wired_slots.items()
            if len(w_slots) == 2 and w not in out_wires
        }
        slot_port_wires = {
            w: (w_slots[0], out_wires.index(w))
            for w, w_slots in wired_slots.items()
            if len(w_slots) == 1 and out_wires.count(w) == 1
        }
        port_port_wires = {
            w: (_i := out_wires.index(w), out_wires.index(w, _i + 1))
            for w, w_slots in wired_slots.items()
            if len(w_slots) == 0 and out_wires.count(w) == 2
        }
        simple_wires = (
            set(slot_slot_wires) | set(slot_port_wires) | set(port_port_wires)
        )
    else:
        simple_wires = set()
    graph = MultiGraph()
    graph.add_nodes_from(
        [("wire", w, None) for w in wiring.wires if w not in simple_wires]
    )
    graph.add_nodes_from([("port", p, None) for p in wiring.ports])
    graph.add_nodes_from(list(map(slot_node, wiring.slots)))
    graph.add_edges_from(
        [
            (("wire", w, None), slot_node(slot))
            for slot, slot_wires in enumerate(wiring.slot_wires_list)
            for w in slot_wires
            if w not in simple_wires
        ]
    )
    graph.add_edges_from(
        [
            (("wire", w, None), ("port", p, None))
            for p, w in enumerate(wiring.out_wires)
            if w not in simple_wires
        ]
    )
    if simplify_wires:
        graph.add_edges_from(
            [
                (("port", port, None), slot_node(slot))
                for w, (slot, port) in slot_port_wires.items()
            ]
        )
        graph.add_edges_from(
            [
                (slot_node(w_slots[0]), slot_node(w_slots[1]))
                for w, w_slots in slot_slot_wires.items()
            ]
        )
    return graph


def _port_order_graph_layers(
    diagram: Diagram, port_order: PortOrderStructure
) -> dict[int, list[DiagramGraphNode]]:
    wiring = diagram.wiring
    slot_wires_list = wiring.slot_wires_list
    wired_slot_ports = wiring.wired_slot_ports
    slot_input_ports_list = port_order.slot_input_ports_list
    slot_output_ports_list = port_order.slot_output_ports_list
    slot_inputs = tuple(map(sorted, slot_input_ports_list))
    slot_outputs = tuple(map(sorted, slot_output_ports_list))
    wire_inputs = tuple(
        tuple(
            (slot, port)
            for slot, port in wired_slot_ports[wire]
            if port not in slot_input_ports_list[slot]
        )
        for wire in wiring.wires
    )
    wire_outputs = tuple(
        tuple(
            (slot, port)
            for slot, port in wired_slot_ports[wire]
            if port in slot_input_ports_list[slot]
        )
        for wire in wiring.wires
    )
    num_slot_inputs_to_visit = list(map(len, slot_inputs))
    num_wire_inputs_to_visit = list(map(len, wire_inputs))
    slot_queue: deque[Slot] = deque(
        [
            slot
            for slot, num_to_visit in enumerate(num_slot_inputs_to_visit)
            if num_to_visit == 0
        ]
    )
    wire_queue: deque[Wire] = deque(
        [
            slot
            for slot, num_to_visit in enumerate(num_wire_inputs_to_visit)
            if num_to_visit == 0
        ]
    )
    wire_layers: dict[Wire, int] = {wire: 1 for wire in wire_queue}
    slot_layers: dict[Slot, int] = {slot: 1 for slot in slot_queue}
    while wire_queue or slot_queue:
        while wire_queue:
            wire = wire_queue.popleft()
            for slot, _ in wire_outputs[wire]:
                num_slot_inputs_to_visit[slot] -= 1
                if num_slot_inputs_to_visit[slot] == 0:
                    slot_queue.append(slot)
                    slot_layers[slot] = (
                        max(
                            [
                                wire_layers[slot_wires_list[slot][port]]
                                for port in slot_inputs[slot]
                            ]
                        )
                        + 1
                    )
        while slot_queue:
            slot = slot_queue.popleft()
            for port in slot_outputs[slot]:
                wire = slot_wires_list[slot][port]
                num_wire_inputs_to_visit[wire] -= 1
                if num_wire_inputs_to_visit[wire] == 0:
                    wire_queue.append(wire)
                    wire_layers[wire] = (
                        max([slot_layers[slot] for slot, _ in wire_inputs[wire]]) + 1
                    )
    max_layer = (
        max(
            max(wire_layers.values(), default=-1),
            max(slot_layers.values(), default=-1),
        )
        + 1
    )
    layers: dict[int, list[DiagramGraphNode]] = {i: [] for i in range(max_layer + 1)}
    layers[0].extend(("port", port, None) for port in sorted(port_order.input_ports))
    for wire, wire_layer in wire_layers.items():
        layers[wire_layer].append(("wire", wire, None))
    for slot, slot_layer in slot_layers.items():
        slot_content = diagram.blocks[slot]
        node: DiagramGraphNode
        if isinstance(slot_content, Box):
            node = ("box", slot, slot_content)
        elif isinstance(slot_content, Diagram):
            node = ("diagram", slot, slot_content)
        else:
            assert slot_content is None
            node = ("hole", slot, None)
        layers[slot_layer].append(node)
    for port in sorted(port_order.output_ports):
        layers[max_layer].append(("port", port, None))
    return layers


_T = TypeVar("_T")
"""Invariant type variable"""


class NodeOptionSetters(TypedDict, Generic[_T], total=False):

    box: ValueSetter[Slot | Box | tuple[Slot, Box], _T]
    """Option value setter for nodes corresponding to boxes."""

    open_slot: ValueSetter[Slot, _T]
    """Option value setter for nodes corresponding to open slots."""

    subdiagram: ValueSetter[
        Slot | Diagram | tuple[Slot, Diagram] | DiagramRecipe[Any, Any], _T
    ]
    """Option value setter for nodes corresponding to subdiagrams."""

    out_port: ValueSetter[Port, _T]
    """Option value setter for nodes corresponding to out ports."""

    wire: ValueSetter[Wire, _T]
    """Option value setter for nodes corresponding to wires."""


class KamadaKawaiLayoutKWArgs(TypedDict, total=False):
    dist: Any
    pos: Any
    weight: str
    scale: int
    center: ArrayLike
    dim: int


class CircuitLayoutKWArgs(TypedDict, total=False):
    port_order: PortOrderStructure


class BFSLayoutKWArgs(TypedDict, total=True):
    sources: Sequence[Port]


class DrawDiagramOptions(TypedDict, total=False):
    """Style options for diagram drawing."""

    node_size: NodeOptionSetters[int]
    """Node size for different kinds of nodes."""

    node_color: NodeOptionSetters[str]
    """Node color for different kinds of nodes."""

    node_label: NodeOptionSetters[str]
    """Node label for different kinds of nodes."""

    node_border_thickness: NodeOptionSetters[float]
    """Node border options for different kinds of nodes."""

    node_border_color: NodeOptionSetters[str]
    """Node border options for different kinds of nodes."""

    edge_thickness: float
    """Thickness of edges for wires."""

    font_size: int
    """Font size for node labels."""

    font_color: str
    """Font color for node labels."""

    edge_color: str
    """Edge color."""

    edge_font_size: int
    """Font size for edge labels."""

    edge_font_color: str
    """Font color for edge labels."""

    simplify_wires: bool | None
    """Whether to simplify wires which could be represented by simple edges."""


class DiagramDrawer:
    """
    A diagram-drawing function, with additional logic to handle default option values.
    Based on :func:`~networkx.drawing.nx_pylab.draw_networkx`.
    """

    __defaults: DrawDiagramOptions

    def __new__(cls) -> Self:
        """
        Instantiates a new diagram drawer, with default values for options.

        :meta public:
        """
        self = super().__new__(cls)
        self.__defaults = {
            "node_size": {
                "box": 100,
                "hole": 200,
                "diagram": 200,
                "port": 100,
                "wire": 30,
            },
            "node_color": {
                "box": "white",
                "hole": "white",
                "diagram": "white",
                "port": "white",
                "wire": "darkgray",
            },
            "node_label": {
                "box": "",
                "hole": "",
                "diagram": "",
                "port": str,
                "wire": "",
            },
            "node_border_thickness": {
                "box": 1,
                "hole": 1,
                "diagram": 1,
                "port": 0,
                "wire": 0,
            },
            "node_border_color": {
                "box": "gray",
                "hole": "gray",
                "diagram": "gray",
                "port": "lightgray",
                "wire": "lightgray",
            },
            "edge_thickness": 1,
            "font_size": 6,
            "font_color": "black",
            "edge_color": "lightgray",
            "edge_font_size": 6,
            "edge_font_color": "black",
            "simplify_wires": True,
        }
        return self

    @property
    def defaults(self) -> DrawDiagramOptions:
        """Current default options."""
        return dict_deep_copy(self.__defaults)

    def clone(self) -> DiagramDrawer:
        """Clones the current diagram drawer."""
        instance = DiagramDrawer()
        instance.set_defaults(**self.__defaults)
        return instance

    def set_defaults(self, **defaults: Unpack[DrawDiagramOptions]) -> None:
        """Sets new values for default options."""
        dict_deep_update(self.__defaults, defaults)

    def with_defaults(self, **defaults: Unpack[DrawDiagramOptions]) -> DiagramDrawer:
        """Returns a clone of this diagram drawer, with new defaults."""
        instance = self.clone()
        instance.set_defaults(**defaults)
        return instance

    @overload
    def __call__(
        self,
        diagram: Diagram,
        *,
        layout: Literal["kamada_kawai"] = "kamada_kawai",
        layout_kwargs: KamadaKawaiLayoutKWArgs = {},
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    @overload
    def __call__(
        self,
        diagram: Diagram,
        *,
        layout: Literal["bfs"],
        layout_kwargs: BFSLayoutKWArgs,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    @overload
    def __call__(
        self,
        diagram: Diagram,
        *,
        layout: Literal["circuit"],
        layout_kwargs: CircuitLayoutKWArgs,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    def __call__(
        self,
        diagram: Diagram,
        *,
        layout: str = "kamada_kawai",
        layout_kwargs: Any = MappingProxyType({}),
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None:
        """
        Draws the given diagram using NetworkX.

        :meta public:
        """
        assert validate(diagram, Diagram)
        # assert validate(options, DrawDiagramOptions) # FIXME: currently not supported by validate
        # Include default options:
        default_options = self.__defaults
        _options: DrawDiagramOptions = dict_deep_update(
            dict_deep_copy(default_options), options
        )
        # Create NetworkX graph for diagram + layout:
        graph = diagram_to_nx_graph(diagram, simplify_wires=_options["simplify_wires"])
        pos: dict[DiagramGraphNode, tuple[float, float]]
        match layout:
            case "kamada_kawai":
                assert validate(layout_kwargs, KamadaKawaiLayoutKWArgs)
                pos = nx.kamada_kawai_layout(
                    graph, **cast(KamadaKawaiLayoutKWArgs, layout_kwargs)
                )
            case "bfs":
                out_ports = diagram.ports
                assert validate(layout_kwargs, BFSLayoutKWArgs)
                sources = cast(BFSLayoutKWArgs, layout_kwargs).get("sources")
                if not sources:
                    raise ValueError(
                        "At least one source port must be selected for layered layout."
                    )
                elif not all(s in out_ports for s in sources):
                    raise ValueError("Sources must be valid ports for diagram.")
                layers_list = list(
                    nx.bfs_layers(
                        graph, sources=[("port", i, None) for i in sorted(sources)]
                    )
                )
                layers = dict(enumerate(layers_list))
                reachable_nodes = frozenset(
                    node for layer_nodes in layers.values() for node in layer_nodes
                )
                nodes_to_remove = [
                    node for node in graph.nodes if node not in reachable_nodes
                ]
                if any(nodes_to_remove):
                    graph.remove_nodes_from(nodes_to_remove)
                pos = nx.multipartite_layout(graph, subset_key=layers)
            case "circuit":
                out_ports = diagram.ports
                assert validate(layout_kwargs, CircuitLayoutKWArgs)
                _layout_kwargs = cast(CircuitLayoutKWArgs, layout_kwargs)
                port_order = _layout_kwargs.get("port_order")
                if port_order is None:
                    port_order = diagram.port_order_struct
                if port_order is None:
                    raise ValueError(
                        "No port order specified and diagram has no port order."
                    )
                layers = _port_order_graph_layers(diagram, port_order)
                nodeset = frozenset(graph.nodes)
                layers = {
                    i: [node for node in layer if node in nodeset]
                    for i, layer in layers.items()
                }
                pos = nx.multipartite_layout(graph, subset_key=layers)
            case _:
                raise ValueError(f"Invalid layout choice {layout!r}.")

        # Define utility function to apply option setter to node:
        def _apply[T](
            setter: NodeOptionSetters[T],
            node: DiagramGraphNode,
            default: dict[DiagramGraphNodeKind, T] | None = None,
        ) -> T | None:
            res: Any
            match node[0]:
                case "box":
                    _, box_idx, box = node
                    res = apply_setter(setter["box"], box_idx)
                    if res is None:
                        res = apply_setter(setter["box"], box)
                    if res is None:
                        res = apply_setter(setter["box"], (box_idx, box))
                    if res is None and default is not None:
                        res = default.get("box")
                    return cast(T | None, res)
                case "hole":
                    _, slot_idx, _ = node
                    res = apply_setter(setter["hole"], slot_idx)
                    if res is None and default is not None:
                        res = default.get("hole")
                    return cast(T | None, res)
                case "diagram":
                    _, slot_idx, subdiag = node
                    res = apply_setter(setter["diagram"], slot_idx)
                    if res is None:
                        res = apply_setter(setter["diagram"], subdiag)
                    if res is None:
                        res = apply_setter(setter["diagram"], (slot_idx, subdiag))
                    if res is None and cast(Diagram, subdiag).recipe_used is not None:
                        res = apply_setter(
                            setter["diagram"], cast(Diagram, subdiag).recipe_used
                        )
                    if res is None and default is not None:
                        res = default.get("diagram")
                    return cast(T | None, res)
                case "port":
                    _, port_idx, _ = node
                    res = apply_setter(setter["port"], port_idx)
                    if res is None and default is not None:
                        res = default.get("port")
                    return cast(T | None, res)
                case "wire":
                    _, wire_idx, _ = node
                    res = apply_setter(setter["wire"], wire_idx)
                    if res is None and default is not None:
                        res = default.get("wire")
                    return cast(T | None, res)
                case unknown_kind:
                    assert False, f"Unknown node kind {unknown_kind}"

        # Set options for nx.draw_networkx:
        draw_networkx_options: dict[str, Any] = {}
        node_size_options = _options["node_size"]
        draw_networkx_options["node_size"] = [
            _apply(node_size_options, node) for node in graph.nodes
        ]
        node_color_options = _options["node_color"]
        draw_networkx_options["node_color"] = [
            _apply(node_color_options, node, default_options["node_color"])
            for node in graph.nodes
        ]
        node_label_options = _options["node_label"]
        draw_networkx_options["labels"] = {
            node: label
            for node in graph.nodes
            if (label := _apply(node_label_options, node)) is not None
        }
        node_border_color_options = _options["node_border_color"]
        draw_networkx_options["edgecolors"] = [
            _apply(node_border_color_options, node) for node in graph.nodes
        ]
        node_border_thickness_options = _options["node_border_thickness"]
        draw_networkx_options["linewidths"] = [
            _apply(node_border_thickness_options, node) for node in graph.nodes
        ]
        draw_networkx_options["edge_color"] = _options["edge_color"]
        draw_networkx_options["width"] = _options["edge_thickness"]
        draw_networkx_options["font_size"] = _options["font_size"]
        draw_networkx_options["font_color"] = _options["font_color"]
        # Draw diagram using Matplotlib and nx.draw_networkx:
        edge_counts = {(u, v): i + 1 for u, v, i in sorted(graph.edges)}
        if figsize is not None and ax is not None:
            raise ValueError("Options 'ax' and 'figsize' cannot both be set.")
        if ax is None:
            plt.figure(figsize=figsize)
        nx.draw_networkx(graph, pos, ax=ax, **draw_networkx_options)
        if any(count > 1 for count in edge_counts.values()):
            draw_networkx_edge_options: dict[str, Any] = {}
            draw_networkx_edge_options["font_size"] = _options["edge_font_size"]
            draw_networkx_edge_options["font_color"] = _options["edge_font_color"]
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                {edge: str(count) for edge, count in edge_counts.items() if count > 1},
                **draw_networkx_edge_options,
            )
        if ax is None:
            plt.gca().invert_yaxis()
            plt.axis("off")
            plt.show()

    @overload
    def s(
        self,
        *diagrams: Diagram,
        layout: Literal["kamada_kawai"] = "kamada_kawai",
        layout_kwargs: KamadaKawaiLayoutKWArgs = {},
        figsize: tuple[float, float],
        subplots: tuple[int, int] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    @overload
    def s(
        self,
        *diagrams: Diagram,
        layout: Literal["bfs"],
        layout_kwargs: BFSLayoutKWArgs,
        figsize: tuple[float, float],
        subplots: tuple[int, int] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    def s(
        self,
        *diagrams: Diagram,
        layout: str = "kamada_kawai",
        layout_kwargs: Any = MappingProxyType({}),
        figsize: tuple[float, float],
        subplots: tuple[int, int] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None:
        assert validate(diagrams, tuple[Diagram, ...])
        # assert validate(options, DrawDiagramOptions) # FIXME: currently not supported by validate
        # Include default options:
        _options: DrawDiagramOptions = dict_deep_update(
            dict_deep_copy(self.__defaults), options
        )
        if subplots is not None:
            nrows, ncols = subplots
        else:
            nrows, ncols = 1, len(diagrams)
        plt.figure(figsize=figsize)
        for idx, diagram in enumerate(diagrams):
            plt.subplot(nrows, ncols, idx + 1)
            self(
                diagram,
                layout=layout,
                layout_kwargs=layout_kwargs,
                ax=plt.gca(),
                **options,
            )  # type: ignore
            plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


draw_diagram: Final[DiagramDrawer] = DiagramDrawer()
""" Diagram-drawing function, based :func:`~networkx.drawing.nx_pylab.draw_networkx`."""
