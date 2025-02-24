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
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final, Literal, Self, TypedDict, Unpack, cast, overload
from numpy.typing import ArrayLike

from ..diagrams import Slot, Box, Diagram, Wire, Port
from ..utils import (
    ValueSetter as OptionSetter,
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
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For diagram visualisation, 'networkx' must be installed."
    )

if __debug__:
    from typing_validation import validate

DiagramGraphNodeKind = Literal["box", "open_slot", "subdiagram", "out_port", "wire"]
"""Type alias for possible kinds of nodes in the NetworkX graph for a diagram."""

DIAGRAM_GRAPH_NODE_KIND: Final[tuple[DiagramGraphNodeKind, ...]] = (
    "box", "open_slot", "subdiagram", "out_port", "wire"
)
"""Possible kinds of nodes in the NetworkX graph for a diagram."""


DiagramGraphNode = (
    tuple[Literal["box"], int, Box]  # ("box", slot, box)
    | tuple[Literal["open_slot"], int, None]  # ("open_slot", slot, None)
    | tuple[Literal["subdiagram"], int, Diagram]  # ("subdiagram", slot, diagram)
    | tuple[Literal["out_port"], int, None]  # ("out_port", port, None)
    | tuple[Literal["wire"], int, None]  # ("wire", wire, None)
)
"""
Type alias for a node in the NetworkX graph representing a diagram,
labelled by the triple ``(kind, index, data)``.
"""

def diagram_to_nx_graph(
    diagram: Diagram,
    # *,
    # simplify_wires: bool = False
) -> nx.Graph:
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
            return ("open_slot", slot, None)
        if slot in subdiagram_slots:
            subdiagram = blocks[slot]
            assert isinstance(subdiagram, Diagram)
            return ("subdiagram", slot, subdiagram)
        assert False, "Slot must be open, filled with a box, or filled with a diagram."

    wiring = diagram.wiring
    # wired_slots = wiring.wired_slots
    # out_wires = set(wiring.out_wires)
    # if simplify_wires:
    #     simple_wires = {
    #         w: w_slots
    #         for w, w_slots in wired_slots.items()
    #         if len(w_slots) + int(w in out_wires) == 2
    #         # FIXME: incorrect^^^^^^^^^^^^^^^^^^^ w may connect to multiple out ports
    #     }
    # else:
    #     simple_wires = {}
    graph = nx.Graph()
    graph.add_nodes_from([
            ("wire", w, None)
            for w in wiring.wires
            # if w not in simple_wires
    ])
    graph.add_nodes_from([("out_port", p, None) for p in wiring.ports])
    graph.add_nodes_from(list(map(slot_node, wiring.slots)))
    graph.add_edges_from(
        [
            (("wire", w, None), slot_node(slot))
            for slot, slot_wires in enumerate(wiring.slot_wires_list)
            for w in slot_wires
            # if w not in simple_wires
        ]
    )
    graph.add_edges_from(
        [
            (("wire", w, None), ("out_port", p, None))
            for p, w in enumerate(wiring.out_wires)
            # if w not in simple_wires
        ]
    )
    # if simplify_wires:
    #     graph.add_edges_from(
    #         [
    #             (("out_port" if w in out_wires else "wire", w, None), slot_node(w_slots[0]))
    #             # FIXME:      this should be the port index ^
    #             for w, w_slots in simple_wires.items()
    #             if len(w_slots) == 1
    #         ]
    #     )
    #     graph.add_edges_from(
    #         [
    #             (slot_node(w_slots[0]), slot_node(w_slots[1]))
    #             for w, w_slots in simple_wires.items()
    #             if len(w_slots) == 2
    #         ]
    #     )
    return graph

class NodeOptionSetters[T](TypedDict, total=False):

    box: OptionSetter[Slot | Box | tuple[Slot, Box], T]
    """Option value setter for nodes corresponding to boxes."""

    open_slot: OptionSetter[Slot, T]
    """Option value setter for nodes corresponding to open slots."""

    subdiagram: OptionSetter[Slot | Diagram | tuple[Slot, Diagram], T]
    """Option value setter for nodes corresponding to subdiagrams."""

    out_port: OptionSetter[Port, T]
    """Option value setter for nodes corresponding to out ports."""

    wire: OptionSetter[Wire, T]
    """Option value setter for nodes corresponding to wires."""


class KamadaKawaiLayoutKWArgs(TypedDict, total=False):
    dist: Any
    pos: Any
    weight: str
    scale: int
    center: ArrayLike
    dim: int

class LayeredLayoutKWArgs(TypedDict, total=False):
    sources: Sequence[Port]

class DrawDiagramOptions(TypedDict, total=False):
    """Options for diagram drawing."""

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
    """Font size."""


class DiagramDrawer:
    """
    A diagram-drawing function, with additional logic to handle default option values.
    Based on :func:`networkx.draw_networkx`.
    """

    __defaults: DrawDiagramOptions

    def __new__(cls) -> Self:
        """Instantiates a new diagram drawer, with default values for options."""
        self = super().__new__(cls)
        self.__defaults = {
            "node_size": {
                "box": 100,
                "open_slot": 200,
                "subdiagram": 200,
                "out_port": 100,
                "wire": 30,
            },
            "node_color": {
                "box": "white",
                "open_slot": "white",
                "subdiagram": "white",
                "out_port": "white",
                "wire": "lightgray",
            },
            "node_label": {
                "box": "",
                "open_slot": "",
                "subdiagram": "",
                "out_port": str,
                "wire": "",
            },
            "node_border_thickness": {
                "box": 1,
                "open_slot": 1,
                "subdiagram": 1,
                "out_port": 0,
                "wire": 0,
            },
            "node_border_color": {
                "box": "gray",
                "open_slot": "gray",
                "subdiagram": "gray",
                "out_port": "lightgray",
                "wire": "lightgray",
            },
            "edge_thickness": 1,
            "font_size": 6,
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
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: Literal["kamada_kawai"],
        layout_kwargs: KamadaKawaiLayoutKWArgs = {},
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    @overload
    def __call__(
        self,
        diagram: Diagram,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: Literal["layered"],
        layout_kwargs: LayeredLayoutKWArgs,
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    @overload
    def __call__(
        self,
        diagram: Diagram,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: None = None,
        layout_kwargs: Any = MappingProxyType({}),
        **options: Unpack[DrawDiagramOptions],
    ) -> None: ...

    def __call__(
        self,
        diagram: Diagram,
        *,
        layout: str | None = None,
        layout_kwargs: Any = MappingProxyType({}),
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **options: Unpack[DrawDiagramOptions],
    ) -> None:
        """Draws the given diagram using NetworkX."""
        assert validate(diagram, Diagram)
        # assert validate(options, DrawDiagramOptions) # FIXME: currently not supported by validate
        # Include default options:
        _options: DrawDiagramOptions = dict_deep_update(
            dict_deep_copy(self.__defaults),
            options
        )
        # Create NetworkX graph for diagram + layout:
        graph: nx.Graph
        pos: dict[DiagramGraphNode, tuple[float, float]]
        if layout is None:
            layout = "kamada_kawai" if diagram.input_ports is None else "layered"
        match layout:
            case "kamada_kawai":
                assert validate(layout_kwargs, KamadaKawaiLayoutKWArgs)
                # graph = diagram_to_nx_graph(diagram, simplify_wires=True)
                graph = diagram_to_nx_graph(diagram)
                pos = nx.kamada_kawai_layout(
                    graph,
                    **cast(KamadaKawaiLayoutKWArgs, layout_kwargs)
                )
            case "layered":
                out_ports = diagram.ports
                assert validate(layout_kwargs, LayeredLayoutKWArgs)
                sources = cast(LayeredLayoutKWArgs, layout_kwargs).get("sources")
                if sources is None:
                    if diagram.input_ports is None:
                        raise ValueError(
                            "Cannot automatically select source nodes for layered layout:"
                            " diagram doesn't have input ports selected."
                        )
                    sources = diagram.input_ports
                elif not sources:
                    raise ValueError(
                        "At least one source port must be selected for layered layout."
                    )
                elif not all(s in out_ports for s in sources):
                    raise ValueError("Sources must be valid ports for diagram.")
                graph = diagram_to_nx_graph(diagram)
                layers_list = list(nx.bfs_layers(graph, sources=[
                    ("out_port", i, None) for i in sorted(sources)
                ]))
                layers = dict(enumerate(layers_list))
                reachable_nodes = frozenset(
                    node
                    for layer_nodes in layers.values()
                    for node in layer_nodes
                )
                nodes_to_remove = [
                    node for node in graph.nodes if node not in reachable_nodes
                ]
                if any(nodes_to_remove):
                    graph.remove_nodes_from(nodes_to_remove)
                pos = nx.multipartite_layout(graph, subset_key=layers)
            case _:
                raise ValueError(f"Invalid layout choice {layout!r}.")
        # Define utility function to apply option setter to node:
        def _apply[T](setter: NodeOptionSetters[T], node: DiagramGraphNode) -> T | None:
            match node[0]:
                case "box":
                    res: T | None
                    res = apply_setter(setter["box"], node[1])  # type: ignore
                    if res is None:
                        res = apply_setter(setter["box"], node[2])  # type: ignore
                    if res is None:
                        res = apply_setter(setter["box"], (node[1], node[2]))  # type: ignore
                    return res
                case "open_slot":
                    return apply_setter(setter["open_slot"], node[1])
                case "subdiagram":
                    res = apply_setter(setter["subdiagram"], node[1])  # type: ignore
                    if res is None:
                        res = apply_setter(setter["subdiagram"], node[2])  # type: ignore
                    if res is None:
                        res = apply_setter(setter["subdiagram"], (node[1], node[2]))  # type: ignore
                    return res
                case "out_port":
                    return apply_setter(setter["out_port"], node[1])
                case "wire":
                    return apply_setter(setter["wire"], node[1])
            raise NotImplementedError()

        # Set options for nx.draw_networkx:
        draw_networkx_options: dict[str, Any] = {}
        node_size_options = _options["node_size"]
        draw_networkx_options["node_size"] = [
            _apply(node_size_options, node) for node in graph.nodes
        ]
        node_color_options = _options["node_color"]
        draw_networkx_options["node_color"] = [
            _apply(node_color_options, node) for node in graph.nodes
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
        draw_networkx_options["edge_color"] = _options["node_color"]["wire"]
        draw_networkx_options["width"] = _options["edge_thickness"]
        draw_networkx_options["font_size"] = _options["font_size"]
        # Draw diagram using Matplotlib and nx.draw_networkx:
        if ax is None:
            plt.figure(figsize=figsize)
        nx.draw_networkx(graph, pos, ax=ax, **draw_networkx_options)
        if ax is None:
            plt.gca().invert_yaxis()
            plt.axis("off")
            plt.show()


draw_diagram: Final[DiagramDrawer] = DiagramDrawer()
""" Diagram-drawing function, based :func:`networkx.draw_networkx`."""
