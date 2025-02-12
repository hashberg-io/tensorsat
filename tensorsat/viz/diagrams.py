"""
Visualisation utilities for diagrams.
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
from collections.abc import Mapping
from typing import Any, Final, Literal, Self, TypedDict, Unpack, cast, reveal_type

from ..diagrams import Block, Slot, Box, Type, Diagram, Wire
from ..utils import (
    ValueSetter as OptionSetter,
    apply_setter,
    dict_deep_copy,
    dict_deep_update,
)

try:
    import matplotlib.pyplot as plt  # noqa: F401
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


def diagram_to_nx_graph(diagram: Diagram) -> nx.Graph:
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
    wired_slots = wiring.wired_slots
    out_wires = set(wiring.out_wires)
    simple_wires = {
        w: w_slots
        for w, w_slots in wired_slots.items()
        if len(w_slots) + int(w in out_wires) == 2
    }
    graph = nx.Graph()
    graph.add_nodes_from(
        [("wire", w, None) for w in wiring.wires if w not in simple_wires]
    )
    graph.add_nodes_from([("out_port", w, None) for w in wiring.out_wires])
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
            (("out_port" if w in out_wires else "wire", w, None), slot_node(w_slots[0]))
            for w, w_slots in simple_wires.items()
            if len(w_slots) == 1
        ]
    )
    graph.add_edges_from(
        [
            (slot_node(w_slots[0]), slot_node(w_slots[1]))
            for w, w_slots in simple_wires.items()
            if len(w_slots) == 2
        ]
    )
    graph.add_edges_from(
        [
            (("wire", w, None), ("out_port", w, None))
            for w in wiring.out_wires
            if w not in simple_wires
        ]
    )
    return graph


# TODO: Reorganise options, grouping sizesm colors, labels, etc,
#       because this is how they are used.
#       Remember to deep-copy the respective dictionaries when setting defaults:
#       it might be helpful to have a function that does such nested typed dict update.


class NodeOptionSetters[T](TypedDict, total=False):

    box: OptionSetter[Slot | Box | tuple[Slot, Box], T]
    """Option value setter for nodes corresponding to boxes."""

    open_slot: OptionSetter[Slot, T]
    """Option value setter for nodes corresponding to open slots."""

    subdiagram: OptionSetter[Slot | Diagram | tuple[Slot, Diagram], T]
    """Option value setter for nodes corresponding to subdiagrams."""

    out_port: OptionSetter[Wire, T]
    """Option value setter for nodes corresponding to out ports."""

    wire: OptionSetter[Wire, T]
    """Option value setter for nodes corresponding to wires."""


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
        """
        Instantiates a new diagram drawer, with default values for options.

        .. warning::

            Default option values are currently subject to change without notice.

        """
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
                "box": "lightgray",
                "open_slot": "lightgray",
                "subdiagram": "lightgray",
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

    def __call__(
        self,
        diagram: Diagram,
        pos: str | Mapping[DiagramGraphNode, tuple[int, int]] = "kamada_kawai",
        **options: Unpack[DrawDiagramOptions],
    ) -> None:
        """Draws the given diagram using NetworkX."""
        assert validate(diagram, Diagram)
        assert validate(pos, str | Mapping[DiagramGraphNode, tuple[int, int]])
        # assert validate(options, DrawDiagramOptions) # FIXME: currently not supported by validate
        # Include default options:
        options = dict_deep_update(dict_deep_copy(self.__defaults), options)
        # Create NetworkX graph for the diagram:
        graph = diagram_to_nx_graph(diagram)
        # If layout function is passed, use it to generate node positions:
        if isinstance(pos, str):
            try:
                layout_fun = getattr(nx, pos + "_layout")
            except AttributeError:
                raise ValueError(f"No NetworkX layout called: {pos}_layout.")
            pos = layout_fun(graph)

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
        node_size_options = options["node_size"]
        draw_networkx_options["node_size"] = [
            _apply(node_size_options, node) for node in graph.nodes
        ]
        node_color_options = options["node_color"]
        draw_networkx_options["node_color"] = [
            _apply(node_color_options, node) for node in graph.nodes
        ]
        node_label_options = options["node_label"]
        draw_networkx_options["labels"] = {
            node: label
            for node in graph.nodes
            if (label := _apply(node_label_options, node)) is not None
        }
        node_border_color_options = options["node_border_color"]
        draw_networkx_options["edgecolors"] = [
            _apply(node_border_color_options, node) for node in graph.nodes
        ]
        node_border_thickness_options = options["node_border_thickness"]
        draw_networkx_options["linewidths"] = [
            _apply(node_border_thickness_options, node) for node in graph.nodes
        ]
        draw_networkx_options["edge_color"] = options["node_color"]["wire"]
        draw_networkx_options["width"] = options["edge_thickness"]
        draw_networkx_options["font_size"] = options["font_size"]
        # Draw diagram using nx.draw_networkx:
        nx.draw_networkx(graph, pos, **draw_networkx_options)


draw_diagram: Final[DiagramDrawer] = DiagramDrawer()
