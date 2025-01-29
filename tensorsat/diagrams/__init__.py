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

from .types import Type, TypeT_co, TypeT_inv, Shape
from .boxes import Box, BoxT_inv, BoxRecipe
from .wirings import (
    Slot,
    Port,
    Wire,
    WiringData,
    Shaped,
    Slotted,
    Wiring,
    WiringBuilder,
)
from .diagrams import Block, Diagram, DiagramBuilder, DiagramRecipe

__all__ = (
    "Type",
    "TypeT_co",
    "TypeT_inv",
    "Shape",
    "Box",
    "BoxT_inv",
    "BoxRecipe",
    "Slot",
    "Port",
    "Wire",
    "WiringData",
    "Shaped",
    "Slotted",
    "Wiring",
    "WiringBuilder",
    "Block",
    "Diagram",
    "DiagramBuilder",
    "DiagramRecipe",
)
