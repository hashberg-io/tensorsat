"""
Top-level functions for the TensorSat package, for use by the
`autoray <https://github.com/jcmgray/autoray>`_ package.
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
from typing import cast


from .diagrams import Box, BoxT_inv, TensorLikeBox

if __debug__:
    from typing_validation import validate

try:
    import autoray  # type: ignore[import-untyped]

    def einsum(contraction: str, /, lhs: BoxT_inv, rhs: BoxT_inv) -> BoxT_inv:
        """Contracts boxes using einsum notation."""
        assert validate(contraction, str)
        assert validate(lhs, Box)
        assert validate(rhs, Box)
        box_class = Box.class_join([type(lhs), type(rhs)])
        _input_wires, _out_wires = contraction.split("->")
        _lhs_wires, _rhs_wires = _input_wires.split(",")
        char_idxs = {
            letter: idx
            for idx, letter in reversed(list(enumerate(_lhs_wires + _rhs_wires)))
        }
        lhs_wires = [char_idxs[c] for c in _lhs_wires]
        rhs_wires = [char_idxs[c] for c in _rhs_wires]
        out_wires = [char_idxs[c] for c in _out_wires]
        res = box_class.contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)
        return cast(BoxT_inv, res)  # Think about whether this cast can be avoided.

    def transpose(box: BoxT_inv, perm: Sequence[int], /) -> BoxT_inv:
        """Rearranges the ports of a box."""
        assert validate(box, Box)
        return box.rewire(perm)

    @autoray.shape.register("tensorsat._autoray")  # type: ignore[misc]
    def shape(box: TensorLikeBox) -> tuple[int, ...]:
        if not isinstance(box, TensorLikeBox):
            raise TypeError("Autoray shape is only supported by tensor-like boxes.")
        return box.tensor_shape

    def stack(boxes: Sequence[BoxT_inv], axis: int = 0) -> BoxT_inv:
        """Stacks boxes along a new axis."""
        assert validate(boxes, Sequence[Box])
        assert validate(axis, int)
        box_class = Box.class_join([type(box) for box in boxes])
        return cast(BoxT_inv, box_class.stack(boxes, axis))

except ModuleNotFoundError:
    pass
