"""Library of boxes and diagrams for binary circuits."""

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
from itertools import product
from types import MappingProxyType
from typing import Any, Final

if __debug__:
    from typing_validation import validate

from ..diagrams import Diagram, DiagramBuilder, DiagramRecipe, Wire
from ..lang.fin_rel import FinSet, FinRel, FinSetShape

bit: Final[FinSet] = FinSet(2)
"""The set {0, 1} of binary values."""


def bits(n: int) -> FinSetShape:
    """Shape consisting of the given number of bits."""
    if n < 0:
        raise ValueError("Number of bits must be non-negative.")
    return (bit,) * n


not_: Final[FinRel] = FinRel.from_callable(bit, bit, lambda b: 1 - b)
"""The NOT gate."""

and_: Final[FinRel] = FinRel.from_callable(bits(2), bit, lambda a, b: a & b)
"""The AND gate."""

or_: Final[FinRel] = FinRel.from_callable(bits(2), bit, lambda a, b: a | b)
"""The OR gate."""

xor_: Final[FinRel] = FinRel.from_callable(bits(2), bit, lambda a, b: a ^ b)
"""The XOR gate."""

bit_0: Final[FinRel] = FinRel.singleton(bit, 0)
"""The constant binary value 0."""

bit_1: Final[FinRel] = FinRel.singleton(bit, 1)
"""The constant binary value 1."""

binop_labels: Final[Mapping[FinRel, str]] = MappingProxyType(
    {
        not_: "~",
        and_: "&",
        or_: "|",
        xor_: "^",
        bit_0: "0",
        bit_1: "1",
    }
)
"""Labels for binary operations."""


half_adder: Diagram
"""
Diagram for a half adder circuit.
See `Half adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Half_adder>`_.
"""


@Diagram.from_recipe  # type: ignore[no-redef]
def half_adder(diag: DiagramBuilder) -> None:
    a, b = diag.add_inputs(bits(2))
    (s,) = xor_ @ diag[a, b]
    (c,) = and_ @ diag[a, b]
    diag.add_outputs([s, c])


full_adder: Diagram
"""
Diagram for a full adder circuit.
See `Full adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Full_adder>`_.
"""


@Diagram.from_recipe  # type: ignore[no-redef]
def full_adder(diag: DiagramBuilder) -> None:
    a, b, c_in = diag.add_inputs(bits(3))
    (x1,) = xor_ @ diag[a, b]
    (x2,) = and_ @ diag[a, b]
    (x3,) = and_ @ diag[x1, c_in]
    (s,) = xor_ @ diag[x1, x3]
    (c_out,) = or_ @ diag[x2, x3]
    diag.add_outputs([s, c_out])


rc_adder: DiagramRecipe[Any, FinSet]
"""
Recipe to create a ripple carry adder circuit,
given the number ``num_bits`` of bits for each summand.
See `Ripple-carry adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Ripple-carry_adder>`_.
"""


@Diagram.recipe  # type: ignore[no-redef]
def rc_adder(diag: DiagramBuilder, num_bits: int) -> None:
    assert validate(num_bits, int)
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")
    inputs = diag.add_inputs(bits(2 * num_bits + 1))
    outputs: list[Wire] = []
    c = inputs[0]
    for i in range(num_bits):
        a, b = inputs[2 * i + 1 : 2 * i + 3]
        s, c = full_adder @ diag[c, a, b]
        outputs.append(s)
    outputs.append(c)
    diag.add_outputs(outputs)


wallace_multiplier: DiagramRecipe[Any]
"""
Recipe to create a Wallace multiplier circuit,
given the number ``num_bits`` of bits for each factor.
See `Wallace tree <https://en.wikipedia.org/wiki/Wallace_tree>`_.
"""


@Diagram.recipe  # type: ignore[no-redef]
def wallace_multiplier(diag: DiagramBuilder, num_bits: int) -> None:
    assert validate(num_bits, int)
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")
    a = diag.add_inputs(bits(num_bits))
    b = diag.add_inputs(bits(num_bits))
    layer: dict[int, list[Wire]] = {}
    for i, j in product(range(num_bits), repeat=2):
        (_out,) = and_ @ diag[a[i], b[j]]
        layer.setdefault(i + j, []).append(_out)
    while any(len(wires) > 1 for wires in layer.values()):
        new_layer: dict[int, list[Wire]] = {}
        for weight, wires in layer.items():
            num_fulladd, _r = divmod(len(wires), 3)
            for idx in range(num_fulladd):
                s, c_out = full_adder @ diag[*wires[3 * idx : 3 * idx + 3]]
                new_layer.setdefault(weight, []).append(s)
                new_layer.setdefault(weight + 1, []).append(c_out)
            if _r == 2:
                s, c_out = half_adder @ diag[*wires[-2:]]
                new_layer.setdefault(weight, []).append(s)
                new_layer.setdefault(weight + 1, []).append(c_out)
            elif _r == 1:
                new_layer.setdefault(weight, []).append(wires[-1])
        layer = new_layer
    assert all(len(wires) == 1 for wires in layer.values())
    diag.add_outputs(wires[0] for wires in layer.values())
