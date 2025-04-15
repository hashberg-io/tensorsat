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
from typing import Final

if __debug__:
    from typing_validation import validate

from ..diagrams import Diagram, DiagramBuilder, Wire
from ..lang.fin_rel import FinSet, FinRel, FinSetShape

bit: Final[FinSet] = FinSet(2)
"""The set {0, 1} of binary values."""


def bits(n: int) -> FinSetShape:
    """Shape consisting of the given number of bits."""
    if n < 0:
        raise ValueError("Number of bits must be non-negative.")
    return (bit,) * n


# We define a constant for the 2-bit shape, because binary operations are very common:
bits2: Final[FinSetShape] = bits(2)
"""The shape of 2 bits."""

not_: Final[FinRel] = FinRel.from_callable(bit, bit, lambda b: 1 - b, name="not_")
"""The NOT gate."""

and_: Final[FinRel] = FinRel.from_callable(bits2, bit, lambda a, b: a & b, name="and_")
"""The AND gate."""

or_: Final[FinRel] = FinRel.from_callable(bits2, bit, lambda a, b: a | b, name="or_")
"""The OR gate."""

impl_: Final[FinRel] = FinRel.from_callable(bits2, bit, lambda a, b: (1-a) | b, name="impl_")
"""The IMPL  gate."""

biimpl_: Final[FinRel] = FinRel.from_callable(bits2, bit, lambda a, b: a == b, name="biimpl_")
"""The BI-IMPL  gate."""

xor_: Final[FinRel] = FinRel.from_callable(bits2, bit, lambda a, b: a ^ b, name="xor_")
"""The XOR gate."""

bit_0: Final[FinRel] = FinRel.singleton(bit, 0)
"""The constant binary value 0."""

bit_1: Final[FinRel] = FinRel.singleton(bit, 1)
"""The constant binary value 1."""

bit_unk: Final[FinRel] = FinRel.from_set(bit, {0, 1})
"""The set ``{0, 1}`` of both possible binary values."""

binop_labels: Final[Mapping[FinRel, str]] = MappingProxyType(
    {
        not_: "~",
        and_: "&",
        or_: "|",
        xor_: "^",
        bit_0: "0",
        bit_1: "1",
        bit_unk: "?",
    }
)
"""Labels for binary operations."""


@Diagram.from_recipe
def half_adder(diag: DiagramBuilder) -> None:
    """
    Diagram for a half adder circuit.
    See `Half adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Half_adder>`_.
    """
    # fmt: off
    a, b = diag.add_inputs(bits(2))
    s, = xor_ @ diag[a, b]
    c, = and_ @ diag[a, b]
    diag.add_outputs([s, c])
    # fmt: on


@Diagram.from_recipe
def full_adder(diag: DiagramBuilder) -> None:
    """
    Diagram for a full adder circuit.
    See `Full adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Full_adder>`_.
    """
    # fmt: off
    a, b, c_in = diag.add_inputs(bits(3))
    x1, = xor_ @ diag[a, b]
    x2, = and_ @ diag[a, b]
    x3, = and_ @ diag[x1, c_in]
    s, = xor_ @ diag[x1, c_in]
    c_out, = or_ @ diag[x2, x3]
    diag.add_outputs([s, c_out])
    # fmt: on


@Diagram.recipe
def rc_adder(diag: DiagramBuilder, num_bits: int) -> None:
    """
    Recipe to create a ripple carry adder circuit,
    given the number ``num_bits`` of bits for each summand.
    See `Ripple-carry adder <https://en.wikipedia.org/wiki/Adder_(electronics)#Ripple-carry_adder>`_.
    """
    # fmt: off
    assert validate(num_bits, int)
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")
    a = diag.add_inputs(bits(num_bits))
    b = diag.add_inputs(bits(num_bits))
    c = diag.add_input(bit)
    for i in range(num_bits):
        s, c = full_adder @ diag[a[i], b[i], c]
        diag.add_output(s)
    diag.add_output(c)
    # fmt: on


@Diagram.recipe
def wallace_multiplier(diag: DiagramBuilder, num_bits: int) -> None:
    """
    Recipe to create a Wallace multiplier circuit,
    given the number ``num_bits`` of bits for each factor.
    See `Wallace tree <https://en.wikipedia.org/wiki/Wallace_tree>`_.
    """
    # fmt: off
    assert validate(num_bits, int)
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")
    a = diag.add_inputs(bits(num_bits))
    b = diag.add_inputs(bits(num_bits))
    layer: dict[int, list[Wire]] = {}
    for i, j in product(range(num_bits), repeat=2):
        _out, = and_ @ diag[a[i], b[j]]
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
    # fmt: on
