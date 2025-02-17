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
from collections.abc import Mapping, Sequence
from itertools import product
from types import MappingProxyType
from typing import Final
from ..diagrams import Diagram, DiagramBuilder, Wire
from ..lang.fin_rel import FinSet, FinRel

bit: Final[FinSet] = FinSet(2)
"""The set {0, 1} of binary values."""

not_: Final[FinRel] = FinRel.from_callable(bit, bit, lambda b: 1 - b)
"""The NOT gate."""

and_: Final[FinRel] = FinRel.from_callable(bit * bit, bit, lambda a, b: a&b)
"""The AND gate."""

or_: Final[FinRel] = FinRel.from_callable(bit * bit, bit, lambda a, b: a|b)
"""The OR gate."""

xor_: Final[FinRel] = FinRel.from_callable(bit * bit, bit, lambda a, b: a^b)
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


@Diagram.from_recipe(bit**2)
def half_adder(circ: DiagramBuilder[FinSet], inputs: Sequence[Wire]) -> Sequence[Wire]:
    """
    Diagram for a half adder circuit.
    See https://en.wikipedia.org/wiki/Adder_(electronics)#Half_adder
    """
    a, b = inputs
    (s,) = xor_ @ circ[a, b]
    (c,) = and_ @ circ[a, b]
    return s, c


@Diagram.from_recipe(bit**3)
def full_adder(circ: DiagramBuilder[FinSet], inputs: Sequence[Wire]) -> Sequence[Wire]:
    """
    Diagram for a full adder circuit.
    See https://en.wikipedia.org/wiki/Adder_(electronics)#Full_adder
    """
    a, b, c_in = inputs
    (x1,) = xor_ @ circ[a, b]
    (x2,) = and_ @ circ[a, b]
    (x3,) = and_ @ circ[x1, c_in]
    (s,) = xor_ @ circ[x1, x3]
    (c_out,) = or_ @ circ[x2, x3]
    return s, c_out


@Diagram.recipe
def rc_adder(circ: DiagramBuilder[FinSet], inputs: Sequence[Wire]) -> Sequence[Wire]:
    """
    Recipe to create a ripple carry adder.
    The recipe can be called on given input types to create a diagram:

    .. code-block:: python

        rc_adder_2bit: Diagram[FinSet] = rc_adder(bit**5)

    Alternatively, the recipe can be applied to wires in a circuit builder,
    obtaining the output wires for the builder sub-circuit in return:

    .. code-block:: python

        s0, s1, c_out = rc_adder @ some_circuit[c_in, a0, b0, a1, b1]

    See https://en.wikipedia.org/wiki/Adder_(electronics)#Ripple-carry_adder
    """
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


@Diagram.recipe
def wallace_multiplier(
    circ: DiagramBuilder[FinSet], inputs: Sequence[Wire]
) -> Sequence[Wire]:
    """
    Recipe to create a Wallace multiplier.
    The recipe can be called on given input types to create a diagram:

    .. code-block:: python

        rc_adder_2bit: Diagram[FinSet] = wallace_multiplier(bit**4)

    Alternatively, the recipe can be applied to wires in a circuit builder,
    obtaining the output wires for the builder sub-circuit in return:

    .. code-block:: python

        c0, c1, c2, c3 = wallace_multiplier @ some_circuit[a0, a1, b0, b1]

    See https://en.wikipedia.org/wiki/Wallace_tree
    """
    if len(inputs) % 2 != 0:
        raise ValueError("Wallace multiplier expects even number of inputs.")
    if not inputs:
        return ()
    n = len(inputs) // 2
    a = inputs[:n]
    b = inputs[n:]
    layer: list[list[Wire]] = [[] for _ in range(2 * n)]
    for i, j in product(range(n), repeat=2):
        (_out,) = and_ @ circ[a[i], b[j]]
        layer[i + j].append(_out)
    while any(len(wires) > 1 for wires in layer):
        new_layer: list[list[Wire]] = [[] for _ in range(2 * n)]
        for weight, wires in enumerate(layer):
            num_fulladd, _r = divmod(len(wires), 3)
            for idx in range(num_fulladd):
                s, c_out = full_adder @ circ[*wires[3 * idx : 3 * idx + 3]]
                new_layer[weight].append(s)
                new_layer[weight + 1].append(c_out)
            if _r == 2:
                s, c_out = half_adder @ circ[*wires[-2:]]
                new_layer[weight].append(s)
                new_layer[weight + 1].append(c_out)
            elif _r == 1:
                new_layer[weight].append(wires[-1])
        layer = new_layer
    assert all(len(wires) == 1 for wires in layer)
    return tuple(wires[0] for wires in layer)
