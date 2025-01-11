"""
Utilities to construct tensor networks for binary circuits.
"""

from __future__ import annotations
from _tensorsat import CircuitBuilder, Rel, circuit_applicable

not_ = Rel.from_callable((2,), (2,), lambda t: 1-t[0])
and_ = Rel.from_callable((2,2), (2,), lambda t: t[0]&t[1])
or_ = Rel.from_callable((2,2), (2,), lambda t: t[0]|t[1])
xor_ = Rel.from_callable((2,2), (2,), lambda t: t[0]^t[1])
bit_0 = Rel.singleton((2,), 0)
bit_1 = Rel.singleton((2,), 1)
bit_unk = Rel.from_subset((2,), {0, 1})

@circuit_applicable
def half_adder(circ: CircuitBuilder, inputs: tuple[int, ...]) -> tuple[int, int]:
    """
    Applies a half adder to the given inputs in the given circuit.
    See https://en.wikipedia.org/wiki/Adder_(electronics)#Half_adder
    """
    if len(inputs) != 2:
        raise ValueError("Half adder expects exactly 2 inputs.")
    a, b = inputs
    s, = xor_ @ circ[a, b]
    c, = and_ @ circ[a, b]
    return s, c

@circuit_applicable
def full_adder(circ: CircuitBuilder, inputs: tuple[int, ...]) -> tuple[int, int]:
    """
    Applies a full adder to the given inputs in the given circuit.
    See https://en.wikipedia.org/wiki/Adder_(electronics)#Full_adder
    """
    if len(inputs) != 3:
        raise ValueError("Full adder expects exactly 3 inputs.")
    c_in, a, b = inputs
    x1, = xor_ @ circ[a, b]     # a^b
    x2, = and_ @ circ[a, b]     # a&b
    x3, = and_ @ circ[x1, c_in] # (a^b)&c
    s, = xor_ @ circ[x1, x3]    # (a^b)^((a^b)&c)
    c_out, = or_ @ circ[x2, x3] # (a&b)|((a^b)&c)
    return s, c_out

@circuit_applicable
def rc_adder(circ: CircuitBuilder, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """
    Applies a ripple carry adder to the given inputs in the given circuit.
    See https://en.wikipedia.org/wiki/Adder_(electronics)#Ripple-carry_adder
    """
    if len(inputs) % 2 != 1:
        raise ValueError("Ripple carry adder expects odd number of inputs.")
    num_bits = len(inputs)//2
    outputs: list[int] = []
    c = inputs[0]
    for i in range(num_bits):
        a, b = inputs[2*i+1:2*i+3]
        s, c = full_adder @ circ[c, a, b]
        outputs.append(s)
    outputs.append(c)
    return tuple(outputs)
