"""
Top-level functions for the TensorSAT package, for use by the :mod:`autoray` package.
"""

from collections.abc import Sequence
from .diagrams import BoxT_inv


def einsum(contraction: str, /, lhs: BoxT_inv, rhs: BoxT_inv) -> BoxT_inv:
    """Contracts boxes using einsum notation."""
    cls = type(lhs)
    if not isinstance(rhs, cls):
        raise NotImplementedError(
            "Contraction between boxes of different box classes is not yet implemented."
        )
    _input_wires, _out_wires = contraction.split("->")
    _lhs_wires, _rhs_wires = _input_wires.split(",")
    char_idxs = {
        letter: idx
        for idx, letter in reversed(list(enumerate(_lhs_wires + _rhs_wires)))
    }
    lhs_wires = [char_idxs[c] for c in _lhs_wires]
    rhs_wires = [char_idxs[c] for c in _rhs_wires]
    out_wires = [char_idxs[c] for c in _out_wires]
    return cls.contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)


def transpose(rel: BoxT_inv, perm: Sequence[int], /) -> BoxT_inv:
    """Rearranges the ports of a box."""
    return rel.transpose(perm)
