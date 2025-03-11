"""Library of diagram factories for SAT problems."""

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
from math import comb
import re
from typing import Literal, Self, TypeAlias
import numpy as np

from .._utils.meta import TensorSatMeta
from ..diagrams import Diagram, DiagramBuilder, Wire, Wiring
from .bincirc import bit, bit_0, bit_1, bit_unk, bits, not_, or_

if __debug__:
    from typing_validation import validate

Clause: TypeAlias = tuple[int, ...]
"""
A SAT clause, as a tuple of non-zero integers representing the literals in the clause,
with the integer sign determining whether the literal is positive or negative.
"""

CNFDiagramMode: TypeAlias = Literal["bintree"]
"""Type alias for available diagram creation modes in :class:`CNFInstance`."""


class CNFInstance(metaclass=TensorSatMeta):
    """A SAT instance in CNF form."""

    @classmethod
    def random(
        cls, k: int, n: int, m: int, *, rng: int | np.random.Generator | None = None
    ) -> Self:
        """
        Generates a random ``k``-SAT instance in CNF form,
        with ``n`` variables and ``m`` clauses.

        See https://arxiv.org/abs/1405.3558
        """
        if k <= 0:
            raise ValueError("Clause size 'k' must be positive.")
        if n < k:
            raise ValueError(f"Number of variables 'n' must be at least {k = }.")
        if m <= 0:
            raise ValueError("Number of clauses 'm' must be positive.")
        if m > comb(n, k) * 2**k:
            raise ValueError(
                f"Number of clauses 'm' cannot exceed {comb(n, k)*2**k = }"
            )
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        num_clauses = 0
        clauses: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while num_clauses < m:
            vs = map(int, rng.choice(range(n), size=k, replace=False))
            signs = map(int, rng.choice(range(2), size=k))
            clause = tuple(
                sorted((v + 1 if n == 0 else -v - 1 for v, n in zip(vs, signs)))
            )
            if clause not in seen:
                clauses.append(clause)
                seen.add(clause)
                num_clauses += 1
        return cls._new(n, tuple(clauses))

    @classmethod
    def from_dimacs(cls, dimacs: str) -> Self:
        """Creates a SAT instance from a DIMACS formatted string."""
        assert validate(dimacs, str)
        lines = [
            stripped_line
            for line in dimacs.strip().split("\n")
            if (stripped_line := line.strip()) and not stripped_line.startswith("c")
        ]
        start_match = re.compile(r"p cnf ([0-9]+) ([0-9]+)").match(lines[0])
        if not start_match:
            raise ValueError(
                "DIMACS code must start with 'p cnf <num vars> <num clauses>'."
            )
        num_vars, num_clauses = map(int, start_match.groups())
        clauses = tuple(tuple(map(int, line.split(" ")))[:-1] for line in lines[1:])
        if len(clauses) != num_clauses:
            raise ValueError("Number of clauses does not match the specified number.")
        if num_vars < max(abs(lit) for clause in clauses for lit in clause):
            raise ValueError("Clauses contain invalid variables.")
        return cls._new(num_vars, clauses)

    @classmethod
    def _new(cls, num_vars: int, clauses: tuple[Clause, ...]) -> Self:
        """Protected constructor for SAT instances."""
        self = super().__new__(cls)
        self.__num_vars = num_vars
        self.__clauses = clauses
        return self

    __num_vars: int
    __clauses: tuple[Clause, ...]

    def __new__(cls, num_vars: int, clauses: Sequence[Sequence[int]]) -> Self:
        """
        Create a SAT instance from a number of vars and a sequence of clauses.

        :meta public:
        """
        assert validate(num_vars, int)
        assert validate(clauses, Sequence[Sequence[int]])
        if num_vars < max(abs(lit) for clause in clauses for lit in clause):
            raise ValueError("Clauses contain invalid variables.")
        return cls._new(num_vars, tuple(tuple(clause) for clause in clauses))

    @property
    def num_vars(self) -> int:
        """Number of variables in the SAT instance."""
        return self.__num_vars

    @property
    def clauses(self) -> tuple[Clause, ...]:
        """Clauses in the SAT instance."""
        return self.__clauses

    def to_dimacs(self) -> str:
        """Convert the SAT instance to the DIMACS format."""
        num_clauses = len(self.clauses)
        lines = [
            f"p cnf {self.num_vars} {num_clauses}",
            *(f"{' '.join(map(str, clause))} 0" for clause in self.clauses),
        ]
        return "\n".join(lines)

    def _normalize_inputs(self, inputs: str | int | None) -> str:
        n = self.num_vars
        if isinstance(inputs, int):
            if inputs < 0:
                raise ValueError("Inputs must be specified by a non-negative number.")
            inputs = f"{inputs:0>{n}b}"
        elif inputs is None:
            inputs = "?" * n
        else:
            assert validate(inputs, str)
            if not all(b in "01?_" for b in inputs):
                raise ValueError(
                    "Input specification characters must be '0', '1', '?' or '_'."
                )
            if len(inputs) != n:
                raise ValueError(
                    f"Expected input specification string to be {n} characters long,"
                    f" found {len(inputs)} characters instead."
                )
        return inputs

    def inputs(
        self,
        inputs: str | int | None,
        /,
        *,
        discard_unk: bool = False,
    ) -> Diagram:
        """
        Returns a diagram corresponding to the given selection of input values.
        Input values for the formula's variables can be selected by

        - Use '0' to set the input to the 0 value (i.e. the ``{0}`` subset).
        - Use '1' to set the input to the 1 value (i.e. the ``{1}`` subset).
        - Use '?' to set the input to an unknown value (i.e. the ``{0, 1}`` subset).
        - Use '_' to leave the input open.

        The integer form must be a non-negative integer, and it is converted to its
        binary string representation.
        If :obj:`None` is passed, all inputs are set to the unknown value.

        If ``discard_unk`` is set to :obj:`True`, then unknown inputs are discarded;
        otherwise, the unknown bit state :obj:`bit_unk` is used (default).
        """
        inputs = self._normalize_inputs(inputs)
        builder = DiagramBuilder()
        for i, b in enumerate(inputs):
            w: Wire
            match b:
                case "0":
                    (w,) = bit_0 @ builder
                case "1":
                    (w,) = bit_1 @ builder
                case "?":
                    if discard_unk:
                        w = builder.wiring.add_wire(bit)
                    else:
                        (w,) = bit_unk @ builder
                case None:
                    w = builder.add_input(bit)
            assert w == i  # Added wires correspond to variable indices.
        builder.add_outputs(builder.wiring.wires)
        return builder.diagram

    def diagram(
        self,
        *,
        mode: CNFDiagramMode = "bintree",
    ) -> Diagram:
        """
        Returns the diagram associated with this CNF instance.

        .. warning::

            Currently, diagram building algorithms are selected via the ``mode`` kwarg,
            but this may be deprecated in the future if we decide to implement a more
            flexible diagram building mechanism for CNF instances.

        """
        match mode:
            case "bintree":
                return self._build_diagram_bintree()
            case _:
                raise NotImplementedError(
                    f"Unknown diagram mode for CNFInstance: {mode!r}"
                )

    def contraction_wiring(
        self,
        *,
        mode: CNFDiagramMode = "bintree",
    ) -> Wiring:
        """
        Returns a wiring which can be used to contract the sequential composition
        of the :meth:`~CNFInstance.diagram` with an arbitrary specification of
        :meth:`~CNFInstance.inputs`.

        .. warning::

            See :meth:`CNFInstance.diagram` for possible deprecation of ``mode`` kwarg.

        """
        return (self.inputs(0) >> self.diagram(mode=mode)).flatten().wiring

    def _build_diagram_bintree(self) -> Diagram:
        builder = DiagramBuilder()
        builder.add_inputs(bits(self.num_vars))
        for clause in self.clauses:
            layer = [x - 1 if x > 0 else (not_ @ builder[-x - 1])[0] for x in clause]
            while (n := len(layer)) > 1:
                new_layer = [
                    (or_ @ builder[layer[2 * i : 2 * i + 2]])[0] for i in range(n // 2)
                ]
                if n % 2 == 1:
                    (new_layer[-1],) = or_ @ builder[new_layer[-1], layer[-1]]
                layer = new_layer
            assert len(layer) == 1
            bit_1 @ builder[layer[0]]
        return builder.diagram

    def __repr__(self) -> str:
        attrs: list[str] = [
            f"{self.num_vars} vars",
            f"{len(self.clauses)} clauses",
        ]
        ks = {len(clause) for clause in self.clauses}
        if len(ks) == 1:
            attrs.append(f"k={ks.pop()}")
        else:
            attrs.extend(f"k in [{min(ks)}..{max(ks)}]")
        return f"<CNFInstance {id(self):#x}: {", ".join(attrs)}>"
