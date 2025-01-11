"""
Utilities to construct tensor networks for SAT problems.
"""

from __future__ import annotations
from collections.abc import Sequence
from math import comb
import re
from typing import Self, TypeAlias
import numpy as np
from _tensorsat import CircuitBuilder, RelNet
from _bincirc import not_, or_, bit_unk

# CNFInstance: TypeAlias = tuple[tuple[int, ...], ...]
# """
# Type alias for a SAT instance in CNF form, as a sequence of clauses.
# """

Clause: TypeAlias = tuple[int, ...]
"""
A SAT clause, as a tuple of non-zero integers representing the literals in the clause,
with the integer sign determining whether the literal is positive or negative.
"""


class CNFInstance:
    """A SAT instance in CNF form."""

    @classmethod
    def random(
        cls, k: int, n: int, m: int, *, rng: int | np.random.Generator | None = None
    ) -> Self:
        """
        Generates a random SAT instance.
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
            vs = rng.choice(range(n), size=k, replace=False)
            signs = rng.choice(range(2), size=k)
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
        """Create a SAT instance from a DIMACS formatted string."""
        dimacs = dimacs.replace("\r\n", "\n").replace("\n", " ").strip()
        start_match = re.compile(r"p cnf ([0-9]+) ([0-9]+)").match(dimacs)
        if not start_match:
            raise ValueError(
                "DIMACS code must start with 'p cnf <num vars> <num clauses>'."
            )
        num_vars, num_clauses = map(int, start_match.groups())
        clauses = tuple(
            tuple(map(int, frag.strip().split(" ")))
            for frag in dimacs[start_match.end() :].split("0")
            if frag.strip()
        )
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
        """Create a SAT instance from a number of vars and a sequence of clauses."""
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
        num_clauses = len(self.__clauses)
        lines = [
            f"p cnf {self.num_vars} {num_clauses}",
            *(f"{' '.join(map(str, clause))} 0" for clause in self.__clauses),
        ]
        return "\n".join(lines)

    def network(self) -> RelNet:
        """Construct a tensor network for the SAT instance."""
        num_vars, clauses = self.__num_vars, self.__clauses
        circ = CircuitBuilder([2] * num_vars)
        for clause in clauses:
            layer = [
                x - 1 if x > 0 else circ.add_gate(not_, [-x - 1])[0] for x in clause
            ]
            while (n := len(layer)) > 1:
                new_layer = [
                    circ.add_gate(or_, layer[2 * i : 2 * i + 2])[0]
                    for i in range(n // 2)
                ]
                if n % 2 == 1:
                    new_layer[-1] = circ.add_gate(or_, [new_layer[-1], layer[-1]])[0]
                layer = new_layer
            circ.add_gate(bit_unk, layer)
        return circ.network
