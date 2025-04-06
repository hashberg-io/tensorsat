from __future__ import annotations
from collections import deque
from typing import List
from enum import Enum
from z3 import Bool, And, Or, BoolRef, Not, Implies
from tensorsat.diagrams import Diagram, DiagramBuilder
from tensorsat.lib.bincirc import bits, bit, or_, and_, bit_0, bit_1, impl_, biimpl_, not_
from tensorsat.contractions.cotengra import CotengraContraction
from tensorsat.lang.fin_rel import FinRel
from tensorsat.lib.sat import *

class Operation(Enum):
    VAR = 1     # variable
    NEG = 2     # negation
    CONJ = 3    # conjunction
    DISJ = 4    # disjunction
    IMPL = 5    # implication
    BIIMPL = 6  # bi-implication
    TRUTH = 7   # top
    FALSUM = 8  # bottom


def arity(op: Operation) -> int:
    """The arity of a given operation."""
    if op == Operation.VAR:
        return 0
    if op == Operation.TRUTH or op == Operation.FALSUM:
        return 0
    elif op == Operation.NEG:
        return 1
    elif op == Operation.CONJ or op == Operation.DISJ:
        return 2
    elif op == Operation.IMPL or op == Operation.BIIMPL:
        return 2
    else:
        raise ValueError("Unknown operation")


class Formula:

    def __init__(self, op: Operation) -> None:
        self.operation: Operation = op
        self.name: str | None = None
        self.subtrees: List[Formula | None] = []

        if arity(op) == 0:
            if op == Operation.VAR:
                self.name = None
                self.subtrees = []
            elif op == Operation.TRUTH or op == Operation.FALSUM:
                self.name = None
                self.subtrees = []
        elif arity(op) == 1:
            self.subtrees = [None]
        elif arity(op) == 2:
            self.subtrees = [None, None]

    def set_subformula_of_unary_formula(self, phi: Formula) -> None:
        if arity(self.operation) == 1:
            self.subtrees[0] = phi
        else:
            raise ValueError("Arity mismatch")

    def set_subformulae_of_binary_formula(self, phi: Formula, psi: Formula) -> None:
        if arity(self.operation) == 2:
            self.subtrees[0] = phi
            self.subtrees[1] = psi
        else:
            raise ValueError("Arity mismatch")

    def subformula_of_unary_formula(self) -> Formula | None:
        if arity(self.operation) == 1:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    def left_subformula_of_binary_formula(self) -> Formula | None:
        if arity(self.operation) == 2:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    def right_subformula_of_binary_formula(self) -> Formula | None:
        if arity(self.operation) == 2:
            return self.subtrees[1]
        else:
            raise ValueError("Arity mismatch")

def variable(s: str) -> Formula:
    """Construct a variable with name `s`."""
    phi = Formula(Operation.VAR)
    phi.name = s
    return phi


def binary_formula(op: Operation, phi: Formula, psi: Formula) -> Formula:
    """Construct a formula with a binary operation."""
    theta = Formula(op)
    theta.set_subformulae_of_binary_formula(phi, psi)
    return theta


def unary_formula(op: Operation, phi: Formula) -> Formula:
    """Construct a formula with a unary operation."""
    psi = Formula(op)
    psi.set_subformula_of_unary_formula(phi)
    return psi


def neg(phi: Formula) -> Formula:
    """Construct the negation of a formula `phi`."""
    return unary_formula(Operation.NEG, phi)


def conj(phi: Formula, psi: Formula) -> Formula:
    """Conjunction of two formulae `phi` and `psi`."""
    return binary_formula(Operation.CONJ, phi, psi)


def disj(phi: Formula, psi: Formula) -> Formula:
    """Disjunction of two formulae `phi` and `psi`."""
    return binary_formula(Operation.DISJ, phi, psi)


def impl(phi: Formula, psi: Formula) -> Formula:
    """Implication of two formulae `phi` and `psi`."""
    return binary_formula(Operation.IMPL, phi, psi)


def biimpl(phi: Formula, psi: Formula) -> Formula:
    """Bi-implication of two formulae `phi` and `psi`."""
    return binary_formula(Operation.BIIMPL, phi, psi)


def show_operation(op: Operation) -> str:
    """String representation of a given operation `op`."""
    if op == Operation.VAR:
        return "Var"
    elif op == Operation.NEG:
        return "Neg"
    elif op == Operation.TRUTH:
        return "Truth"
    elif op == Operation.FALSUM:
        return "Falsum"
    elif op == Operation.IMPL:
        return "Impl"
    elif op == Operation.BIIMPL:
        return "Biimpl"
    elif op == Operation.CONJ:
        return "Conj"
    elif op == Operation.DISJ:
        return "Disj"

# def show_formula(phi: Formula) -> str:
#     """String representation of a given formula `phi`."""
#     if arity(phi.operation) == 0:
#         return f"(Var \"{phi.name}\")"
#     elif arity(phi.operation) == 1:
#         s1: str = show_operation(phi.operation)
#         s2: str = show_formula(phi.subformula_of_unary_formula())
#         return f"({s1} {s2})"
#     else:
#         assert arity(phi.operation) == 2
#         sa: str = show_operation(phi.operation)
#         sb: str = show_formula(phi.left_subformula_of_binary_formula())
#         sc: str = show_formula(phi.right_subformula_of_binary_formula())
#         return f"({sa} {sb} {sc})"

def size(phi: Formula) -> int:
    if arity(phi.operation) == 0:
        return 1
    elif arity(phi.operation) == 1:
        psi = phi.subformula_of_unary_formula()
        return 1 + size(psi)
    else:
        psi = phi.left_subformula_of_binary_formula()
        theta = phi.right_subformula_of_binary_formula()
        return 1 + size(psi) + size(theta)

def to_z3_formula(phi: Formula) -> BoolRef:
    if phi.operation == Operation.VAR:
        return Bool(phi.name)
    elif phi.operation == Operation.NEG:
        if phi.subformula_of_unary_formula() is not None:
            f: BoolRef = to_z3_formula(phi.subformula_of_unary_formula())
            return Not(f)
    elif phi.operation == Operation.CONJ:
        f1: BoolRef = to_z3_formula(phi.left_subformula_of_binary_formula())
        f2: BoolRef = to_z3_formula(phi.right_subformula_of_binary_formula())
        return And(f1, f2)
    elif phi.operation == Operation.DISJ:
        f1: BoolRef = to_z3_formula(phi.left_subformula_of_binary_formula())
        f2: BoolRef = to_z3_formula(phi.right_subformula_of_binary_formula())
        return Or(f1, f2)
    elif phi.operation == Operation.IMPL:
        f1: BoolRef = to_z3_formula(phi.left_subformula_of_binary_formula())
        f2: BoolRef = to_z3_formula(phi.right_subformula_of_binary_formula())
        return Or(f1, f2)
    else:
        f1: BoolRef = to_z3_formula(phi.left_subformula_of_binary_formula())
        f2: BoolRef = to_z3_formula(phi.right_subformula_of_binary_formula())
        return And(Implies(f1, f2), Implies(f2, f1))

def variables(phi: Formula) -> set[str]:
    if phi.operation == Operation.VAR:
        return set([phi.name])
    elif phi.operation == Operation.NEG:
        return variables(phi.subformula_of_unary_formula())
    elif phi.operation == Operation.CONJ:
        return variables(phi.left_subformula_of_binary_formula()).union(variables(phi.right_subformula_of_binary_formula()))
    elif phi.operation == Operation.DISJ:
        return variables(phi.left_subformula_of_binary_formula()).union(variables(phi.right_subformula_of_binary_formula()))
    elif phi.operation == Operation.IMPL:
        return variables(phi.left_subformula_of_binary_formula()).union(variables(phi.right_subformula_of_binary_formula()))
    elif phi.operation == Operation.BIIMPL:
        return variables(phi.left_subformula_of_binary_formula()).union(variables(phi.right_subformula_of_binary_formula()))

def diagram_of_formula(phi: Formula) -> DiagramBuilder:
    diag = DiagramBuilder()
    vs: list[str] = sorted(list(variables(phi)))
    num_vars: int = len(vs)
    inputs = diag.add_inputs(bits(num_vars))

    def traversal_rec(psi: Formula) -> int:
        if psi.operation == Operation.VAR:
            i = vs.index(psi.name)
            return inputs[i]
        elif psi.operation == Operation.NEG:
            a = traversal_rec(psi.subformula_of_unary_formula())
            out, = not_ @ diag[a]
            return out
        elif psi.operation == Operation.CONJ:
            child_left = psi.left_subformula_of_binary_formula()
            child_right = psi.right_subformula_of_binary_formula()
            a = traversal_rec(child_left)
            b = traversal_rec(child_right)
            out, = and_ @ diag[a, b]
            return out
        elif psi.operation == Operation.DISJ:
            child_left = psi.left_subformula_of_binary_formula()
            child_right = psi.right_subformula_of_binary_formula()
            a = traversal_rec(child_left)
            b = traversal_rec(child_right)
            out, = or_ @ diag[a, b]
            return out
        elif psi.operation == Operation.IMPL:
            child_left = psi.left_subformula_of_binary_formula()
            child_right = psi.right_subformula_of_binary_formula()
            a = traversal_rec(child_left)
            b = traversal_rec(child_right)
            out, = impl_ @ diag[a, b]
            return out
        elif psi.operation == Operation.BIIMPL:
            child_left = psi.left_subformula_of_binary_formula()
            child_right = psi.right_subformula_of_binary_formula()
            a = traversal_rec(child_left)
            b = traversal_rec(child_right)
            out, = biimpl_ @ diag[a, b]
            return out

    out = traversal_rec(phi)
    diag.add_outputs((out,))

    return diag

# def contract_diagram(diag: DiagramBuilder) -> bool:
#     d = diag.diagram()
#     sat_diagram = d.flatten()
#     sat_contraction = CotengraContraction(FinRel, sat_diagram.wiring)

#     return bool(sat_contraction.contract(d, progbar=False))

if __name__ == "__main__":
    with open("examples/formulae/formulae.txt", "r") as f:
        phi: Formula | None = None
        for line in f:
            phi = eval(line)
            d = diagram_of_formula(phi)
            print(d)
