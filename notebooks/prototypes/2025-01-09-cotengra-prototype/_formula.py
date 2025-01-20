"""
Representation of propositional logical formulae along with machinery for CNF transformation.
"""

from enum import Enum
from typing import (List, Optional)


class Operation(Enum):
    VAR    = 1  ## variable
    NEG    = 2  ## negation
    CONJ   = 3  ## conjunction
    DISJ   = 4  ## disjunction
    IMPL   = 5  ## implication
    BIIMPL = 6  ## bi-implication
    TRUTH  = 7  ## top
    FALSUM = 8  ## bottom


def arity(op : Operation) -> int:
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
        raise ValueError("Unknown expression sort")


class Formula:

    def __init__(self, sort) -> None:
         self.sort     : Operation  = sort
         self.name     : str | None = None
         self.subtrees : List[Formula | None] = []

         if arity(sort) == 0:
             if sort == Operation.VAR:
                self.name     = None
                self.subtrees = []
             elif sort == Operation.TRUTH or sort == Operation.FALSUM:
                self.name     = None
                self.subtrees = []
         elif arity(sort) == 1:
             self.subtrees = [None]
         elif arity(sort) == 2:
             self.subtrees = [None, None]

    def set_subformula_of_unary_operation(self, e):
        if arity(self.sort) == 1:
            self.subtrees[0] = e
        else:
            raise ValueError("Arity mismatch")

    def set_subformulae_of_binary_operation(self, phi, psi):
        if arity(self.sort) == 2:
            self.subtrees[0] = phi
            self.subtrees[1] = psi
        else:
            raise ValueError("Arity mismatch")

    def subformula_of_unary_operation(self):
        if arity(self.sort) == 1:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    def left_subformula_of_binary_operation(self):
        if arity(self.sort) == 2:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    def right_subformula_of_binary_operation(self):
        if arity(self.sort) == 2:
            return self.subtrees[1]
        else:
            raise ValueError("Arity mismatch")


def variable(s : str) -> Formula:
    """Construct a formula consisting only of a variable identified with the name `s`."""
    phi      = Formula(Operation.VAR)
    phi.name = s
    return phi


def binary_expression(op : Operation, phi : Formula, psi : Formula) -> Formula:
    """Construct a binary formula consisting of the application of `op` to `phi` and `psi`."""
    theta = Formula(op)
    theta.set_subformulae_of_binary_operation(phi, psi)
    return theta


def unary_formula(op : Operation, phi : Formula) -> Formula:
    psi = Formula(op)
    psi.set_subformula_of_unary_operation(phi)
    return psi


def neg(phi : Formula) -> Formula:
    """Construct the negation of a formula `phi`."""
    return unary_formula(Operation.NEG, phi)


def conj(phi : Formula, psi : Formula) -> Formula:
    """Shorthand for the conjunction of two formulae `phi` and `psi`."""
    return binary_expression(Operation.CONJ, phi, psi)


def disj(e1 : Formula, e2 : Formula) -> Formula:
    """Shorthand for constructing a disjunction."""
    return binary_expression(Operation.DISJ, e1, e2)


falsum : Formula = Formula(Operation.FALSUM)

def impl(phi : Formula, psi : Formula) -> Formula:
    return binary_expression(Operation.IMPL, phi, psi)


def biimpl(phi : Formula, psi : Formula) -> Formula:
    return binary_expression(Operation.BIIMPL, phi, psi)


def show_operation(op : Operation) -> str:
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


def show_formula(phi : Formula) -> str:
    if arity(phi.sort) == 0:
        return f"(Var \"{phi.name}\")"
    elif arity(phi.sort) == 1:
        s1 : str = show_operation(phi.sort)
        s2 : str = show_formula(phi.subformula_of_unary_operation())
        return f"({s1} {s2})"
    else:
        assert(arity(phi.sort) == 2)
        sa : str = show_operation(phi.sort)
        sb : str = show_formula(phi.left_subformula_of_binary_operation())
        sc : str = show_formula(phi.right_subformula_of_binary_operation())
        return f"({sa} {sb} {sc})"


def eliminate_implications(phi : Formula) -> Formula:
    """Eliminate implications and bi-implications from a given formula `phi` by repeatedly applying
       the identity `p => q <-> ¬p ∨ q`."""
    if arity(phi.sort) == 0:
        return phi
    elif arity(phi.sort) == 1:
        if phi.sort == Operation.NEG:
            return neg(eliminate_implications(phi.subformula_of_unary_operation()))
        else:
            raise ValueError("Unknown logical operation")
    else:
        assert(arity(phi.sort) == 2)
        psi    = phi.left_subformula_of_binary_operation()
        theta  = phi.right_subformula_of_binary_operation()
        psi0   = eliminate_implications(psi)
        theta0 = eliminate_implications(theta)
        if phi.sort == Operation.IMPL:
            return disj(neg(psi0), theta0)
        elif phi.sort == Operation.BIIMPL:
            return eliminate_implications(conj(psi0, theta0))
        else:
            return binary_expression(phi.sort, psi0, theta0)


def dualize(op : Operation) -> Optional[Operation]:
    if op == Operation.CONJ:
        return Operation.DISJ
    elif op == Operation.DISJ:
        return Operation.CONJ
    else:
        return None

def demorganize(phi : Formula) -> Formula:
    """Transforms a formula by pushing all negations inside through the repeated application of the de Morgan laws.
       N.B. Should not be used on formulae containing implications."""

    if arity(phi.sort) == 0:
        return phi
    elif arity(phi.sort) == 1:
        ## Currently, negation is the only unary operation so the check below is redundant. But this might change
        ## in the future.
        psi: Formula = phi.subformula_of_unary_operation()
        if phi.sort == Operation.NEG:
            if psi.sort == Operation.NEG:
                ## This case means we have just seen a double-negation, which we get rid of.
                return demorganize(psi.subformula_of_unary_operation())
            elif psi.sort in [Operation.CONJ, Operation.DISJ]:
                ## This case entails having seen a formula of the form `Neg(BinOp(psi, theta))`.
                theta = phi.subformula_of_unary_operation().left_subformula_of_binary_operation()
                gamma = phi.subformula_of_unary_operation().right_subformula_of_binary_operation()
                op    = dualize(psi.sort)
                assert(op is not None)
                return binary_expression(op, demorganize(neg(theta)), demorganize(neg(gamma)))
            else:
                ## Reaching this case means having seen a formula of the form `Neg(phi)`, where phi is neither
                ## of a conjunction, a disjunction, or a double negation. In this case, we simply do a recursive
                ## call.
                return neg(demorganize(psi))
        else:
            raise ValueError("Unknown logical operation")
    else:
        assert(arity(phi.sort) == 2)
        if phi.sort == Operation.IMPL or phi.sort == Operation.BIIMPL:
            raise ValueError("The `demorganize` function should not be used on formulae containing implications or bi-implications.")
        elif phi.sort == Operation.CONJ or phi.sort == Operation.DISJ:
            ## redundant check in case other operations are added in the future
            psi   : Formula = phi.left_subformula_of_binary_operation()
            theta : Formula = phi.right_subformula_of_binary_operation()
            return binary_expression(phi.sort, demorganize(psi), demorganize(theta))
        else:
            raise ValueError("Unknown logical operation")


def merge(cnf1 : Formula, cnf2 : Formula) -> Formula:
    """Construct disjunctions of formulae in a way that preserves the CNF-property (for formulae that are already in CNF)."""
    if cnf1.sort == Operation.CONJ:
        cnf1a = cnf1.left_subformula_of_binary_operation()
        cnf1b = cnf1.right_subformula_of_binary_operation()
        return conj(merge(cnf1a, cnf2), merge(cnf1b, cnf2))
    elif cnf2.sort == Operation.CONJ:
        cnf2a = cnf2.left_subformula_of_binary_operation()
        cnf2b = cnf2.right_subformula_of_binary_operation()
        return conj(merge(cnf1, cnf2a), merge(cnf1, cnf2b))
    else:
        return disj(cnf1, cnf2)

def distribute(phi : Formula) -> Formula:
    """Pushes all disjunctions inside by repeatedly applying the distributivity law."""
    if arity(phi.sort) == 0:
        return phi
    elif arity(phi.sort) == 1:
        return unary_formula(phi.sort, phi.subformula_of_unary_operation())
    elif arity(phi.sort) == 2:
        if phi.sort == Operation.CONJ:
            psi   = phi.left_subformula_of_binary_operation()
            theta = phi.right_subformula_of_binary_operation()
            return conj(distribute(psi), distribute(theta))
        elif phi.sort == Operation.DISJ:
            psi   = phi.left_subformula_of_binary_operation()
            theta = phi.right_subformula_of_binary_operation()
            return merge(distribute(psi), distribute(theta))
        else:
            raise ValueError("Unexpected logical operation in `distribute`.")


def to_cnf(phi : Formula) -> Formula:
    return distribute(demorganize(eliminate_implications(phi)))

lem       : Formula = disj(variable("p"), neg(variable("p")))
meet_conj : Formula = impl(conj(variable("p"), variable("q")), conj(variable("q"), variable("p")))

print(show_formula(eliminate_implications(meet_conj)))
print(show_formula(demorganize(eliminate_implications(meet_conj))))
print(show_formula(distribute(demorganize(eliminate_implications(meet_conj)))))