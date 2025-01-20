"""
Representation of propositional formulae along with basic machinery for CNF
transformation.
"""

from enum import Enum
from typing import (List, Optional)


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
        raise ValueError("Unknown expression sort")


class Formula:

    def __init__(self, op) -> None:
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

    @classmethod
    def set_subformula_of_unary_operation(self, phi):
        if arity(self.operation) == 1:
            self.subtrees[0] = phi
        else:
            raise ValueError("Arity mismatch")

    @classmethod
    def set_subformulae_of_binary_operation(self, phi, psi):
        if arity(self.operation) == 2:
            self.subtrees[0] = phi
            self.subtrees[1] = psi
        else:
            raise ValueError("Arity mismatch")

    @classmethod
    def subformula_of_unary_operation(self):
        if arity(self.operation) == 1:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    @classmethod
    def left_subformula_of_binary_operation(self):
        if arity(self.operation) == 2:
            return self.subtrees[0]
        else:
            raise ValueError("Arity mismatch")

    @classmethod
    def right_subformula_of_binary_operation(self):
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
    theta.set_subformulae_of_binary_operation(phi, psi)
    return theta


def unary_formula(op: Operation, phi: Formula) -> Formula:
    """Construct a formula with a unary operation."""
    psi = Formula(op)
    psi.set_subformula_of_unary_operation(phi)
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
    return binary_formula(Operation.IMPL, phi, psi)


def biimpl(phi: Formula, psi: Formula) -> Formula:
    return binary_formula(Operation.BIIMPL, phi, psi)


def show_operation(op: Operation) -> str:
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


def show_formula(phi: Formula) -> str:
    if arity(phi.operation) == 0:
        return f"(Var \"{phi.name}\")"
    elif arity(phi.operation) == 1:
        s1: str = show_operation(phi.operation)
        s2: str = show_formula(phi.subformula_of_unary_operation())
        return f"({s1} {s2})"
    else:
        assert arity(phi.operation) == 2
        sa: str = show_operation(phi.operation)
        sb: str = show_formula(phi.left_subformula_of_binary_operation())
        sc: str = show_formula(phi.right_subformula_of_binary_operation())
        return f"({sa} {sb} {sc})"


def eliminate_implications(phi: Formula) -> Formula:
    """Eliminate implications and bi-implications from a given formula."""
    if arity(phi.operation) == 0:
        return phi
    elif arity(phi.operation) == 1:
        if phi.operation == Operation.NEG:
            phi0 = phi.subformula_of_unary_operation()
            return neg(eliminate_implications(phi0))
        else:
            raise ValueError("Unknown logical operation")
    else:
        assert arity(phi.operation) == 2
        psi = phi.left_subformula_of_binary_operation()
        theta = phi.right_subformula_of_binary_operation()
        psi0 = eliminate_implications(psi)
        theta0 = eliminate_implications(theta)
        if phi.operation == Operation.IMPL:
            return disj(neg(psi0), theta0)
        elif phi.operation == Operation.BIIMPL:
            return eliminate_implications(conj(psi0, theta0))
        else:
            return binary_formula(phi.operation, psi0, theta0)


def dualize(op: Operation) -> Optional[Operation]:
    if op == Operation.CONJ:
        return Operation.DISJ
    elif op == Operation.DISJ:
        return Operation.CONJ
    else:
        return None


def demorganize(phi: Formula) -> Formula:
    """Applies the de Morgan laws repeatedly and removes double negations.
    """

    if arity(phi.operation) == 0:
        return phi
    elif arity(phi.operation) == 1:
        # Currently, negation is the only unary operation so the check below is
        # redundant. But this might change in the future so I'm keeping this
        # here.
        psi: Formula = phi.subformula_of_unary_operation()
        if phi.operation == Operation.NEG:
            if psi.operation == Operation.NEG:
                # This case means we have just seen a double-negation, which we
                # get rid of.
                return demorganize(psi.subformula_of_unary_operation())
            elif psi.operation in [Operation.CONJ, Operation.DISJ]:
                # This case means we have just seen a formula of the
                # form `Neg(BinOp(psi, theta))`.
                theta = psi.left_subformula_of_binary_operation()
                gamma = psi.right_subformula_of_binary_operation()
                op: Operation = dualize(psi.operation)
                assert op is not None
                theta0 = demorganize(neg(theta))
                gamma0 = demorganize(neg(gamma))
                return binary_formula(op, theta0, gamma0)
            else:
                # This case means we have just seen a formula of the form
                # `Neg(phi)`, where `phi` is neither of: a conjunction, a
                # disjunction, or a double negation. In this case, we simply do
                # a recursive call.
                return neg(demorganize(psi))
        else:
            raise ValueError("Unknown logical operation")
    else:
        assert arity(phi.operation) == 2
        op = phi.operation
        if op == Operation.IMPL or op == Operation.BIIMPL:
            raise ValueError("Formula contains implication.")
        elif op == Operation.CONJ or phi.operation == Operation.DISJ:
            # Redundant check in case other operations are added in the future.
            psi: Formula = phi.left_subformula_of_binary_operation()
            theta: Formula = phi.right_subformula_of_binary_operation()
            psi0 = demorganize(psi)
            theta0 = demorganize(theta)
            return binary_formula(phi.operation, psi0, theta0)
        else:
            raise ValueError("Unknown logical operation")


def merge(cnf1: Formula, cnf2: Formula) -> Formula:
    """Construct disjunctions of formulae in a way that preserves the
       CNF property (for formulae that are already in CNF).
    """
    if cnf1.operation == Operation.CONJ:
        cnf1a = cnf1.left_subformula_of_binary_operation()
        cnf1b = cnf1.right_subformula_of_binary_operation()
        return conj(merge(cnf1a, cnf2), merge(cnf1b, cnf2))
    elif cnf2.operation == Operation.CONJ:
        cnf2a = cnf2.left_subformula_of_binary_operation()
        cnf2b = cnf2.right_subformula_of_binary_operation()
        return conj(merge(cnf1, cnf2a), merge(cnf1, cnf2b))
    else:
        return disj(cnf1, cnf2)


def distribute(phi: Formula) -> Formula:
    """Pushes all disjunctions inside."""
    op = phi.operation
    if arity(op) == 0:
        return phi
    elif arity(op) == 1:
        return unary_formula(op, phi.subformula_of_unary_operation())
    elif arity(op) == 2:
        if op == Operation.CONJ:
            psi = phi.left_subformula_of_binary_operation()
            theta = phi.right_subformula_of_binary_operation()
            return conj(distribute(psi), distribute(theta))
        elif op == Operation.DISJ:
            psi = phi.left_subformula_of_binary_operation()
            theta = phi.right_subformula_of_binary_operation()
            return merge(distribute(psi), distribute(theta))
        else:
            raise ValueError("Unexpected logical operation in `distribute`.")


def to_cnf(phi: Formula) -> Formula:
    """Convert a formula into CNF form."""
    return distribute(demorganize(eliminate_implications(phi)))


def is_negated_variable(phi: Formula) -> bool:
    if phi.operation == Operation.NEG:
        psi = phi.subformula_of_unary_operation()
        return psi.operation == Operation.VAR
    else:
        return False


def literal_list(phi: Formula) -> list[tuple[bool, str]]:
    """Takes a formula consisting of disjunctions in literals and returns a
       representation of it is a clause."""
    if phi.operation == Operation.VAR:
        return [(True, phi.name)]
    elif is_negated_variable(phi):
        return [(False, phi.subformula_of_unary_operation().name)]
    else:
        assert phi.operation == Operation.DISJ
        psi = phi.left_subformula_of_binary_operation()
        theta = phi.right_subformula_of_binary_operation()
        return literal_list(psi) + literal_list(theta)


falsum: Formula = Formula(Operation.FALSUM)


var_p = variable("p")
var_q = variable("q")
lem: Formula = disj(variable("p"), neg(variable("p")))
meet_conj: Formula = impl(conj(var_p, var_q), conj(var_q, var_p))

print(show_formula(eliminate_implications(meet_conj)))
print(show_formula(demorganize(eliminate_implications(meet_conj))))
print(show_formula(distribute(demorganize(eliminate_implications(meet_conj)))))
