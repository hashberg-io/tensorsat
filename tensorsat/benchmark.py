# from tensorsat.lib.bincirc import and_
from tensorsat.lib.sat import *
from tensorsat.lang.fin_rel import FinRel
from tensorsat.contractions.cotengra import CotengraContraction
from tensorsat.lib.formula import *
import numpy as np
import lzma
import sys
from z3 import Solver, sat

def benchmark_formulae() -> None:
    solver = Solver()
    with open("examples/formulae/formulae.txt", "r") as f:
        num_sat = 0
        num_unsat = 0
        for line in f:
            if len(line) > 1:
                phi: Formula | None = eval(line)
                if phi:
                    solver.push()
                    solver.add(to_z3_formula(phi))

                    if solver.check() == sat:
                        # print("sat")
                        num_sat += 1
                    else:
                        # print("unsat")
                        # print(line)
                        num_unsat += 1

                    solver.pop()
                else:
                    raise ParseError("Could not read formula.")

        print(f"{num_sat} sat, {num_unsat} unsatisfiable.")

def main() -> None:
    benchmark_formulae()
