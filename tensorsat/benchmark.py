from tensorsat.lib.bincirc import binop_labels
from tensorsat.lib.sat import *
from tensorsat.lang.fin_rel import FinRel
from tensorsat.contractions.cotengra import CotengraContraction
from tensorsat.contractions.simple import SimpleContraction
from tensorsat.lib.formula import *
from tensorsat.viz import draw_diagram
import numpy as np
import lzma
import sys
import time
import cotengra as ctg
from z3 import Solver, sat

def benchmark_formulae(timeout: float) -> None:
    solver = Solver()
    args = sys.argv

    with open("examples/all_shuffled.txt", "r") as bfile:
        num_sat_z3 = 0
        num_unsat_z3 = 0
        time_total_z3 = 0.0
        time_total_tensorsat = 0.0

        ## Cactus plot data for Z3
        for line in bfile:
            if time_total_z3 > timeout:
                break

            if len(line) > 1:
                phi: Formula | None = eval(line)
                if phi:
                    # print(line)
                    # print("Formula has the variables: {}.".format(variables(phi)))
                    solver.push()
                    f = to_z3_formula(phi)

                    start_time = time.time()
                    solver.add(f)
                    if solver.check() == sat:
                        num_sat_z3 += 1
                    else:
                        num_unsat_z3 += 1
                    time_taken_z3 = (time.time() - start_time) * 1000
                    time_total_z3 += time_taken_z3

                    # print(f"{num_sat_z3} sat, {num_unsat_z3} unsatisfiable.")

                    solver.pop()
                else:
                    raise ParseError("Could not read formula.")

        num_solved_z3 = num_sat_z3 + num_unsat_z3

        # Cactus plot data for TensorSat
        bfile.seek(0)
        num_sat_tensorsat = 0
        num_unsat_tensorsat = 0
        for line in bfile:
            if time_total_tensorsat > timeout:
                break

            if len(line) > 1:
                phi: Formula | None = eval(line)
                if phi:
                    diag = diagram_of_formula(phi)
                    d = diag.diagram()
                    d_sat = d.flatten()
                    rgo = ctg.pathfinders.path_basic.RandomGreedyOptimizer(max_repeats=1024, costmod=(0.1, 2.0), temperature=(0.1, 0.3), simplify=False, parallel=True, accel=True)

                    cnf_sat_contraction = CotengraContraction(FinRel, d_sat.wiring, optimize=rgo)

                    start_time_tensorsat = time.time()
                    result = bool(cnf_sat_contraction.contract(d))
                    time_taken_tensorsat = (time.time() - start_time_tensorsat) * 1000
                    time_total_tensorsat += time_taken_tensorsat

                    if result:
                        num_sat_tensorsat += 1
                    else:
                        num_unsat_tensorsat += 1

                    # print(f"{num_sat_tensorsat} sat, {num_unsat_tensorsat} unsatisfiable.")
                else:
                    raise ParseError("Could not read formula.")

        num_solved_tensorsat = num_sat_tensorsat + num_unsat_tensorsat
        print(f"{timeout}, {num_solved_z3}, {num_solved_tensorsat}")


def main() -> None:
    for i in range(1, 500):
        benchmark_formulae(25.0 * i)
