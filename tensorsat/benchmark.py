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

def benchmark_formulae() -> None:
    solver = Solver()
    with open("examples/all_shuffled.txt", "r") as f:
        num_sat = 0
        num_unsat = 0
        time_total_z3 = 0
        time_total_tensorsat = 0
        num_errors = 0
        i = 0
        for line in f:
            if len(line) > 1:
                phi: Formula | None = eval(line)
                if phi:
                    # print(line)
                    # print("Formula has the variables: {}.".format(variables(phi)))
                    solver.push()
                    f = to_z3_formula(phi)

                    # print("Constructing diagram...")
                    diag = diagram_of_formula(phi)
                    d = diag.diagram()
                    # draw_diagram(
                    #     d,
                    #     node_label={ "box": binop_labels, "wire": str },
                    #     layout="circuit",
                    #     simplify_wires=False,
                    #     figsize=(8, 4)
                    # )
                    d_sat = d.flatten()
                    # print("Done.")
                    rgo = ctg.pathfinders.path_basic.RandomGreedyOptimizer(max_repeats=512, costmod=(0.1, 2.0), temperature=(0.1, 0.3), simplify=True, parallel=True, accel=True)
                    cnf_sat_contraction = CotengraContraction(FinRel, d_sat.wiring, optimize=rgo)

                    start_time_tensorsat = time.time()
                    result = bool(cnf_sat_contraction.contract(d))
                    time_taken_tensorsat = (time.time() - start_time_tensorsat) * 1000
                    time_total_tensorsat += time_taken_tensorsat

                    start_time_z3 = time.time()
                    solver.add(f)
                    if solver.check() == sat:
                        # print("z3: sat")
                        assert(result)
                        num_sat += 1
                    else:
                        # print("z3: unsat")
                        if result:
                            print(line)
                        assert(not result)
                        num_unsat += 1

                    # print(f"{num_sat} sat, {num_unsat} unsatisfiable.")

                    # print(f"Num errors: {num_errors}")
                    solver.pop()
                    time_taken_z3 = (time.time() - start_time_z3) * 1000
                    time_total_z3 += time_taken_z3
                    # print("{} ms".format(time_taken))
                else:
                    raise ParseError("Could not read formula.")

                i += 1

        print(f"{num_sat} sat, {num_unsat} unsatisfiable.")
        print("Z3 total time: {} seconds".format(time_total_z3 / 1000))
        print("TensorSat total time: {} seconds".format(time_total_tensorsat / 1000))

def main() -> None:
    benchmark_formulae()
