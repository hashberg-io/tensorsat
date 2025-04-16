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
from z3 import Solver, sat

def benchmark_formulae() -> None:
    solver = Solver()
    with open("examples/counterexample3.txt", "r") as f:
        num_sat = 0
        num_unsat = 0
        time_total = 0
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
                    draw_diagram(
                        d,
                        node_label={ "box": binop_labels, "wire": str },
                        layout="circuit",
                        simplify_wires=False,
                        figsize=(8, 4)
                    )
                    d_sat = d.flatten()
                    # print("Done.")
                    cnf_sat_contraction = CotengraContraction(FinRel, d.wiring)
                    result = bool(cnf_sat_contraction.contract(d))
                    print("TensorSat: {}".format(result))

                    start_time = time.time()
                    solver.add(f)
                    if solver.check() == sat:
                        print("sat")
                        if not result:
                            num_errors += 1
                            print(line)
                        num_sat += 1
                    else:
                        print("unsat")
                        if result:
                            num_errors += 1
                            print(line)
                        num_unsat += 1

                    # print(f"Num errors: {num_errors}")
                    solver.pop()
                    time_taken = (time.time() - start_time) * 1000
                    time_total += time_taken
                    # print("{} ms".format(time_taken))
                else:
                    raise ParseError("Could not read formula.")

                i += 1

        # print(f"{num_sat} sat, {num_unsat} unsatisfiable.")
        # print("Total time: {} seconds".format(time_total / 1000))

def main() -> None:
    benchmark_formulae()
