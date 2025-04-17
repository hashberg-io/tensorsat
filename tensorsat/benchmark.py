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
from concurrent.futures import ProcessPoolExecutor

lines = None
diagrams = None

def benchmark_formulae(timeout: float) -> None:
    time_total_z3 = 0.0
    time_total_tensorsat = 0.0

    # Cactus plot data for TensorSat
    num_sat_tensorsat = 0
    num_unsat_tensorsat = 0
    for d in diagrams:
        if time_total_tensorsat > timeout:
            break

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

    num_solved_tensorsat = num_sat_tensorsat + num_unsat_tensorsat
    return (timeout, num_solved_tensorsat)


def main() -> None:
    bfile = open("examples/all_shuffled.txt", "r")
    global lines
    global diagrams
    lines = list([ line for line in list(bfile.read().split("\n")) if len(line) > 0 ])
    formulae = [ eval(line) for line in lines ]
    diagrams = [ diagram_of_formula(phi).diagram() for phi in formulae ]
    print(len(lines))
    bfile.close()

    intervals = [ 30.0 * i for i in range(1, 180) ]

    for i, timeout in enumerate(intervals):
        sys.stderr.write("%{:2f} complete\n".format((i / 180) * 100))
        print(benchmark_formulae(timeout))
