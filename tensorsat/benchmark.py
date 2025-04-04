from tensorsat.lib.bincirc import and_
from tensorsat.lib.sat import *
from tensorsat.lang.fin_rel import FinRel
from tensorsat.contractions.cotengra import CotengraContraction
import numpy as np
import lzma
import sys


def main() -> None:
    if len(sys.argv) <= 1:
        print("Please provide a file name as input.")
        sys.exit(1)

    filename: str = sys.argv[1]

    print(f"Filename is {filename}")

    dimacs: str = ""

    if filename.endswith(".xz"):
        with lzma.open(filename, mode="rt") as f:
            dimacs = f.read()
    else:
        with open(filename, mode="r") as f:
            dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)

    cnf_diagram = cnf.diagram()
    cnf_sat_diagram = (cnf.inputs(None)>>cnf_diagram).flatten()
    cnf_sat_contraction = CotengraContraction(FinRel, cnf_sat_diagram.wiring)

    result: bool = cnf_sat_contraction.contract(cnf_sat_diagram, progbar=False, autojit=False)

    if result:
        print("sat")
        sys.exit(10)
    else:
        print("unsat")
        sys.exit(20)

    # print(cnf_solutions)

    # print(cnf)

    # print("This is tensorsat")
    # print("-----------------\n")

    # print("Representation of logical conjunction as a tensor.")
    # print(and_.tensor)
    # print("\n")

    # and_tensor = and_.tensor

    # print({
    #     point
    #     for point in np.ndindex(and_.tensor.shape)
    #     if and_.tensor[point]
    # })

    # print("\n")
    # print(and_tensor.reshape(4, 2))
