from tensorsat.lib.sat import *

def test_dimacs_parser_unit1():
    dimacs_unit1 = ""
    dimacs_unit2 = ""
    with open("examples/kissat/cnf/unit1.cnf") as file_unit1:
        dimacs_unit1 = file_unit1.read()

    cnf_unit1: CNFInstance = CNFInstance.from_dimacs(dimacs_unit1)
    expected_unit1: list[Clause] = [(1,), (-1,)]

    assert set(cnf_unit1.clauses) == set(expected_unit1)

def test_dimacs_parser_unit2():
    dimacs_unit2 = ""
    with open("examples/kissat/cnf/unit2.cnf") as file_unit2:
        dimacs_unit2 = file_unit2.read()

    cnf_unit2: CNFInstance = CNFInstance.from_dimacs(dimacs_unit2)
    expected_unit2: list[Clause] = [(1,), (-1,)]

    assert set(cnf_unit2.clauses) == set(expected_unit2)

def test_dimacs_parser_unit3():
    dimacs = ""
    with open("examples/kissat/cnf/unit3.cnf") as f:
        dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)
    expected: list[Clause] = [(2,), (-2,)]

    assert set(cnf.clauses) == set(expected)

def test_dimacs_parser_unit4():
    dimacs = ""
    with open("examples/kissat/cnf/unit4.cnf") as f:
        dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)
    expected: list[Clause] = [(7,),
                              (-7, -1),
                              (-7, 1, 3),(-7, 1, 5),
                              (-7, -3, -5)]

    assert set(cnf.clauses) == set(expected)

def test_dimacs_parser_unit5():
    dimacs = ""
    with open("examples/kissat/cnf/unit5.cnf") as f:
        dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)
    expected: tuple[Clause] = (((-1, 2,), (1, 2,), (-2,)))

    assert cnf.clauses == expected

def test_dimacs_parser_bin1():
    dimacs = ""
    with open("examples/kissat/cnf/bin1.cnf") as f:
        dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)
    expected: tuple[Clause] = ((1, 2,),)

    assert cnf.clauses == expected
