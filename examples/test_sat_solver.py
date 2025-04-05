from tensorsat.lib.solver import *

def check_sat_file(filename: str) -> bool:
    dimacs = ""
    with open(f"examples/kissat/cnf/{filename}.cnf") as f:
        dimacs = f.read()

    cnf: CNFInstance = CNFInstance.from_dimacs(dimacs)

    return is_satisfiable(cnf)


def test_example_def1() -> None:
    is_sat: bool = check_sat_file("def1")
    assert(not is_sat)

def test_example_congr1() -> None:
    is_sat: bool = check_sat_file("congr1")
    assert(not is_sat)

def test_example_congr2() -> None:
    is_sat: bool = check_sat_file("congr2")
    assert(not is_sat)

def test_example_congr3() -> None:
    is_sat: bool = check_sat_file("congr3")
    assert(not is_sat)

def test_example_congr4() -> None:
    is_sat: bool = check_sat_file("congr4")
    assert(is_sat)

def test_example_congr5() -> None:
    is_sat: bool = check_sat_file("congr5")
    assert(not is_sat)

def test_example_congr6() -> None:
    is_sat: bool = check_sat_file("congr6")
    assert(not is_sat)

def test_example_congr7() -> None:
    is_sat: bool = check_sat_file("congr7")
    assert(is_sat)

def test_example_xor0() -> None:
    is_sat: bool = check_sat_file("xor0")
    assert(is_sat)

def test_example_xor1() -> None:
    is_sat: bool = check_sat_file("xor1")
    assert(is_sat)

def test_example_xor2() -> None:
    is_sat: bool = check_sat_file("xor2")
    assert(is_sat)

def test_example_xor3() -> None:
    is_sat: bool = check_sat_file("xor3")
    assert(is_sat)

def test_example_xor4() -> None:
    is_sat: bool = check_sat_file("xor4")
    assert(is_sat)

def test_example_xor5() -> None:
    is_sat: bool = check_sat_file("xor5")
    assert(not is_sat)

def test_example_ph2() -> None:
    is_sat: bool = check_sat_file("ph2")
    assert(not is_sat)

def test_example_full2() -> None:
    is_sat: bool = check_sat_file("full2")
    assert(not is_sat)

def test_example_full3() -> None:
    is_sat: bool = check_sat_file("full3")
    assert(not is_sat)

def test_example_full4() -> None:
    is_sat: bool = check_sat_file("full4")
    assert(not is_sat)

def test_example_strash1() -> None:
    is_sat: bool = check_sat_file("strash1")
    assert(not is_sat)

def test_example_strash3() -> None:
    is_sat: bool = check_sat_file("strash3")
    assert(is_sat)

def test_example_tieshirt() -> None:
    is_sat: bool = check_sat_file("tieshirt")
    assert(is_sat)

def test_example_prime4() -> None:
    is_sat: bool = check_sat_file("prime4")
    assert(is_sat)

def test_example_unit1() -> None:
    is_sat: bool = check_sat_file("unit1")
    assert(not is_sat)

def test_example_unit2() -> None:
    is_sat: bool = check_sat_file("unit2")
    assert(not is_sat)

def test_example_probe1() -> None:
    is_sat: bool = check_sat_file("probe1")
    assert(is_sat)

def test_example_ite17() -> None:
    is_sat: bool = check_sat_file("ite17")
    assert(is_sat)

def test_example_ite18() -> None:
    is_sat: bool = check_sat_file("ite18")
    assert(is_sat)
