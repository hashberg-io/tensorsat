"""Utilities for checking the satisfiability of CNF instances."""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from tensorsat.lib.bincirc import and_
from tensorsat.lib.sat import *
from tensorsat.lang.fin_rel import FinRel
from tensorsat.contractions.cotengra import CotengraContraction
import numpy as np
import sys

def is_satisfiable(cnf: CNFInstance) -> bool:
    cnf_diagram = cnf.diagram()
    cnf_sat_diagram = (cnf.inputs(None)>>cnf_diagram).flatten()
    cnf_sat_contraction = CotengraContraction(FinRel, cnf_sat_diagram.wiring)

    return bool(cnf_sat_contraction.contract(cnf_sat_diagram, progbar=False))
