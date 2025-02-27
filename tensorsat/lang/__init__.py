"""
Diagrammatic languages for the compact-closed category of sets and relations.
Every language corresponds to a choice of parametrisation for a class of relations:
specified by defining types and/or boxes:

- Every (concrete) subclass of :class:`~tensorsat.diagrams.Type` parametrises a
  class of sets (via its constructor arguments).
- Every (concrete) subclass of :class:`~tensorsat.diagrams.Box` parametrises a
  class of relations (via its constructor arguments).

The following languages are currently implemented:

- The language :mod:`~tensorsat.lang.fin_rel` of finite, explicitly enumerated sets
  and relations between them, densely represented by Boolean tensors.

"""

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

from __future__ import annotations
