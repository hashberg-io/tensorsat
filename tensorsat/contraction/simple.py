"""Simple diagram contractions, based on explicit contraction paths."""

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
from collections.abc import Iterable
from typing import Self, final

from .abc import Contraction
from ..diagrams import Box, Diagram, Wiring, Type

if __debug__:
    from typing_validation import validate


@final
class SimpleContraction(Contraction):
    """A simple contraction based on an explicit contraction path."""

    __wiring: Wiring
    __path: tuple[tuple[int, int], ...]

    __slots__ = ("__wiring", "__path")

    def __new__(cls, wiring: Wiring, path: Iterable[tuple[int, int]]) -> Self:
        path = tuple(path)
        assert validate(wiring, Wiring)
        assert validate(path, tuple[tuple[int, int], ...])
        # TODO: validate path against wiring and pre-compute indices for Box.contract2
        raise NotImplementedError()

    def _contract[_T: Type](self, diagram: Diagram[_T]) -> Box[_T]:
        """Diagram contraction logic, to be implemented by subclasses."""
        # TODO: apply Box.contract2 in order of contraction
        # TODO: handle special cases with no boxes (spiders) or single box (transpose)
        raise NotImplementedError()

    def _validate(self, diagram: Diagram) -> None:
        """Diagram validation logic, to be implemented by subclasses."""
        if diagram.wiring != self.__wiring:
            raise ValueError("Diagram's wiring must match contraction wiring.")
        if not diagram.is_flat:
            raise ValueError("Diagram must be flat.")
