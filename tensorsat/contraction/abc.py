"""Abstract base classes for diagrammatic contraction."""

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
from abc import ABCMeta, abstractmethod
from typing import final
from ..diagrams import Box, Diagram, Type


class Contraction(metaclass=ABCMeta):
    """Abstract base class for contractions."""

    __slots__ = ("__weakref__",)

    @final
    def can_contract(self, diagram: Diagram) -> bool:
        """Whether the diagram can be contracted using this contraction."""
        try:
            self._validate(diagram)
            return True
        except ValueError:
            return False

    @final
    def contract[_T: Type](self, diagram: Diagram[_T]) -> Box[_T]:
        """
        Contracts the diagram using this contraction.

        :raises ValueError: if the diagram cannot be contracted.
        """
        self._validate(diagram)
        return self._contract(diagram)

    @abstractmethod
    def _contract[_T: Type](self, diagram: Diagram[_T]) -> Box[_T]:
        """Diagram contraction logic, to be implemented by subclasses."""

    @abstractmethod
    def _validate(self, diagram: Diagram) -> None:
        """Diagram validation logic, to be implemented by subclasses."""
