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
from typing import Generic, Self, Type as SubclassOf, final
from ..diagrams import Box, BoxT_inv, Diagram

if __debug__:
    from typing_validation import validate

class Contraction(Generic[BoxT_inv], metaclass=ABCMeta):
    """Abstract base class for contractions."""

    __box_class: SubclassOf[BoxT_inv]

    __slots__ = ("__weakref__", "__box_class")

    def __new__(cls, box_class: SubclassOf[BoxT_inv]) -> Self:
        assert validate(box_class, SubclassOf[Box])
        if not box_class.can_be_contracted():
            raise ValueError("Given box class cannot be contracted.")
        self = super().__new__(cls)
        self.__box_class = box_class
        return self

    @property
    def box_class(self) -> SubclassOf[BoxT_inv]:
        """Box class associated with this contraction."""
        return self.__box_class

    @final
    def can_contract(self, diagram: Diagram) -> bool:
        """Whether the diagram can be contracted using this contraction."""
        try:
            self.validate(diagram)
            return True
        except ValueError:
            return False

    @final
    def validate(self, diagram: Diagram) -> None:
        """Raises :class:`ValueError` if the diagram cannot be contracted."""
        assert validate(diagram, Diagram)
        if not issubclass(diagram.box_class, self.box_class):
            raise ValueError(
                f"Cannot contract diagram: diagram box class {diagram.box_class} is"
                f" not a subclass of contraction box class {self.box_class}"
            )
        self._validate(diagram)

    @final
    def contract(self, diagram: Diagram) -> Box:
        """
        Contracts the diagram using this contraction.

        :raises ValueError: if the diagram cannot be contracted.
        """
        self._validate(diagram)
        box_class = diagram.box_class
        if not box_class.can_be_contracted():
            raise ValueError(
                f"Diagram's join box class {box_class.__name__} cannot be contracted."
            )
        return self._contract(diagram)

    @abstractmethod
    def _contract(self, diagram: Diagram) -> Box:
        """Diagram contraction logic, to be implemented by subclasses."""

    @abstractmethod
    def _validate(self, diagram: Diagram) -> None:
        """
        Diagram validation logic, to be implemented by subclasses.

        :raises ValueError: if the diagram cannot be contracted.
        """
