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
from abc import abstractmethod
from typing import Generic, Self, Type as SubclassOf, final

from .._utils.meta import TensorSatMeta
from ..diagrams import Box, BoxT_inv, Diagram, Wiring

if __debug__:
    from typing_validation import validate


class Contraction(Generic[BoxT_inv], metaclass=TensorSatMeta):
    """Abstract base class for contractions."""

    __box_class: SubclassOf[BoxT_inv]
    __wiring: Wiring

    def __new__(cls, box_class: SubclassOf[BoxT_inv], wiring: Wiring) -> Self:
        assert validate(box_class, SubclassOf[Box])
        assert validate(wiring, Wiring)
        if not box_class.can_be_contracted():
            raise ValueError("Given box class cannot be contracted.")
        self = super().__new__(cls)
        self.__box_class = box_class
        self.__wiring = wiring
        return self

    @property
    def box_class(self) -> SubclassOf[BoxT_inv]:
        """Box class associated with this contraction."""
        return self.__box_class

    @property
    def wiring(self) -> Wiring:
        """The wiring contracted by this contraction."""
        return self.__wiring

    @final
    def can_contract(self, diagram: Diagram) -> bool:
        """Whether the diagram can be contracted using this contraction."""
        try:
            self.validate(diagram)
            return True
        except ValueError:
            return False

    def validate(self, diagram: Diagram) -> None:
        """
        Raises :class:`ValueError` if the diagram cannot be contracted,
        because it doesn't respect one or more of the following conditions:

        - The diagram's :attr:`~Diagram.box_class` must be the contraction's
          :attr:`Contraction.box_class` or a subclass thereof.
        - The diagram's :attr:`~Diagram.wiring` must be the same as the
          contraction's :attr:`~Contraction.wiring`.
        - The diagram cannot have :attr:`~Diagram.open_slots`.
        - The diagram must be flat (cf. :attr:`~Diagram.is_flat`).

        This method can be overridden by subclasses for additional validation.
        """
        assert validate(diagram, Diagram)
        if not all(issubclass(cls, self.__box_class) for cls in diagram.box_classes):
            raise ValueError(
                f"Cannot contract diagram: diagram box class {diagram.box_class} is"
                f" not a subclass of contraction box class {self.__box_class}"
            )
        if diagram.wiring != self.wiring:
            raise ValueError("Diagram's wiring must match contraction wiring.")
        if diagram.num_open_slots > 0:
            raise ValueError("Diagram cannot have open slots.")
        if not diagram.is_flat:
            raise ValueError("Diagram must be flat.")

    @final
    def contract(self, diagram: Diagram) -> BoxT_inv:
        """
        Contracts the diagram using this contraction.

        :raises ValueError: if the diagram cannot be contracted.
        """
        self.validate(diagram)
        return self._contract(diagram)

    @abstractmethod
    def _contract(self, diagram: Diagram) -> BoxT_inv:
        """
        Diagram contraction logic, to be implemented by subclasses.

        It is guaranteed that the diagram has already been validated.
        """
