"""
Implementation of boxes and box recipes for the :mod:`tensorsat.diagrams` module.
"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
    final,
)

if __debug__:
    from typing_validation import validate

from .types import Shape, TypeT_co, TypeT_inv
from .wirings import Port, Shaped, Wire

if TYPE_CHECKING:
    from .diagrams import SelectedInputWires


class BoxMeta(ABCMeta):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> BoxMeta:
        cls = super().__new__(mcs, name, bases, namespace)
        if not cls.__abstractmethods__:
            try:
                import autoray  # type: ignore
                autoray.register_backend(cls, "tensorsat._autoray")
            except ModuleNotFoundError:
                pass
        return cls


class Box(Shaped[TypeT_co], metaclass=BoxMeta):
    """
    Abstract base class for boxes in diagrams.
    """

    __final__: ClassVar[bool] = False

    @final
    @classmethod
    def contract2(
        cls,
        lhs: Self,
        lhs_wires: Sequence[Wire],
        rhs: Self,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire] | None = None,
    ) -> Self:
        assert validate(lhs, cls)
        assert validate(lhs_wires, Sequence[Wire])
        assert validate(rhs, cls)
        assert validate(rhs_wires, Sequence[Wire])
        assert validate(out_wires, Sequence[Wire] | None)
        if len(lhs_wires) != len(lhs.shape):
            raise ValueError(
                f"Number of wires in lhs ({len(lhs_wires)}) does not match"
                f" the number of ports in lhs shape ({len(lhs.shape)})."
            )
        if len(rhs_wires) != len(rhs.shape):
            raise ValueError(
                f"Number of wires in rhs ({len(rhs_wires)}) does not match"
                f" the number of ports in rhs shape ({len(rhs.shape)})."
            )
        if out_wires is None:
            out_wires = sorted(set(lhs_wires).symmetric_difference(rhs_wires))
        else:
            out_wires_set = set(out_wires)
            if len(out_wires) != len(out_wires_set):
                raise NotImplementedError("Output wires cannot be repeated.")
                # TODO: This is not ordinarily handled by einsum,
                #       but it is pretty natural in the context of the wirings we use,
                #       so we might wish to add support for it in the future.
            out_wires_set.difference_update(lhs_wires)
            out_wires_set.difference_update(rhs_wires)
            if out_wires_set:
                raise ValueError("Every output wire must appear in LHR or RHS.")
        return cls._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)

    @classmethod
    @abstractmethod
    def _contract2(
        cls,
        lhs: Self,
        lhs_wires: Sequence[Wire],
        rhs: Self,
        rhs_wires: Sequence[Wire],
        out_wires: Sequence[Wire],
    ) -> Self:
        """
        Protected version of :meth:`Box.contract2`, to be implemented by subclasses.
        It is guaranteed that:
        - The length of ``lhs_wires`` matches the length of ``lhs.shape``
        - The length of ``rhs_wires`` matches the length of ``rhs.shape``
        - Indices in ``out_wires`` are not repeated
        - Every index in ``out_wires`` appears in ``lhs_wires`` or ``rhs_wires``
        """

    @final
    @staticmethod
    def recipe(
        recipe: Callable[[Shape[TypeT_inv]], BoxT_inv],
    ) -> BoxRecipe[TypeT_inv, BoxT_inv]:
        return BoxRecipe(recipe)

    __recipe_used: BoxRecipe[TypeT_co, Self] | None

    __slots__ = ("__weakref__", "__recipe_used")

    def __new__(cls) -> Self:
        """Constructs a new box."""
        if not cls.__final__:
            raise TypeError("Only final subclasses of Box can be instantiated.")
        self = super().__new__(cls)
        self.__recipe_used = None
        return self

    @final
    @property
    def recipe_used(self) -> BoxRecipe[TypeT_co, Self] | None:
        """The recipe used to create the box, if any."""
        return self.__recipe_used

    @final
    def transpose(self, perm: Sequence[Port]) -> Self:
        """Transposes the output ports into the given order."""
        assert validate(perm, Sequence[Port])
        if len(perm) != self.num_ports or set(perm) != set(self.ports):
            raise ValueError(
                "Input to transpose method must be a permutation of the box ports."
            )
        return self._transpose(perm)

    @abstractmethod
    def _transpose(self, perm: Sequence[Port]) -> Self:
        """
        Protected version of :meth:`Box.transpose`, to be implemented by subclasses.
        It is guaranteed that ``perm`` is a permutation of ``range(self.ports)``.
        """

    def __mul__(self, other: Self) -> Self:
        """
        Takes the product of this relation with another relation of the same class.
        The resulting relation has as its ports the ports of this relation followed
        by the ports of the other relation, and is of the same class of both.
        """
        lhs, rhs = self, other
        lhs_len, rhs_len = len(lhs.shape), len(rhs.shape)
        lhs_wires, rhs_wires = range(lhs_len), range(lhs_len, lhs_len + rhs_len)
        out_wires = range(lhs_len + rhs_len)
        return type(self)._contract2(lhs, lhs_wires, rhs, rhs_wires, out_wires)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        num_ports = len(self.shape)
        return f"<{cls_name} {id(self):#x}: {num_ports} ports>"


BoxT_inv = TypeVar("BoxT_inv", bound=Box, default=Box)
"""
Invariant type variables for box classes.

(A generic type variable would be perfect here, but Python doesn't have those yet...)
"""


@final
class BoxRecipe(Generic[TypeT_inv, BoxT_inv]):
    """
    Utility class wrapping box building logic, which can be executed on
    demand for given input types.

    Supports usage of the ``@`` operator with selected input wires on the rhs,
    analogously to the special block addition syntax for diagram builders.

    See the :func:`Box.recipe` decorators for an example of how this works.
    """

    __recipe: Callable[[Shape[TypeT_inv]], BoxT_inv]
    __name: str | None

    __slots__ = ("__weakref__", "__recipe", "__name")

    def __new__(
        cls,
        recipe: Callable[[Shape[TypeT_inv]], BoxT_inv],
    ) -> Self:
        """Wraps the given box building logic."""
        self = super().__new__(cls)
        self.__recipe = recipe
        self.__name = getattr(recipe, "__name__", None)
        return self

    def __call__(self, shape: Iterable[TypeT_inv]) -> BoxT_inv:
        """
        Creates a box starting from a given shape, which we can think of as the
        shape of its "input" ports.
        The shape of the box returned is guaranteed to start with the given types,
        but may contain further types, corresponding to the "output" ports of the box.
        """
        box = self.__recipe(Shape(shape))
        if not hasattr(box, "_Box__recipe_used"):
            box._Box__recipe_used = self  # type: ignore[attr-defined]
        assert box._Box__recipe_used is self, (  # type: ignore[attr-defined]
            "Box recipe set incorrectly."
        )
        return box

    def __matmul__(self, selected: SelectedInputWires[TypeT_inv]) -> tuple[Wire, ...]:
        """
        Executes the recipe for the input types specified by the selected input wires,
        then adds the diagram resulting from the recipe as a block in the the broader
        diagram being built, connected to the given input wires, and returns the
        resulting output wires.
        """
        selected_wires = selected.wires
        num_ports = len(selected_wires)
        if set(selected_wires) != set(range(num_ports)):
            raise ValueError("Selected ports must form a contiguous zero-based range.")
        wire_types = selected.builder.wiring.wire_types
        input_types = tuple(
            wire_types[selected_wires[port]] for port in range(num_ports)
        )
        box = cast(Box[TypeT_inv], self(input_types))
        return box @ selected

    def __repr__(self) -> str:
        name = self.__name
        if name is None:
            return f"<BoxRecipe {id(self):#x}>"
        return f"<BoxRecipe {id(self):#x} {name!r}>"
