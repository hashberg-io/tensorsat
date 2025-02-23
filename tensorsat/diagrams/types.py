"""
Implementation of types and shapes for the :mod:`tensorsat.diagrams` module.
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
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    TypeVar,
    cast,
    final,
    overload,
)
from hashcons import InstanceStore

if __debug__:
    from typing_validation import validate

if TYPE_CHECKING:
    from .boxes import Box


class Type(ABC):
    """
    Abstract base class for types in diagrams.

    Types are used to signal compatibility between boxes, by requiring that ports wired
    together in a diagram all have the same type.
    By sharing common types, boxes from multiple languages can be wired together in the
    same diagram.
    """

    __final__: ClassVar[bool] = False

    __slots__ = ("__weakref__",)

    def __new__(cls) -> Self:
        """Constructs a new type."""
        if not cls.__final__:
            raise TypeError("Only final subclasses of Type can be instantiated.")
        return super().__new__(cls)

    @final
    def spider(self, num_ports: int) -> Box[Self]:
        """
        The box corresponding to a single wire connected to the given number of ports,
        all ports being of this type.
        """
        validate(num_ports, int)
        if num_ports <= 0:
            raise ValueError("Number of ports must be strictly positive.")
        return self._spider(num_ports)

    @abstractmethod
    def _spider(self, num_ports: int) -> Box[Self]:
        """
        Protected version of :meth:`Type.spider`, to be implemented by subclasses.
        It is guaranteed that ``num_ports`` is strictly positive.
        """

    @final
    def __mul__[T: Self, _T: Type](self: T, other: _T | Shape[_T]) -> Shape[T | _T]:
        """Takes the product of this type with another type or shape."""
        if isinstance(other, Shape):
            return Shape([self, *other])
        return Shape([self, other])

    @final
    def __pow__[T: Self](self: T, rhs: int, /) -> Shape[T]:
        """Repeats a type a given number of times."""
        assert validate(rhs, int)
        return Shape([self] * rhs)


TypeT_co = TypeVar("TypeT_co", bound=Type, covariant=True, default=Type)
"""Covariant type variable for a type."""

TypeT_inv = TypeVar("TypeT_inv", bound=Type, default=Type)
"""Invariant type variable for a type."""


class Shape(Sequence[TypeT_co]):
    """A Shape, as a finite tuple of types."""

    _store: ClassVar[InstanceStore] = InstanceStore()

    @classmethod
    def prod(cls, shapes: Iterable[Shape[TypeT_co]], /) -> Shape[TypeT_co]:
        """Takes the product of multiple shapes, i.e. concatenates their types."""
        shapes = tuple(shapes)
        assert validate(shapes, tuple[Shape[Type], ...])
        return Shape._prod(shapes)

    @classmethod
    def _prod(cls, shapes: tuple[Shape[TypeT_co], ...], /) -> Shape[TypeT_co]:
        return cls._new(sum((shape.__components for shape in shapes), ()))

    __components: tuple[TypeT_co, ...]

    __slots__ = ("__weakref__", "__components")

    @classmethod
    def _new(cls, components: tuple[TypeT_co, ...]) -> Self:
        """Protected constructor."""
        with Shape._store.instance(cls, components) as self:
            if self is None:
                self = super().__new__(cls)
                self.__components = components
                Shape._store.register(self)
        return self

    def __new__(cls, components: Iterable[TypeT_co]) -> Self:
        """
        Constructs a new shape with given component types.
        If iterables of types are passed, their types are extracted and inserted
        into the shape at the selected point.
        """
        components = tuple(components)
        assert validate(components, tuple[Type, ...])
        return cls._new(components)

    def __mul__[T: Type](self, rhs: T | Shape[T], /) -> Shape[TypeT_co | T]:
        """Takes the product of two shapes (i.e. concatenates their types)."""
        if isinstance(rhs, Type):
            return Shape([*self, cast(T, rhs)])
        assert validate(rhs, Shape)
        return Shape([*self, *rhs])

    def __pow__(self, rhs: int, /) -> Shape[TypeT_co]:
        """Repeats a shape a given number of times."""
        assert validate(rhs, int)
        return Shape(tuple(self) * rhs)

    def __iter__(self) -> Iterator[TypeT_co]:
        """Iterates over the components of the shape."""
        return iter(self.__components)

    def __len__(self) -> int:
        """Returns the number of components in the shape."""
        return len(self.__components)

    @overload
    def __getitem__(self, index: int, /) -> TypeT_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Shape[TypeT_co]: ...
    @overload
    def __getitem__(self, index: Iterable[int], /) -> Shape[TypeT_co]: ...
    def __getitem__(
        self, index: int | slice | Iterable[int], /
    ) -> TypeT_co | Shape[TypeT_co]:
        """Returns the component(s) at the given index(es)."""
        if isinstance(index, slice):
            return Shape._new(self.__components[index])
        if isinstance(index, int):
            return self.__components[index]
        index = list(index)
        assert validate(index, list[int])
        components = self.__components
        return Shape._new(tuple(components[i] for i in index))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.__components == other.__components

    def __hash__(self) -> int:
        return hash((Shape, self.__components))

    def __repr__(self) -> str:
        return f"<Shape {id(self):#x}: {len(self)} components>"
