"""Metaclass enforcing an inheritance forest for its own instance classes."""

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
from abc import ABCMeta
from collections.abc import Iterable
from typing import Any, cast


class InheritanceForestMeta(ABCMeta):
    """Metaclass enforcing an inheritance forest for its own instance classes."""

    __parent_class: InheritanceForestMeta | None
    __root_class: InheritanceForestMeta
    __class_tree_depth: int

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        parent_class: InheritanceForestMeta | None = None
        for base in bases:
            if isinstance(base, InheritanceForestMeta):
                if parent_class is not None:
                    raise TypeError(
                        "Class of metaclass InheritanceForestMeta can have at most"
                        " one base class of metaclass InheritanceForestMeta."
                    )
                parent_class = base
        cls = super().__new__(mcs, name, bases, namespace)
        cls.__parent_class = parent_class
        cls.__root_class = cls if parent_class is None else parent_class.__root_class
        cls.__class_tree_depth = (
            0 if parent_class is None else parent_class.__class_tree_depth + 1
        )
        return cls

    @property
    def _parent_class(self) -> InheritanceForestMeta | None:
        """
        The parent class in this class's inheritance tree,
        or :obj:`None` if this class is the root of an inheritance tree.
        """
        return self.__parent_class

    @property
    def _root_class(self) -> InheritanceForestMeta:
        """The root class in this class's inheritance tree."""
        return self.__root_class

    @classmethod
    def _join(
        mcs, classes: Iterable[InheritanceForestMeta]
    ) -> InheritanceForestMeta | None:
        """
        Computes the join of the given instance classes.
        Returns :obj:`None` if the join doesn't exist.
        """
        res: InheritanceForestMeta | None = None
        for cls in classes:
            if res is None:
                res = cls
            elif issubclass(res, cls):
                res = cls
            elif issubclass(cls, res):
                pass
            else:
                if res.__class_tree_depth < cls.__class_tree_depth:
                    res, cls = cls, res
                while not issubclass(res, cls):
                    if (subcls_parent := cls._parent_class) is None:
                        return None
                    if not issubclass(subcls_parent, cls):
                        return None
                    cls = subcls_parent
                res = cls
        return res

    def _subclass_join[_T: type](cls: _T, subclasses: Iterable[_T]) -> _T | None:
        """
        Computes the join of the given subclasses, under the constraint that it be
        a subclass of this class.
        Returns :obj:`None` if the join either doesn't exist or it isn't a subclass
        of this class.
        """
        join = type(cast(InheritanceForestMeta, cls))._join(
            cast(Iterable[InheritanceForestMeta], subclasses)
        )
        if join is None:
            return None
        if not issubclass(join, cls):
            return None
        return cast(_T, join)
