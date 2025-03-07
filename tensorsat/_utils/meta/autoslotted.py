"""Metaclass to automatically declare slots for annotated instance attributes."""

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
from collections import deque
from typing import Any

def __all_ancestors(classes: tuple[type, ...]) -> set[type]:
    """The set of all ancestors of the given sequence of classes (incl. themselves)."""
    ancestors = set(classes)
    q = deque(classes)
    while q:
        t = q.popleft()
        new_bases = (s for s in t.__bases__ if s not in ancestors)
        ancestors.update(new_bases)
        q.extend(new_bases)
    return ancestors

def _weakref_slot_present(bases: tuple[type, ...]) -> bool:
    """Whether a class with given bases has ``__weakref__`` in its slots."""
    return any(
        "__weakref__" in getattr(cls, "__slots__", {})
        for cls in __all_ancestors(bases)
    )

def _dict_slot_present(bases: tuple[type, ...]) -> bool:
    """Whether a class with given bases has ``__dict__`` in its slots."""
    return any(
        not hasattr(cls, "__slots__") or "__dict__" in cls.__slots__
        for cls in __all_ancestors(bases)
    )

class AutoSlottedMeta(ABCMeta):
    """Metaclass to automatically declare slots for annotated instance attributes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        use_weakref: bool = True,
        use_dict: bool = False
    ) -> Any:
        if "__slots__" not in namespace:
            slots: list[str] = []
            if use_weakref and not _weakref_slot_present(bases):
                slots.append("__weakref__")
            if use_dict and not _dict_slot_present(bases):
                slots.append("__dict__")
            annotations: dict[str, Any] = namespace.get("__annotations__", {})
            slots.extend(
                attr_name for attr_name in annotations
                if attr_name not in namespace
                and all(attr_name not in base.__dict__ for base in bases)
            )
            namespace["__slots__"] = tuple(slots)
        return super().__new__(mcs, name, bases, namespace)
