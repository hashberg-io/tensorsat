"""Metaclass to automatically set define readonly descriptors for public attributes."""

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
from typing import Any

from ..descriptors import ReadonlyAttribute


class AutoReadonlyMeta(ABCMeta):
    """
    Metaclass to automatically set define readonly descriptors for public attributes.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        annotations: dict[str, Any] = namespace.get("__annotations__", {})
        public_instance_attrs: dict[str, Any] = {
            attr_name: annotation
            for attr_name, annotation in annotations.items()
            if attr_name not in namespace
            and all(attr_name not in base.__dict__ for base in bases)
            and not attr_name.startswith("_")
        }
        for attr_name, annotation in public_instance_attrs.items():
            __attr_name = "__" + attr_name
            if __attr_name in annotations:
                raise TypeError(
                    f"Cannot define public instance attribute {name}.{attr_name} when"
                    f" private instance attribute {name}.{__attr_name} is also defined."
                )
            if attr_name in namespace:
                raise TypeError(
                    f"Cannot define public instance attribute {name}.{attr_name} when"
                    f" class attribute by the same name is already defined."
                )
            del annotations[attr_name]
            annotations[__attr_name] = annotation
            namespace[attr_name] = ReadonlyAttribute()
        return super().__new__(mcs, name, bases, namespace)
