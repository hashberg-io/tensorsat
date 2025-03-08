"""Common metaclass for TensorSat classes."""

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
from typing import Any, get_type_hints

from ..descriptors import cached_property
from .autoslotted import AutoSlottedMeta
from .autoreadonly import AutoReadonlyMeta


class TensorSatMeta(AutoReadonlyMeta, AutoSlottedMeta):
    """
    Common metaclass for TensorSat classes, implementing the following features:

    - Automatically introduces ``__slots__`` entries for all annotated attributes.
    - Automatically transforms public attribute annotations into readonly public
      descriptors backed by private attributes (automatically added to ``__slots__``).
    - Automatically introduces ``__slots__`` entries for the backing private attributes
      of properties defined by the :class:`cached_property` decorator.

    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        annotations: dict[str, Any] = namespace.setdefault("__annotations__", {})
        cached_properties: dict[str, cached_property] = {}
        for member_name, member in namespace.items():
            if isinstance(member, cached_property):
                cached_properties[member_name]
        for prop_name, prop in cached_properties.items():
            backing_attr_name = "__" + prop_name + "_cache"
            if backing_attr_name in annotations:
                raise AttributeError(
                    f"Backing attribute {backing_attr_name} for cached property"
                    f"{prop_name} should not be explicitly annotated."
                )
            prop_func_annotations = get_type_hints(prop.func)
            if "return" not in prop_func_annotations:
                raise AttributeError(
                    "Return type for cached property must be explicitly annotated."
                )
            return_annotation = prop_func_annotations["return"]
            annotations[backing_attr_name] = return_annotation
        return super().__new__(mcs, name, bases, namespace)
