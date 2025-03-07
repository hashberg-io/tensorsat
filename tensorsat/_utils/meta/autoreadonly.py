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
from typing import Any, Self, Type, TypeAlias, final, overload

def name_mangle(owner: type, attr_name: str) -> str:
    """
    If the given attribute name is private and not dunder,
    return its name-mangled version for the given owner class.
    """
    if not attr_name.startswith("__"):
        return attr_name
    if attr_name.endswith("__"):
        return attr_name
    return f"_{owner.__name__}{attr_name}"

def name_unmangle(owner: type, attr_name: str) -> str:
    """
    If the given attribute name is name-mangled for the given owner class,
    removes the name-mangling prefix.
    """
    name_mangling_prefix = f"_{owner.__name__}"
    if attr_name.startswith(name_mangling_prefix + "__"):
        return attr_name[len(name_mangling_prefix) :]
    return attr_name

def class_slots(cls: type) -> tuple[str, ...] | None:
    """
    Returns a tuple consisting of all slots for the given class and all
    non-private slots for all classes in its MRO.
    Returns :obj:`None` if slots are not defined for the class.
    """
    if not hasattr(cls, "__slots__"):
        return None
    slots: list[str] = list(cls.__slots__)
    for cls in cls.__mro__[1:-1]:
        for slot in getattr(cls, "__slots__", ()):
            assert isinstance(slot, str)
            if slot.startswith("__") and not slot.endswith("__"):
                continue
            slots.append(slot)
    return tuple(slots)

def class_slotset(*classes: type) -> frozenset[str]:
    """
    Returns the set of slots defined by the classes.
    If a class does not define slots, ``__dict__`` is added to the slotset.
    """
    slotset: set[str] = set()
    for cls in classes:
        slots = class_slots(cls)
        if slots is None:
            slotset.add("__dict__")
        else:
            slotset.update(slots)
    return frozenset(slotset)


AttributeValue: TypeAlias = Any
"""Type alias for attribute values."""

@final
class ReadonlyAttribute:
    """Descriptor for readonly attributes of slotted classes."""

    @staticmethod
    def __validate(owner: Type[Any], name: str) -> None:
        attrname = "__"+name
        slots = class_slots(owner)
        if slots is not None and attrname not in slots:
            raise TypeError(
                f"Backing attribute name {attrname!r} must appear in class slots."
            )

    __name: str
    __owner: Type[Any]
    __mangled_attrname: str

    __slots__ = ("__name", "__owner", "__mangled_attrname")

    @property
    def name(self) -> str:
        """The name of the attribute."""
        return self.__name

    @property
    def owner(self) -> Type[Any]:
        """The class that owns the attribute."""
        return self.__owner

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """
        Sets the owner and backing attribute name for the descriptor.

        :meta public:
        """
        if hasattr(self, "_ReadonlyAttribute__owner"):
            raise TypeError("Cannot set owner/name for the same descriptor twice.")
        name = name_unmangle(owner, name)
        ReadonlyAttribute.__validate(owner, name)
        self.__owner = owner
        self.__name = name
        self.__mangled_attrname = name_mangle(owner, "__"+name)

    @overload
    def __get__(self, instance: None, _: Type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: Any, _: Type[Any]) -> AttributeValue: ...

    def __get__(self, instance: Any, _: Type[Any]) -> Self | AttributeValue:
        """
        Gets the value of the attribute on the given instance.
        If no instance is passed, returns the attribute descriptor itsef.

        :meta public:
        """
        if instance is None:
            return self
        try:
            return getattr(instance, self.__mangled_attrname)
        except AttributeError:
            raise AttributeError(f"Attribute {self} is not set.") from None

    def __set__(self, instance: Any, value: AttributeValue) -> None:
        """
        Sets the value of the attribute on the given instance.
        Can only be called once, after which it raises :class:`AttributeError`.

        :meta public:
        """
        if hasattr(instance, self.__mangled_attrname):
            raise AttributeError(f"Readonly attribute {self} can only be set once.")
        setattr(instance, self.__mangled_attrname, value)

    def __delete__(self, instance: Any) -> None:
        """
        Raises :class:`AttributeError`, because readonly attributes cannot be deleted.

        :meta public:
        """
        raise AttributeError(f"Readonly attribute {self} cannot be deleted.")

    def __str__(self) -> str:
        try:
            return f"{self.__owner.__qualname__}.{self.__name}"
        except AttributeError:
            return super().__repr__()

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
            __attr_name = "__"+attr_name
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
            annotation[__attr_name] = annotation
            namespace[attr_name] = ReadonlyAttribute()
        return super().__new__(mcs, name, bases, namespace)
