"""Utility descriptors, for internal use."""

from __future__ import annotations
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Never,
    NoReturn,
    Self,
    Type,
    TypeAlias,
    TypeVar,
    cast,
    final,
    overload,
)


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


InstanceT = TypeVar("InstanceT", default=Any)
"""Type variable for instances for the owner class of a :class:`cached_property`."""

ValueT = TypeVar("ValueT", default=Any)
"""Type variable for instances for return value of a :class:`cached_property`."""


@final
class cached_property(Generic[InstanceT, ValueT]):
    """
    A cached property descriptor for slotted classes.

    The cached property value is stored in a slot named by prepending `__` to the
    cached property name and appending `_cache`.
    For example, if the cached property is named `prop`, the slot used to store the
    cached value is `__prop_cache`.
    """

    @staticmethod
    def __validate(owner: Type[Any], name: str) -> None:
        attrname = "__" + name + "_cache"
        slots = class_slots(owner)
        if slots is not None and attrname not in slots:
            raise TypeError(
                f"Backing attribute name {attrname!r} must appear in class slots."
            )

    __func: Callable[[InstanceT], ValueT]
    __name: str
    __owner: Type[InstanceT]
    __mangled_attrname: str

    __slots__ = ("__name", "__owner", "__func", "__mangled_attrname")

    def __new__(cls, func: Callable[[InstanceT], ValueT]) -> Self:
        """
        Public constructor, can be used as a decorator to turn a method into a cached
        property, as in the following example:

        .. code-block:: python

            class MyClass:

                @cached_property
                def prop(self) -> int:
                    return 10

        :meta public:
        """
        self = super().__new__(cls)
        self.__func = func
        return self

    @property
    def name(self) -> str:
        """The name of the cached property."""
        return self.__name

    @property
    def owner(self) -> Type[Any]:
        """The class that owns the cached property."""
        return self.__owner

    @property
    def func(self) -> Callable[[InstanceT], ValueT]:
        """The function implementing this cached property."""
        return self.__func

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """
        Sets the owner and name for the cached property.

        :meta public:
        """
        if hasattr(self, "_cached_property__owner"):
            raise TypeError("Cannot set owner/name for the same descriptor twice.")
        name = name_unmangle(owner, name)
        cached_property.__validate(owner, name)
        self.__owner = owner
        self.__name = name
        self.__mangled_attrname = name_mangle(owner, "__" + name + "_cache")

    @overload
    def __get__(self, instance: None, _: Type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: InstanceT, _: Type[Any]) -> ValueT: ...

    def __get__(self, instance: InstanceT | None, _: Type[Any]) -> ValueT | Self:
        """
        Gets the value of the cached_property on the given instance.
        If no instance is passed, returns the descriptor itsef.

        If the value is not cached, it is computed, cached, and then returned.

        :meta public:
        """
        if instance is None:
            return self
        try:
            return cast(ValueT, getattr(instance, self.__mangled_attrname))
        except AttributeError:
            value = self.__func(instance)
            setattr(instance, self.__mangled_attrname, value)
            return value

    def __set__(self, instance: Any, value: Never) -> NoReturn:
        """
        Cached property value cannot be explicitly set.

        :raises AttributeError: unconditionally.

        :meta public:
        """
        raise AttributeError(
            f"cached property {self.__name!r} of"
            f" {self.__owner.__name__!r} object has no setter."
        )

    def __delete__(self, instance: Any) -> None:
        """
        Deletes the cached value for the property.

        :meta public:
        """
        delattr(instance, self.__mangled_attrname)

    def __str__(self) -> str:
        try:
            return f"{self.__owner.__qualname__}.{self.__name}"
        except AttributeError:
            return super().__repr__()


AttributeValue: TypeAlias = Any
"""Type alias for attribute values."""


@final
class ReadonlyAttribute:
    """Descriptor for readonly attributes of slotted classes."""

    @staticmethod
    def __validate(owner: Type[Any], name: str) -> None:
        attrname = "__" + name
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
        self.__mangled_attrname = name_mangle(owner, "__" + name)

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
            raise AttributeError(f"Readonly attribute {self} is not set.") from None

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
