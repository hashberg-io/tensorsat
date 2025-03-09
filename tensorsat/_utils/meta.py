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
from abc import ABCMeta
from collections import deque
from collections.abc import Callable, Iterable
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Never,
    NoReturn,
    Self,
    Type,
    TypeVar,
    cast,
    final,
    get_type_hints,
    overload,
)


def is_special_name(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def is_private_name(name: str) -> bool:
    return name.startswith("__") and not name.endswith("__")


def is_protected_name(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def is_public_name(name: str) -> bool:
    return not name.startswith("_")


def access_modifier(name: str) -> Literal["public", "protected", "private", "special"]:
    if not name.startswith("_"):
        return "public"
    if not name.startswith("__"):
        return "protected"
    return "private" if name.endswith("__") else "private"


def all_ancestor_classes(classes: tuple[type, ...]) -> set[type]:
    """The set of all ancestors of the given sequence of classes (incl. themselves)."""
    ancestors = set(classes)
    q = deque(classes)
    while q:
        t = q.popleft()
        new_bases = tuple(s for s in t.__bases__ if s not in ancestors)
        ancestors.update(new_bases)
        q.extend(new_bases)
    return ancestors


def weakref_slot_present(bases: tuple[type, ...]) -> bool:
    """Whether a class with given bases has ``__weakref__`` in its slots."""
    return any(
        "__weakref__" in getattr(cls, "__slots__", {})
        for cls in all_ancestor_classes(bases)
    )


def namespace_union(classes: Iterable[type]) -> dict[str, Any]:
    """
    Union of namespaces from the given classes, with names from earlier classes in the
    iterable shawoding the same names from later classes (if any).
    """
    classes = reversed(tuple(classes))
    namespace: dict[str, Any] = {}
    for base in classes:
        namespace.update(base.__dict__)
    return namespace


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


InstanceT = TypeVar("InstanceT", default=Any)
"""Type variable for instances for the owner class of a :class:`cached_property`."""


ValueT = TypeVar("ValueT", default=Any)
"""Type variable for instances for return value of a :class:`cached_property`."""


@final
class cached_property(Generic[InstanceT, ValueT]):
    """
    A cached property descriptor for slotted classes.
    Usage example:

    .. code-block:: python

        class MyVec:

            _components: tuple[float, ...]

            @cached_property
            def norm(self) -> float:
                return sum(x**2 for x in self._components)

    """

    __WRAPPER_ASSIGNMENTS: ClassVar[tuple[str, ...]] = (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
    )

    @classmethod
    def attrname(cls, name: str) -> str:
        """
        The backing attribute name for a cached property/method with given name.
        Strips underscores, prepends ``__`` and appends ``_cache``.
        Examples:

        .. code-block::

            foo     -> __foo_cache
            _foo    -> __foo_cache
            __foo   -> __foo_cache
            __foo_  -> __foo_cache
            __foo__ -> __foo_cache

        """
        return name.strip("_") + "_cache"

    @staticmethod
    def __validate(owner: Type[Any], name: str) -> None:
        slots = class_slots(owner)
        attrname = cached_property.attrname(name)
        if slots is not None and attrname not in slots:
            raise TypeError(
                f"Backing attribute name {attrname!r} must appear in class slots."
            )

    __name: str
    __owner: Type[InstanceT]
    __mangled_attrname: str
    __wrapped__: Callable[[InstanceT], ValueT]

    def __new__(cls, func: Callable[[InstanceT], ValueT]) -> Self:
        """Public constructor."""
        self = super().__new__(cls)
        # Adapted from functools.update_wrapper:
        for attr in self.__WRAPPER_ASSIGNMENTS:
            try:
                value = getattr(func, attr)
                setattr(self, attr, value)
            except AttributeError:
                pass
        self.__dict__.update(getattr(func, "__dict__", {}))
        self.__wrapped__ = func
        return self

    @property
    def name(self) -> str:
        """The name of the cached property/method."""
        return self.__name

    @property
    def owner(self) -> Type[Any]:
        """The class that owns the cached property/method."""
        return self.__owner

    @property
    def is_assigned(self) -> bool:
        """Whether the descriptor has been assigned to its owner class."""
        try:
            self.__owner
            return True
        except AttributeError:
            return False

    @property
    def mangled_attrname(self) -> str:
        """The mangled attribute name for the cached property/method."""
        return self.__mangled_attrname

    @property
    def fget(self) -> Callable[[InstanceT], ValueT]:
        """The function implementing this cached property."""
        return self.__wrapped__

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """
        Sets the owner and name for the cached property/method.

        :meta public:
        """
        if self.is_assigned:
            raise TypeError("Cannot assign the same descriptor to multiple classes.")
        name = name_unmangle(owner, name)
        cached_property.__validate(owner, name)
        self.__owner = owner
        self.__name = name
        self.__mangled_attrname = name_mangle(
            owner, cached_property.attrname(self.__name)
        )

    @final
    def __set__(self, instance: Any, value: Never) -> NoReturn:
        """
        Cached property/method value cannot be explicitly set.

        :raises AttributeError: unconditionally.

        :meta public:
        """
        raise AttributeError(
            f"cached property {self.__name!r} of"
            f" {self.__owner.__name__!r} object has no setter."
        )

    def __str__(self) -> str:
        try:
            return f"{self.__owner.__qualname__}.{self.__name}"
        except AttributeError:
            return super().__repr__()

    @overload
    def __get__(self, instance: None, _: Type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: InstanceT, _: Type[Any] | None = None) -> ValueT: ...

    def __get__(
        self, instance: InstanceT | None, _: Type[Any] | None = None
    ) -> ValueT | Self:
        """
        Gets the value of the cached_property on the given instance.
        If no instance is passed, returns the descriptor itsef.

        If the value is not cached, it is computed, cached, and then returned.

        :meta public:
        """
        if instance is None:
            return self
        try:
            return cast(
                ValueT, getattr(instance, mangled_attrname := self.mangled_attrname)
            )
        except AttributeError:
            value = self.fget(instance)
            setattr(instance, mangled_attrname, value)
            return value

    def __delete__(self, instance: Any) -> None:
        """
        Deletes the cached value for the property.

        :meta public:
        """
        if hasattr(instance, mangled_attrname := self.mangled_attrname):
            delattr(instance, mangled_attrname)


class TensorSatMeta(ABCMeta):
    """
    Common metaclass for TensorSat classes, implementing the following features:

    - Bans public annotations.
    - Automatically introduces ``__slots__`` entries for all annotated attributes.
    - Automatically introduces ``__slots__`` entries for the backing private attributes
      of properties defined by the :class:`cached_property` decorator.
    - Ensures that only classes marked as :func:`~typing.final` can be instantiated.

    The ban on public annotations forbids the definition of public attributes.
    In the future, we will make a choice to allow them via either one of the following:

    - Annotated as ``ReadOnly``, cf. [PEP 767](https://peps.python.org/pep-0767/)
    - Implemented by a decorator which enforces the readonly condition at runtime,
      once ``TypeForm`` from [PEP 747](https://peps.python.org/pep-0747/) is available.

    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        # Checks that annotations are not public:
        annotations: dict[str, Any] = namespace.setdefault("__annotations__", {})
        for attrname in annotations:
            if is_public_name(attrname):
                raise TypeError(
                    f"Public annotations are not allowed: {name}.{attrname}"
                )
        # Define backing attributes for cached properties:
        cached_properties = {
            member_name: member
            for member_name, member in namespace.items()
            if isinstance(member, cached_property)
        }
        for propname, prop in cached_properties.items():
            slotname = cached_property.attrname(propname)
            if slotname in annotations:
                raise AttributeError(
                    f"Backing attribute {slotname} for cached property"
                    f"{propname} should not be explicitly annotated."
                )
            prop_func_annotations = get_type_hints(prop.__wrapped__)
            if "return" not in prop_func_annotations:
                raise AttributeError(
                    "Return type for cached property must be explicitly annotated."
                )
            return_annotation = prop_func_annotations["return"]
            annotations[slotname] = return_annotation
        # Define slots:
        if "__slots__" in namespace:
            raise TypeError("Class __slots__ should not be explicitly declared.")
        slots: list[str] = []
        if not weakref_slot_present(bases):
            slots.append("__weakref__")
        ext_namespace = namespace_union(bases) | namespace
        slots.extend(
            attrname
            for attrname in annotations
            if attrname not in ext_namespace
            # and not is_special_name(attrname)
        )
        namespace["__slots__"] = tuple(slots)
        # Instantiate and return class:
        cls = super().__new__(mcs, name, bases, namespace)
        cls.__final__ = False
        return cls

    __final__: bool
    """Whether the class is marked final."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if not cls.__final__:
            raise TypeError(
                f"Class {cls.__name__} is not final, so it cannot be instantiated."
            )
        return cls.__new__(cls, *args, **kwargs)  # type: ignore[arg-type]


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
