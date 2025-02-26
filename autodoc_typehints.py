r"""
    Autodoc extension dealing with local type references and function signatures.
"""

# Some References:
#
# https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html
# https://github.com/sphinx-doc/sphinx/blob/49d1e7142eca6d303f1e929bdbe9d8e4749858b7/sphinx/ext/autodoc/__init__.py#L2737
# https://stackoverflow.com/questions/39565203/how-to-specialize-documenters-in-sphinx-autodoc


from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
import inspect
import re
import traceback
from types import FunctionType, ModuleType
from typing import Any, ForwardRef, Optional
from sphinx.application import Sphinx
from sphinx.util import logging
from typed_descriptors import TypedDescriptor
from typing_validation import inspect_type

logger = logging.getLogger(__name__)

### 1. Parse Type Annotations ###

@dataclass(frozen=True)
class ParsedType:
    r""" Dataclass for a parsed type. """
    name: str
    args: None|str|tuple[ParsedType, ...] = None
    variadic: bool = False

    def __post_init__(self) -> None:
        name, args, variadic = self.name, self.args, self.variadic
        assert isinstance(name, str)
        assert isinstance(variadic, bool)
        if args is not None and not isinstance(args, str):
            assert isinstance(args, tuple)
            assert all(isinstance(arg, ParsedType) for arg in args)
        if variadic:
            assert isinstance(args, tuple)
        if "(" in name or ")" in name:
            raise ValueError(
                "Round brackets in type annotations are only supported for "
                "empty type arguments lists, in the form TypeName[()]."
            )
        if isinstance(args, str) and not args:
            raise ValueError(
                "Literal type must include at least one value."
            )

    def crossref(self, globalns: Optional[Mapping[str, Any]] = None) -> str:
        r""" Generates Sphinx cross-reference link for the given type, using local names. """
        # pylint: disable = eval-used
        if globalns is None:
            globalns = {}
        name, args, variadic = self.name, self.args, self.variadic
        role = "obj"
        if name in globalns:
            obj = globalns[name]
            if isinstance(obj, ModuleType):
                role = "mod"
            elif isinstance(obj, property):
                role = "attr"
            elif isinstance(obj, type):
                if name not in ("Any", "typing.Any"):
                    role = "class"
            elif isinstance(obj, FunctionType):
                role = "func"
        name_crossref = f":{role}:`{name}`"
        if args is None:
            return name_crossref
        if name in ("UnionType", "types.UnionType"):
            assert isinstance(args, tuple)
            return " | ".join((arg.crossref(globalns) for arg in args))
        if isinstance(args, str):
            _args = eval(f"({args}, )")
            arg_crossrefs = ", ".join(f"``{repr(arg)}``" for arg in _args)
        elif not args:
            arg_crossrefs = "()"
        else:
            arg_crossrefs = ", ".join((arg.crossref(globalns) for arg in args))
        if variadic:
            arg_crossrefs += ", ..."
        return fr"{name_crossref}\ [{arg_crossrefs}]"

    def _repr(self, level: int = 0) -> list[str]:
        basic_indent = "  "
        indent = basic_indent * level
        next_indent = basic_indent * (level+1)
        name, args, variadic = self.name, self.args, self.variadic
        if args is None:
            assert not variadic
            return [indent+f"ParsedType({name!r})"]
        lines = [
            indent+"ParsedType(",
            next_indent+f"name = {self.name!r},"
        ]
        if isinstance(args, str):
            assert not variadic
            lines.append(next_indent+f"args = {args!r}")
        else:
            assert isinstance(args, tuple)
            if not args:
                assert not variadic
                lines.append(next_indent+"args = ()")
            else:
                lines.append(next_indent+"args = (")
                for i, arg in enumerate(args):
                    assert isinstance(arg, ParsedType)
                    lines.extend(arg._repr(level+2))
                    sep = "," if i < len(args)-1 or len(args) == 1 else ""
                    lines[-1] = lines[-1]+sep
                lines.append(next_indent+")")
        if variadic:
            lines[-1] = lines[-1]+","
            lines.append(next_indent+"variadic = True")
        lines.append(indent+")")
        return lines

    def __repr__(self) -> str:
        """
        Structured representation of the parsed type.
        """
        return "\n".join(self._repr())

def _outer_bracket_ranges(
    s: str,
    start: int, stop: int
) -> Iterator[range]:
    if stop is None:
        stop = len(s)
    open: int|None = None
    level = 0
    for i in range(start, stop):
        c = s[i]
        if c == "[":
            if open is None:
                assert level == 0
                open = i
            level += 1
        if c == "]":
            if open is None:
                raise ValueError(f"Unbalanced ']' at index {i}.")
            assert level > 0
            level -= 1
            if level == 0:
                yield range(open, i+1)
                open = None
    if open is not None:
        raise ValueError(f"Unbalanced '[' at index {open}.")

def _find_outside_ranges(
    char: str, s: str, ranges: Iterable[range],
    start: int, stop: int
) -> Iterator[int]:
    assert len(char) == 1, f"Expected single char, found {s!r}."
    if stop is None:
        stop = len(s)
    ranges = deque(sorted(ranges, key=lambda r: r.start))
    _start = start
    while (char_idx := s.find(char, _start, stop)) >= 0:
        while ranges and ranges[0].stop <= _start:
            ranges.popleft()
        if not ranges or char_idx not in ranges[0]:
            yield char_idx
            _start = char_idx+1
        else:
            r = ranges.popleft()
            _start = r.stop

def _split_at(
    idxs: Iterable[int],
    start: int, stop: int
) -> Iterable[range]:
    idxs = sorted(idxs)
    assert all(idx in range(start, stop) for idx in idxs)
    _start = start
    for idx in idxs:
        if idx > _start:
            yield range(_start, idx)
        else:
            assert idx == _start
        _start = idx + 1
    if _start < stop:
        yield range(_start, stop)

def _parsed_type(
    annotation: str,
    start: int,
    stop: int,
    name: str, args:
    None|str|tuple[ParsedType, ...] = None,
    variadic: bool = False
) -> ParsedType:
    try:
        t = ParsedType(name, args, variadic)
    except ValueError as e:
        raise ValueError(
            "Error parsing type at \n"
            f"{start = }, {stop = }, {annotation[start:stop] = }, {annotation = }.\n"
            f"ValueError: {e}"
        )
    return t

def _parse_type_args(
    annotation: str, start: int, stop: int
) -> tuple[tuple[ParsedType, ...], bool]:
    # print(f"_parse_type_args({annotation!r}, {start}, {stop})")
    # print(f"{annotation[start:stop] = }")
    if annotation[start:stop].strip() == "()":
        return (), False
    bracket_ranges = tuple(_outer_bracket_ranges(annotation, start, stop))
    comma_idxs = tuple(_find_outside_ranges(",", annotation, bracket_ranges, start, stop))
    if not comma_idxs:
        return (_parse_type(annotation, start, stop),), False
    arg_ranges = tuple(_split_at(comma_idxs, start, stop))
    args: list[ParsedType] = []
    variadic = False
    for i, r in enumerate(arg_ranges):
        arg = _parse_type(annotation, r.start, r.stop)
        if arg.name == "...":
            if i < len(arg_ranges)-1:
                raise ValueError(
                    "Ellipsis found in args, but not in last position,  at "
                    f"{start = }, {stop = }, {annotation[start:stop] = }, {annotation = }"
                )
            variadic = True
            break
        args.append(arg)
    return tuple(args), variadic

def _parse_atom_type(annotation: str, start: int, stop: int) -> ParsedType:
    # print(f"_parse_atom_type({annotation!r}, {start}, {stop})")
    # print(f"{annotation[start:stop] = }")
    while start < stop and annotation[start].isspace():
        start += 1
    while start < stop and annotation[stop-1].isspace():
            stop -= 1
    assert annotation[start:stop] == annotation[start:stop].strip()
    bracket_ranges = tuple(_outer_bracket_ranges(annotation, start, stop))
    if len(bracket_ranges) > 1:
        raise ValueError(
            "Non-union type must take the form 'TypeName' or 'TypeName[Args]'. "
            f"Found multiple outer bracket pairs at "
            f"{start = }, {stop = }, {annotation[start:stop] = }, {annotation = }"
        )
    r = bracket_ranges[0] if bracket_ranges else None
    if r is None:
        name = annotation[start:stop].strip()
        return _parsed_type(annotation, start, stop, name)
    elif r.stop < stop:
        raise ValueError(
            "Non-union type must take the form 'TypeName' or 'TypeName[Args]'. "
            "Found text after bracket pair at "
            f"{start = }, {stop = }, {annotation[start:stop] = }, {annotation = }"
        )
    name = annotation[start:r.start].strip()
    if name in ("Literal", "typing.Literal"):
        return _parsed_type(annotation, start, stop, name, annotation[r.start+1:r.stop-1])
    if not name:
        raise ValueError(
            f"Found empty type name at "
            f"{start = }, {stop = }, {annotation[start:stop] = }, {annotation = }"
        )
    args, variadic = _parse_type_args(annotation, r.start+1, r.stop-1)
    return _parsed_type(annotation, start, stop, name, args, variadic)

def _parse_type(annotation: str, start: int, stop: int) -> ParsedType:
    # print(f"_parse_type({annotation!r}, {start}, {stop})")
    # print(f"{annotation[start:stop] = }")
    bracket_ranges = tuple(_outer_bracket_ranges(annotation, start, stop))
    ors_idxs = tuple(_find_outside_ranges("|", annotation, bracket_ranges, start, stop))
    if not ors_idxs:
        return _parse_atom_type(annotation, start, stop)
    member_ranges = tuple(_split_at(ors_idxs, start, stop))
    name = "UnionType"
    args = tuple(
        _parse_atom_type(annotation, r.start, r.stop)
        for r in member_ranges
    )
    return _parsed_type(annotation, start, stop, name, args)

def parse_type(annotation: str) -> ParsedType:
    return _parse_type(annotation, 0, len(annotation))


### 2. Track Classes Known to Autodoc ###

_class_dict: dict[str, type] = {}

def class_tracking_handler(app: Sphinx, what: str, fullname: str, obj: Any, options: Any, lines: list[str]) -> None:
    r"""
        Handler for Sphinx Autodoc's event
        `autodoc-process-docstring <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring>`_
        which keeps track of classes known to autodoc, for used in other handlers.
    """
    if what != "class":
        return
    _class_dict[fullname] = obj


### 3. Document Function Parameter and Return Types ###

def _sigdoc(fun: FunctionType, lines: list[str]) -> None:
    r"""
        Returns doclines documenting the parameter and return type of the given function
    """
    # pylint: disable = too-many-branches
    doc = "\n".join(lines)
    lines.append("")
    # FIXME: if an :rtype: line already exists, remove it here and re-append it after all param type lines.
    globalns = fun.__globals__
    sig = inspect.signature(fun)
    for p in sig.parameters.values():
        annotation = p.annotation
        if annotation == p.empty:
            continue
        if not isinstance(annotation, str):
            if isinstance(annotation, ForwardRef):
                annotation = annotation.__forward_arg__
            else:
                logger.warning(
                    f"Found non-string annotation: {repr(annotation)}."
                    " Did you forget to import annotation from __future__?."
                )
                annotation = str(annotation)
        try:
            t = parse_type(annotation)
            tx = t.crossref(globalns)
        except Exception as e:
            logger.error(
                f"Parsing param {p.name!r} type annotation {annotation!r},"
                f" {type(e)}: {e}"
            )
            tx = annotation
        default = p.default if p.default != p.empty else None
        is_args = p.kind == p.VAR_POSITIONAL
        is_kwargs = p.kind == p.VAR_KEYWORD
        if is_args:
            extra_info = "variadic positional"
        elif is_kwargs:
            extra_info = "variadic keyword"
        elif default is not None:
            default_str = default.__qualname__ if isinstance(default, FunctionType) else repr(default)
            extra_info = f"default = ``{default_str}``"
        else:
            extra_info = None
        if extra_info is None:
            line = f":type {p.name}: {tx}"
        else:
            line = f":type {p.name}: {tx}; {extra_info}"
        if f":param {p.name}:" not in doc:
            lines.append(f":param {p.name}:")
        if f":type {p.name}:" not in doc:
            lines.append(line)
    if sig.return_annotation == sig.empty:
        return

    try:
        t = parse_type(sig.return_annotation)
        tx = t.crossref(globalns)
    except Exception as e:
        logger.error(
            f"Parsing return type annotation {annotation!r},"
            f" {type(e)}: {e}"
        )
        tx = annotation
    line = f":rtype: {tx}"
    if ":rtype:" not in doc:
        lines.append(line)

def signature_doc_handler(app: Sphinx, what: str, fullname: str, obj: Any, options: Any, lines: list[str]) -> None:
    r"""
        Handler for Sphinx Autodoc's event
        `autodoc-process-docstring <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring>`_
        which automatically documents parameter types and return type for
        functions, methods and properties.
    """
    # pylint: disable = too-many-arguments
    if what not in ("function", "method", "property"):
        return
    if what == "property":
        fun: FunctionType = obj.fget
    else:
        fun = obj
    try:
        _sigdoc(fun, lines)
    except Exception as e:
        msg = (
            f"In signature_doc_handler(app, {what = }, {fullname = }, ...),"
            f" raised {type(e)}: {e}"
        )
        logger.error(msg)

### 4. Document Attribute Types ###

def attr_doc_handler(app: Sphinx, what: str, fullname: str, obj: Any, options: Any, lines: list[str]) -> None:
    r"""
        Handler for Sphinx Autodoc's event
        `autodoc-process-docstring <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring>`_
        which automatically documents the type of attributes, including those
        defined by typed descriptors.
    """
    # pylint: disable = too-many-arguments
    if what != "attribute":
        return
    try:
        attrname = fullname.split(".")[-1]
        classname = ".".join(fullname.split(".")[:-1])
        parent_class = _class_dict.get(classname)
        if parent_class is not None:
            annotations = parent_class.__annotations__
            if attrname in annotations:
                type_annotation = annotations[attrname]
            elif isinstance(obj, TypedDescriptor):
                ty = obj.__descriptor_type__
                inspector = inspect_type(ty)
                type_annotation = inspector.type_annotation
            else:
                type_annotation = None
            if type_annotation is not None:
                if not isinstance(type_annotation, str):
                    if isinstance(type_annotation, ForwardRef):
                        type_annotation = type_annotation.__forward_arg__
                    else:
                        logger.warning(
                            f"Found non-string annotation: {repr(type_annotation)}."
                            " Did you forget to import annotation from __future__?."
                        )
                        type_annotation = str(type_annotation)
                t = parse_type(type_annotation)
                tx = t.crossref()
                if ":rtype:" not in "\n".join("lines"):
                    lines.append(f":rtype: {tx}")
    except Exception as e:
        msg = (
            f"In attr_doc_handler(app, {what = }, {fullname = }, ...),"
            f" raised {type(e)}: {e}\n{"\n".join(traceback.format_tb(e.__traceback__))}"
        )
        logger.error(msg)


### 5. Replace Cross-references with Fully Qualified Names ###

def simple_crossref_pattern(name: str) -> re.Pattern[str]:
    r"""
        Pattern for simple imports:

        .. code-block :: python

            f":{role}:`{name}`"        # e.g. ":class:`MyClass`"
            f":{role}:`~{name}`"       # e.g. ":class:`~MyClass`"
            f":{role}:`{name}{tail}`"  # e.g. ":attr:`MyClass.my_property.my_subproperty`"
            f":{role}:`~{name}{tail}`" # e.g. ":attr:`~MyClass.my_property.my_subproperty`"

    """
    return re.compile(rf":([a-z]+):`(~)?{name}(\.[\.a-zA-Z0-9_]+)?`")

def simple_crossref_repl(name: str, fullname: str) -> Callable[[re.Match[str]], str]:
    r"""
        Replacement function for the pattern generated by :func:`simple_crossref_pattern`:

        .. code-block :: python

            f":{role}:`~{fullname}`"                    # e.g. ":class:`~mymod.mysubmod.MyClass`"
            f":{role}:`~{fullname}`"                    # e.g. ":class:`~mymod.mysubmod.MyClass`"
            f":{role}:`{name}{tail}<{fullname}{tail}>`" # e.g. ":attr:`MyClass.my_property.my_subproperty<mymod.mysubmod.MyClass.my_property.my_subproperty>`"
            f":{role}:`~{fullname}{tail}`"              # e.g. ":attr:`~mymod.mysubmod.MyClass.my_property.my_subproperty`"

    """
    def repl(match: re.Match[str]) -> str:
        role = match[1]
        short = match[2] is not None
        tail = match[3]
        if tail is None:
            return f":{role}:`~{fullname}`"
        if short:
            return f":{role}:`~{fullname}{tail}`"
        return f":{role}:`{name}{tail}<{fullname}{tail}>`"
    return repl

def labelled_crossref_pattern(name: str) -> re.Pattern[str]:
    r"""
        Pattern for labelled imports:

        .. code-block :: python

            f":{role}:`{label}<{name}>`"       # e.g. ":class:`my class<MyClass>`"
            f":{role}:`{label}<{name}{tail}>`" # e.g. ":attr:`my_property<MyClass.my_property>`"

    """
    return re.compile(rf":([a-z]+):`([\.a-zA-Z0-9_]+)<{name}(\.[\.a-zA-Z0-9_]+)?>`")

def labelled_crossref_repl(name: str, fullname: str) -> Callable[[re.Match[str]], str]:
    r"""
        Replacement function for the pattern generated by :func:`labelled_crossref_pattern`:

        .. code-block :: python

            f":{role}:`{label}<{fullname}>`"       # e.g. ":class:`my class<mymod.mysubmod.MyClass>`"
            f":{role}:`{label}<{fullname}{tail}>`" # e.g. ":attr:`my_property<mymod.mysubmod.MyClass.my_property>`"

    """
    def repl(match: re.Match[str]) -> str:
        role = match[1]
        label = match[2]
        tail = match[3]
        if tail is None:
            return f":{role}:`{label}<{fullname}>`"
        return f":{role}:`{label}<{fullname}{tail}>`"
    return repl

_crossref_subs: list[tuple[Callable[[str], re.Pattern[str]],
                           Callable[[str, str], Callable[[re.Match[str]], str]]]] = [
    (simple_crossref_pattern, simple_crossref_repl),
    (labelled_crossref_pattern, labelled_crossref_repl),
]
r"""
    Substitution patterns and replacement functions for various kinds of cross-reference scenarios.
"""

def _get_module_by_name(modname: str) -> ModuleType:
    r"""
        Gathers a module object by name.
    """
    # pylint: disable = exec-used, eval-used
    namespace: dict[str, Any] = {}
    exec(f"import {modname} as _mymodule", namespace)
    mod: ModuleType = namespace["_mymodule"]
    # exec(f"import {modname.split('.')[0]}")
    # mod: ModuleType = eval(modname)
    if not isinstance(mod, ModuleType):
        return None
    return mod

def _get_obj_mod(app: Sphinx, what: str, fullname: str, obj: Any) -> Optional[ModuleType]:
    r"""
        Gathers the containing module for the given ``obj``.
    """
    autodoc_type_aliases = app.config.__dict__.get("autodoc_type_aliases")
    name = fullname.split(".")[-1]
    obj_mod: Optional[ModuleType]
    if autodoc_type_aliases is not None:
        if name in autodoc_type_aliases and fullname == autodoc_type_aliases[name]:
            modname = ".".join(fullname.split(".")[:-1])
            obj_mod = _get_module_by_name(modname)
            return obj_mod
    if what == "module":
        obj_mod = obj
    elif what in ("function", "class", "method", "exception"):
        obj_mod = inspect.getmodule(obj)
    elif what == "property":
        obj_mod = inspect.getmodule(obj.fget)
    elif what == "data":
        modname = ".".join(fullname.split(".")[:-1])
        obj_mod = _get_module_by_name(modname)
    elif what == "attribute":
        modname = ".".join(fullname.split(".")[:-2])
        obj_mod = _get_module_by_name(modname)
    else:
        logger.warning(
            f"Encountered unexpected value for what = {what} at fullname = {fullname}"
        )
        obj_mod = None
    return obj_mod

def _build_fullname_dict(app: Sphinx, fullname: str, obj_mod: Optional[ModuleType], ) -> dict[str, str]:
    r"""
        Builds a dictionary of substitutions from module global names to their fully qualified names,
        based on :func:`inspect.getmodule` and `autodoc_type_aliases` (if specified in the Sphinx app config).
    """
    autodoc_type_aliases = app.config.__dict__.get("autodoc_type_aliases")
    fullname_dict: dict[str, str] = {}
    if obj_mod is not None:
        globalns = obj_mod.__dict__
        for g_name, g_obj in globalns.items():
            if isinstance(g_obj, (FunctionType, type)):
                g_mod = inspect.getmodule(g_obj)
            elif isinstance(g_obj, ModuleType):
                g_mod = g_obj
            else:
                g_mod = inspect.getmodule(g_obj)
            if g_mod is None or g_mod == obj_mod:
                continue
            if g_name not in g_mod.__dict__:
                continue
            g_modname = g_mod.__name__
            fullname_dict[g_name] = f"{g_modname}.{g_name}"
    if autodoc_type_aliases is not None:
        for a_name, a_fullname in autodoc_type_aliases.items():
            if a_name not in fullname_dict:
                fullname_dict[a_name] = a_fullname
    return fullname_dict

def local_crossref_handler(app: Sphinx, what: str, fullname: str, obj: Any, options: Any, lines: list[str]) -> None:
    r"""
        Handler for Sphinx Autodoc's event
        `autodoc-process-docstring <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring>`_
        which replaces cross-references specified in terms of module globals with their fully qualified version.
    """
    # pylint: disable = too-many-arguments, too-many-locals
    obj_mod = _get_obj_mod(app, what, fullname, obj)
    fullname_dict = _build_fullname_dict(app, fullname, obj_mod)
    for sub_name, sub_fullname in fullname_dict.items():
        for idx, line in enumerate(lines):
            for pattern_fun, repl_fun in _crossref_subs:
                pattern = pattern_fun(sub_name)
                repl = repl_fun(sub_name, sub_fullname)
                line = re.sub(pattern, repl, line)
            lines[idx] = line


### 6. Register Sphinx Event Handlers ###

def setup(app: Sphinx) -> None:
    r"""
        Registers handlers for Sphinx events.
    """
    app.connect("autodoc-process-docstring", class_tracking_handler)
    app.connect("autodoc-process-docstring", signature_doc_handler)
    app.connect("autodoc-process-docstring", attr_doc_handler)
    app.connect("autodoc-process-docstring", local_crossref_handler)
