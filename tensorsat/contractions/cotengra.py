"""Contractions based on contraction trees from :mod:`cotengra`."""

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
from collections.abc import Callable
from typing import (
    Any,
    Literal,
    Self,
    Type as SubclassOf,
    cast,
    final,
)
from opt_einsum.paths import PathOptimizer  # type: ignore[import-untyped]

from .abc import Contraction
from .simple import ContractionPath
from ..diagrams import (
    Diagram,
    TensorLikeBox,
    TensorLikeBoxT_inv,
    TensorLikeType,
    Wiring,
)
from .._utils.misc import SupportsRichComparison

try:
    import cotengra as ctg  # type: ignore[import-untyped]
    from cotengra import ContractionTree
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For cotengra-based contractions, 'cotengra' must be installed."
    )

if __debug__:
    from typing_validation import validate


@final
class CotengraContraction(Contraction[Any, TensorLikeBoxT_inv]):
    """A contraction based on contraction trees from :mod:`cotengra`."""

    __tree: ContractionTree

    def __new__(
        cls,
        box_class: SubclassOf[TensorLikeBoxT_inv],
        wiring: Wiring,
        *,
        optimize: str | PathOptimizer = "auto",
        canonicalize: bool = True,
        sort_contraction_indices: bool = False,
    ) -> Self:
        assert validate(box_class, SubclassOf[TensorLikeBox])
        assert validate(wiring, Wiring)
        assert validate(optimize, str | PathOptimizer)
        assert validate(canonicalize, bool)
        assert validate(sort_contraction_indices, bool)
        if wiring.num_wires == 0:
            raise ValueError("Cannot define contraction for empty wiring.")
        for t in wiring.wire_types:
            if not isinstance(t, TensorLikeType):
                raise ValueError("Wiring must have tensor-like wire types.")
        wire_types = cast(tuple[TensorLikeType, ...], wiring.wire_types)
        self = super().__new__(cls, box_class, wiring)
        self.__tree = ctg.array_contract_tree(
            wiring.slot_wires_list,
            wiring.out_wires,
            {w: t.tensor_dim for w, t in enumerate(wire_types)},
            optimize=optimize,
            canonicalize=canonicalize,
            sort_contraction_indices=sort_contraction_indices,
        )
        return self

    @property
    def tree(self) -> ContractionTree:
        """Current contraction tree."""
        return self.__tree

    @property
    def path(self) -> ContractionPath:
        """Current contraction path."""
        path = self.tree.get_path()
        assert validate(path, ContractionPath)
        return cast(ContractionPath, path)

    def contract(
        self,
        diagram: Diagram,
        *,
        order: str | Callable[[frozenset[int]], SupportsRichComparison] | None = None,
        autojit: bool | Literal["auto"] = "auto",
        progbar: bool = False,
    ) -> TensorLikeBoxT_inv:
        """
        Validates and contracts the given diagram.

        For the meaning of keyword arguments,
        see :meth:`ContractionTree.contract <cotengra.core.ContractionTree.contract>`.

        :raises ValueError: if the diagram cannot be contracted.
        """
        self.validate(diagram)
        tree = self.tree
        if tree is None:
            raise ValueError()
        box = tree.contract(
            diagram.boxes,
            order=order,
            prefer_einsum=True,
            autojit=autojit,
            progbar=progbar,
        )
        assert isinstance(box, TensorLikeBox)
        return cast(TensorLikeBoxT_inv, box)

    def __repr__(self) -> str:
        return "<CotengraContraction>"
