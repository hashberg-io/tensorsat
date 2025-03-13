"""
Simple contractions, based on explicit contraction paths.
"""

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
from collections.abc import Sequence
from typing import Any, Literal, Self, Type as SubclassOf, TypeAlias, cast, final
import opt_einsum  # type: ignore[import-untyped]

from .abc import Contraction
from ..diagrams import (
    Diagram,
    TensorLikeBox,
    TensorLikeBoxT_inv,
    TensorLikeType,
    Wire,
    Wires,
    Wiring,
)

if __debug__:
    from typing_validation import validate


ContractionPath = tuple[tuple[int, int], ...]
"""
Type alias for a contraction path, as a sequence of pairs of indices to be contracted.
Indices follow the convention of :func:`opt_einsum.contract_path` and
:func:`numpy.einsum_path`: an array of boxes is maintained, the indices of each
contraction in the path are used to select boxes in the array at the time when that
contraction is to be performed in the path, the boxes at those indices are removed,
contracted, and the result is appended to the end of the list.
"""

Contract2Args: TypeAlias = tuple[int, Wires, int, Wires, Wires]
"""
Type alias for data used to specify arguments to :func:`Box.contract2` calls as part
of a :meth:`~Contraction.contract` call on a :meth:`SimpleContraction`.
Essentially the same as the arguments to :func:`Box.contract2`, but with the ``lhs``
and ``rhs`` arguments replaced by indices ``lhs_idx`` and ``rhs_idx`` from the
underlying :obj:`ContractionPath`.

.. code-block:: python

    lhs_idx: int
    lhs_wires: Wires
    rhs_idx: int
    rhs_wires: Wires
    res_wires: Wires

"""

OptEinsumOptimize: TypeAlias = Literal[
    "optimal", "branch-all", "branch-2", "greedy", "auto", False, True
]
"""
Possible values that can be passed to the ``optimize`` argument
of :func:`opt_einsum.contract_path`
"""


@final
class SimpleContraction(Contraction[Any, TensorLikeBoxT_inv]):
    """A simple contraction based on an explicit contraction path."""

    @classmethod
    def using_opt_einsum[
        _S: TensorLikeBox
    ](
        cls,
        box_class: SubclassOf[_S],
        wiring: Wiring,
        optimize: OptEinsumOptimize = "auto",
    ) -> SimpleContraction[_S]:
        """
        Creates a simple contraction using :func:`opt_einsum.contract_path`.

        .. warning::

            Currently broken because of a bug in :func:`opt_einsum.contract_path`.
            Pull request to fix this is under review:
            https://github.com/dgasmith/opt_einsum/pull/247
        """
        assert validate(box_class, SubclassOf[TensorLikeBox])
        assert validate(wiring, Wiring)
        if wiring.num_wires == 0:
            raise ValueError("Cannot define contraction for empty wiring.")
        for t in wiring.wire_types:
            if not isinstance(t, TensorLikeType):
                raise ValueError("Wiring must have tensor-like wire types.")
        contract_path_operands: list[Any] = []
        wire_dims = [cast(TensorLikeType, t).tensor_dim for t in wiring.wire_types]
        for slot_wires in wiring.slot_wires_list:
            contract_path_operands.append(tuple(wire_dims[w] for w in slot_wires))
            contract_path_operands.append(slot_wires)
        contract_path_operands.append(wiring.out_wires)
        path, _ = opt_einsum.contract_path(
            *contract_path_operands,
            optimize=optimize,
            use_blas=False,
            shapes=True,
        )
        return cls._new(box_class, wiring, path)  # type: ignore

    @classmethod
    def contract_using_opt_einsum[
        _S: TensorLikeBox
    ](
        cls,
        box_class: SubclassOf[_S],
        diagram: Diagram,
        optimize: OptEinsumOptimize = "auto",
    ) -> _S:
        """
        Generates a contraction from the diagram's wiring
        using :meth:`SimpleContraction.using_opt_einsum`, then contracts the diagram.s
        """
        contraction = cls.using_opt_einsum(box_class, diagram.wiring, optimize=optimize)
        return contraction.contract(diagram)

    @classmethod
    def _new(
        cls,
        box_class: SubclassOf[TensorLikeBoxT_inv],
        wiring: Wiring,
        path: ContractionPath,
    ) -> Self:
        self = super().__new__(cls, box_class, wiring)
        if path:
            # == non-trivial contraction case ==
            # Compute arguments to contract2 calls:
            contract2_args: list[tuple[int, Wires, int, Wires, list[Wire]]] = []
            wires_list = list(wiring.slot_wires_list)
            for lhs_idx, rhs_idx in path:
                lhs_wires = wires_list.pop(lhs_idx)
                rhs_wires = wires_list.pop(rhs_idx - int(lhs_idx < rhs_idx))
                common_wireset = set(lhs_wires) & set(rhs_wires)
                if common_wireset:
                    lhs_only_wires = tuple(
                        w for w in lhs_wires if w not in common_wireset
                    )
                    rhs_only_wires = tuple(
                        w for w in rhs_wires if w not in common_wireset
                    )
                    common_wires = tuple(w for w in lhs_wires if w in common_wireset)
                    res_wires = lhs_only_wires + common_wires + rhs_only_wires
                else:
                    res_wires = lhs_wires + rhs_wires
                wires_list.append(res_wires)
                contract2_args.append(
                    (lhs_idx, lhs_wires, rhs_idx, rhs_wires, list(res_wires))
                )
            # Remove each wire not appearing in wiring's out wires from the
            # out_wire arg of the last contract2 call in which it features:
            final_res_wires = contract2_args[-1][-1]
            out_wires = wiring.out_wires
            discarded_wires = sorted(set(final_res_wires) - set(out_wires))
            for w in discarded_wires:
                for _, _, _, _, _res_wires in reversed(contract2_args):
                    if w in _res_wires:
                        _res_wires.remove(w)
                        break
            # Store contract2 calls and wires of last contraction result:
            self.__contract2_args = tuple(
                (lhs, lhs_wires, rhs, rhs_wires, tuple(_res_wires))
                for lhs, lhs_wires, rhs, rhs_wires, _res_wires in contract2_args
            )
            self.__box_out_wires = tuple(final_res_wires)
        else:
            # == trivial contraction cases ==
            # No contract2 calls necessary:
            self.__contract2_args = ()
            # There may be a single box, or none:
            if wiring.num_slots == 0:
                self.__box_out_wires = ()
            else:
                assert wiring.num_slots == 1
                self.__box_out_wires = wiring.slot_wires_list[0]
        # Store dangling wires in canonical order, if any:
        self.__dangling_wires = tuple(sorted(wiring.dangling_wires))
        return self

    __wiring: Wiring
    __contract2_args: tuple[Contract2Args, ...]
    __box_out_wires: Wires
    __dangling_wires: Wires

    def __new__(
        cls,
        box_class: SubclassOf[TensorLikeBoxT_inv],
        wiring: Wiring,
        path: Sequence[tuple[int, int]],
    ) -> Self:
        """
        Constructs a simple contraction for a given wiring, from a contraction path.
        """
        path = tuple(path)
        # Validate arguments:
        assert validate(box_class, SubclassOf[TensorLikeBox])
        assert validate(path, ContractionPath)
        assert validate(wiring, Wiring)
        if wiring.num_wires == 0:
            raise ValueError("Cannot define contraction for empty wiring.")
        for t in wiring.wire_types:
            if not isinstance(t, TensorLikeType):
                raise ValueError("Wiring must have tensor-like wire types.")
        n = wiring.num_slots
        if not path and n >= 2:
            raise ValueError(
                "Contraction path can only be empty for number of wiring slots <= 1."
            )
        if len(path) >= n:
            raise ValueError(
                "Path is too long for the given wiring:"
                f" there are {len(path)} contractions, but {n} slots to contract."
            )
        if len(path) != n - 1:
            raise ValueError(
                "Path does not fully contract the given wiring:"
                f" there are {len(path)} contractions, but {n} slots to contract."
            )
        for idx, (lhs, rhs) in enumerate(path):
            if not 0 <= lhs < n - idx:
                raise ValueError(
                    f"Invalid lhs for contraction {(lhs, rhs) = } at {idx = }."
                )
            if not 0 <= rhs < n - idx:
                raise ValueError(
                    f"Invalid rhs for contraction {(lhs, rhs) = } at {idx = }."
                )
        # Construct and return contraction:
        return cls._new(box_class, wiring, path)

    @property
    def wiring(self) -> Wiring:
        """The wiring of the contraction."""
        return self.__wiring

    @property
    def contract2_args(self) -> tuple[Contract2Args, ...]:
        """The arguments to contract2 calls in the contraction."""
        return self.__contract2_args

    def contract(self, diagram: Diagram) -> TensorLikeBoxT_inv:
        """
        Validates and contracts the given diagram.

        :raises ValueError: if the diagram cannot be contracted.
        """
        self.validate(diagram)
        box_class, wiring = self.box_class, self.wiring
        contract2 = box_class._contract2
        rewire = box_class._rewire
        spider = box_class._spider
        wire_types, out_wires = wiring.wire_types, wiring.out_wires
        contract2_args = self.contract2_args
        box_out_wires, dangling_wires = self.__box_out_wires, self.__dangling_wires
        # 1. Contract all boxes using contract2:
        boxes = cast(list[TensorLikeBoxT_inv], list(diagram.boxes))
        #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ may be removed with box-generic diagrams
        box: TensorLikeBoxT_inv | None = None
        if boxes:
            for lhs_idx, lhs_wires, rhs_idx, rhs_wires, res_wires in contract2_args:
                lhs = boxes.pop(lhs_idx)
                rhs = boxes.pop(rhs_idx - int(lhs_idx < rhs_idx))
                assert len(lhs.shape) == len(lhs_wires)
                assert len(rhs.shape) == len(rhs_wires)
                res = contract2(lhs, lhs_wires, rhs, rhs_wires, res_wires)
                boxes.append(res)
            assert len(boxes) == 1
            box = boxes[0]
            assert len(box.shape) == len(box_out_wires)
        # 2. If dangling wires present, add them using spider and prod:
        if dangling_wires:
            box_out_wires += dangling_wires
            box = box_class.prod(
                [boxes[0], *(spider(wire_types[w], 1) for w in dangling_wires)]
            )
        assert box is not None
        # 3. If output wires have repetition/altered order, rewire:
        if out_wires != box_out_wires:
            box = rewire(box, [box_out_wires.index(w) for w in out_wires])
        # 4. Return contracted box:
        assert box is not None
        return box
