"""Diagram contractions based on :mod:`cotengra`'s contraction utilities."""

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
from typing import Self, final

from .abc import Contraction
from ..diagrams import Box, Diagram, Wiring, Type

if __debug__:
    from typing_validation import validate

try:
    import cotengra as ctg  # type: ignore
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Library 'cotengra' must be installed to use 'tensorsat.contraction.cotengra'."
    )
