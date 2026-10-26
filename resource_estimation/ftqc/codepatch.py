# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import abc
import typing

import cirq
from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y
import numpy as np
from dataclasses import dataclass, field


def _validate_id(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value

class PhysicalQubit(cirq.Qid):
    """A physical qubit represented as a wire in a Cirq circuit"""
    def __init__(self, patch_id: int, local_coord: tuple[int, int], role: tuple[typing.Literal['data', 'measure'], typing.Literal['X', 'Z'] | None], info: dict[str, int] | None = None) -> None:
        """
        patch_id: the patch id of the CodePatch object to which this physical qubit belongs
        local_coord: local coordinate tuple representing the position of this qubit in the footprint of the CodePatch
        role: Designation of the job of the physical qubit as a data or measure qubit
        """
        self._patch_id = _validate_id(patch_id, "patch_id")
        self._local_coord = local_coord
        self._role = role
        self._info = info

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    def local_coord(self) -> tuple[int, int]:
        return self._local_coord

    @property
    def role(self) -> tuple[typing.Literal['data', 'measure'], typing.Literal['X', 'Z'] | None]:
        return self._role

    @property
    def info(self) -> dict[str, int] | None:
        return self._info
    
    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, int, int, typing.Literal["data", "measure"], typing.Literal["X", "Z"] | None]:
        return self.patch_id, self.local_coord[0], self.local_coord[1], self.role[0], self.role[1]

    def __str__(self) -> str:
        return f"{self.patch_id}:({self.local_coord[0], self.local_coord[1]}, {self.role[1] if self.role[1] else self.role[0]})"

    def __repr__(self) -> str:
        return f"codepatch.PhysicalQubit({self.patch_id!r}, {self.local_coord!r}, {self.role})"    


class LogicalQubit(cirq.Qid):
    """A logical qubit represented as a wire in a Cirq circuit."""

    def __init__(
        self,
        patch_id: int,
        logical_index: int,
        x_support: typing.Iterable[int],
        z_support: typing.Iterable[int],
    ) -> None:
        self._patch_id = _validate_id(patch_id, "patch_id")
        self._logical_index = _validate_id(logical_index, "logical_index")
        self._x_support = self._normalize_support(x_support, "x_support")
        self._z_support = self._normalize_support(z_support, "z_support")
        if not self.x_support | self.z_support:
            raise ValueError("LogicalQubit supports must include at least one physical qubit.")

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    def logical_index(self) -> int:
        return self._logical_index

    @property
    def x_support(self) -> frozenset[int]:
        """Return the physical-qubit indices supporting the logical X operator."""
        return self._x_support

    @property
    def z_support(self) -> frozenset[int]:
        """Return the physical-qubit indices supporting the logical Z operator."""
        return self._z_support

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, int]:
        return self.patch_id, self.logical_index

    @staticmethod
    def _normalize_support(support: typing.Iterable[int], name: str) -> frozenset[int]:
        return frozenset(_validate_id(qubit, f"LogicalQubit {name} entry") for qubit in support)

    def __str__(self) -> str:
        return f"{self.patch_id}:{self.logical_index}"

    def __repr__(self) -> str:
        return f"codepatch.LogicalQubit({self.patch_id!r}, {self.logical_index!r})"


class CodePatch(abc.ABC):
    """Common metadata for an error-correcting code patch."""

    def __init__(
        self,
        patch_id: int,
        n: int,
        k: int,
        d: int,
    ) -> None:
        self._patch_id = _validate_id(patch_id, "patch_id")
        self.n = n
        self.k = k
        self.d = d
        self.num_data_qubits = self.n

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    @abc.abstractmethod
    def num_measure_qubits(self) -> int:
        """Return the number of measurement qubits in the code patch."""

    @property
    @abc.abstractmethod
    def logical_qubits(self) -> tuple[LogicalQubit, ...]:
        """Return the logical qubits encoded by the code patch."""

    @property
    def num_logical_qubits(self) -> int:
        """Return the number of logical qubits encoded by the code patch."""
        return len(self.logical_qubits)

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def num_physical_qubits(self) -> int:
        """The total number of data and measurement qubits in the code patch."""
        return self.num_data_qubits + self.num_measure_qubits

    @property
    def footprint(self) -> Rect:
        qubits = self.default_layout()
        rows = [q.local_coord[0] for q in qubits]
        cols = [q.local_coord[1] for q in qubits]

        return Rect(
            rows=max(rows) - min(rows) + 1,
            cols=max(cols) - min(cols) + 1,
        )   


class CSSCodePatch(CodePatch):
    """Metadata shared by qLDPC CSS code patches."""

    @abc.abstractmethod
    def __init__(self, patch_id: int, qldpc_code: codes.CSSCode) -> None:
        self._qldpc_code = qldpc_code
        n, k, code_distance = self._qldpc_code.get_code_params()
        self._logical_qubits = tuple(self._logical_qubits_from_qldpc_code(patch_id))
        super().__init__(
            patch_id=patch_id,
            n=int(n),
            k=int(k),
            d=int(code_distance),
        )

    @property
    def qldpc_code(self) -> codes.CSSCode:
        return self._qldpc_code

    @property
    def logical_qubits(self) -> tuple[LogicalQubit, ...]:
        return self._logical_qubits

    @property
    def num_measure_qubits(self) -> int:
        return int(self.qldpc_code.num_checks)

    def num_x_stabilizers(self) -> int:
        """Return the number of X-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_x)

    def num_z_stabilizers(self) -> int:
        """Return the number of Z-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_z)

    def total_x_check_weight(self) -> int:
        """Return the total weight of all X-type stabilizer checks."""
        return len(self.qldpc_code.matrix_x.nonzero()[0])

    def total_z_check_weight(self) -> int:
        """Return the total weight of all Z-type stabilizer checks."""
        return len(self.qldpc_code.matrix_z.nonzero()[0])

    def _logical_qubits_from_qldpc_code(self, patch_id: int) -> list[LogicalQubit]:
        logical_x = self.qldpc_code.get_logical_ops(Pauli.X)
        logical_z = self.qldpc_code.get_logical_ops(Pauli.Z)
        return [
            LogicalQubit(
                patch_id=patch_id,
                logical_index=index,
                x_support=self._logical_op_support(logical_x[index]),
                z_support=self._logical_op_support(logical_z[index]),
            )
            for index in range(len(logical_x))
        ]

    @staticmethod
    def _logical_op_support(logical_op: typing.Iterable[int]) -> set[int]:
        return {index for index, value in enumerate(logical_op) if value}


class RotatedSurfaceCodePatch(CSSCodePatch):
    """A rotated surface-code patch generated from the qldpc library."""

    def __init__(self, patch_id: int, d: int) -> None:
        if (d - 1) % 2 != 0:
            raise ValueError("CodePatches must be odd distance")
        super().__init__(patch_id=patch_id, qldpc_code=codes.SurfaceCode(d))

    def __repr__(self) -> str:
        args = [
            f"patch_id={self.patch_id!r}",
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
        ]
        return f"codepatch.RotatedSurfaceCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        def transform(x: int, y: int) -> tuple[int, int]:
            return (
                (x + y)//2,
                (x - y + 2 * (self.d - 1))//2,
            )
        qubits = []
        # Data Qubits
        for r in range(self.d):
            for c in range(self.d):
                local_coord = transform(2*r, 2*c)
                role = ("data", None)
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        # Interior weight 4 checks
        for r in range(self.d-1):
            for c in range(self.d-1):
                basis = "X" if (r+c)%2==0 else "Z"
                role = ("measure", basis)
                local_coord = transform(2*r+1, 2*c+1)
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        # Boundary weight 2 checks
        for k in range(self.d - 1):
            if k%2==0:
                # Top Z Boundary
                local_coord = transform(-1, 2*k+1)
                role = ("measure", "Z")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
                # Right X Boundary
                local_coord = transform(2*k+1, 2*self.d-1)
                role = ("measure", "X")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
            else:
                # Bottom Z Boundary
                local_coord = transform(2*self.d-1, 2*k+1)
                role = ("measure", "Z")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
                # Left X Boundary
                local_coord = transform(2*k+1, -1)
                role = ("measure", "X")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        return qubits

class BBCodePatch(CSSCodePatch):
    """A Bivariate Bicyle code patch generated from the qldpc library"""

    def __init__(self, patch_id: int, l: int, m: int):
        ## Using the polynomial from Quera STAR paper  ##
        # Probably needs better logic for choosing coeffs
        orders = {x: l, y: m}
        poly_a = 1 + y**3 + x * y**2 + x * y**4
        poly_b = 1 + y**-3 + x**-1 * y**-2 + x**-1 * y**-4
        qldpc_code = codes.BBCode(orders, poly_a, poly_b)
        super().__init__(patch_id=patch_id, qldpc_code=qldpc_code)

    def __repr__(self) -> str:
        args = [
            f"patch_id={self.patch_id!r}",
            f"l={self.qldpc_code.orders[0]!r}",
            f"m={self.qldpc_code.orders[1]!r}",
        ]
        return f"codepatch.BBCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        """Produces a plaquette-based physical layout for the physical qubits in the code patch"""
        qubits = []
        l, m = self.qldpc_code.orders
        for index in range(self.n):
            half, col, row = map(int, np.unravel_index(index, (2, l, m)))
            if half == 0:  # L data + X check
                data_local_coord = (2 * row, 2 * col)
                meas_local_coord = (2 * row, 2 * col + 1)
                basis = "X"
            else:  # R data + Z check
                data_local_coord = (2 * row + 1, 2 * col + 1)
                meas_local_coord = (2 * row + 1, 2 * col)
                basis = "Z"
            data_qubit = PhysicalQubit(patch_id=self.patch_id, local_coord=data_local_coord, role=("data", None))
            meas_qubit = PhysicalQubit(patch_id=self.patch_id, local_coord=meas_local_coord, role=("measure", basis))
            qubits.extend([data_qubit, meas_qubit])
        return qubits


@dataclass(frozen=True)
class Rect:
    rows: int
    cols: int
    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rect dimensions must be positive")

class Placeable(typing.Protocol):
    @property
    def footprint(self) -> Rect:
        ...

@dataclass(frozen=True)
class Placement:
    element: Placeable
    row: int
    col: int

    @property
    def row_end(self) -> int:
        return self.row + self.element.footprint.rows

    @property
    def col_end(self) -> int:
        return self.col + self.element.footprint.cols

    def overlaps(self, placement: Placement):
        return not (
            placement.row_end <= self.row
            or self.row_end <= placement.row
            or placement.col_end <= self.col
            or self.col_end <= placement.col
        )

@dataclass
class Zone:
    rows: int
    cols: int
    role: str

    @property
    def footprint(self) -> Rect:
        return Rect(self.rows, self.cols)

@dataclass
class Layout2:
    rows: int
    cols: int
    spacing: float = 1.0
    placements: dict[int, Placement] = field(default_factory=dict)
    regions: dict[int, Zone] = field(default_factory=dict)

    def add(self, element: Placeable, row, col) -> int:
        """Adds a new proposed placement if possible, returning its unique identifier"""
        candidate_placement = Placement(element=element, row=row, col=col)
        if not self.in_bounds(placement=candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement):
            raise ValueError("This space is already occupied")
        next_id = max(self.placements, default=0) + 1
        self.placements[next_id] = candidate_placement
        return next_id

    def remove(self, placement_id: int):
        """Removes the Placement at placement_id"""
        del self.placements[placement_id]

    def move(self, placement_id: int, row: int, col: int) -> None:
        """Moves the anchor of a Placement with placement_id to the given row, col coordinate"""
        prior_placement = self.placements[placement_id]
        candidate_placement = Placement(element=prior_placement.element, row=row, col=col)
        if not self.in_bounds(candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement, exclude=placement_id):
            raise ValueError("This space is already occupied")
        self.placements[placement_id] = candidate_placement

    def global_coord(self, placement_id: int, local_coord: tuple[int, int]) -> tuple[int, int]:
        """Returns global coordinate for a local object within the Placement with the given placement_id"""
        placement = self.placements[placement_id]
        r, c = local_coord
        return placement.row + r, placement.col + c
        
    def occupied(self, placement: Placement, exclude: int | None = None) -> bool:
        """Returns True if the proposed Placement overlaps with any existing placement, subject to excluding one placement"""
        return any(placement.overlaps(prior_placement) for pid, prior_placement in self.placements.items() if pid != exclude)

    def in_bounds(self, placement: Placement):
        """Returns True if the proposed placement is within the bounds of the Layout"""
        return (
            placement.row >= 0
            and placement.col >= 0
            and placement.row_end <= self.rows
            and placement.col_end <= self.cols
        )

    def get_patch(self, patch_id: int):
        for placement in self.placements.values():
            element = placement.element
            if isinstance(element, CodePatch) and element.patch_id == patch_id:
                return element
        raise ValueError(f"Found no CodePatch with patch_id={patch_id} in Layout")
        
