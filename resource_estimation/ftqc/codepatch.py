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
import numpy as np
from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y

from . import layout as lyt


def _validate_id(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


class PhysicalQubit(cirq.Qid):
    """A physical qubit represented as a wire in a Cirq circuit"""

    def __init__(
        self,
        patch_id: int,
        local_coord: tuple[int, int],
        role: int,
        info: dict[str, int] | None = None,
    ) -> None:
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
    def role(self) -> int:
        return self._role

    @property
    def info(self) -> dict[str, int] | None:
        return self._info

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(
        self,
    ) -> tuple[int, int, int, int]:
        return self.patch_id, self.local_coord[0], self.local_coord[1], self.role

    def __str__(self) -> str:
        return f"{self.patch_id}:{self.local_coord[0], self.local_coord[1], self.role}"

    def __repr__(self) -> str:
        return f"cpa.PhysicalQubit({self.patch_id!r}, {self.local_coord!r}, {self.role})"


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
        return f"cpa.LogicalQubit({self.patch_id!r}, {self.logical_index!r})"


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
    def footprint(self) -> lyt.Rect:
        qubits = self.default_layout()
        rows = [q.local_coord[0] for q in qubits]
        cols = [q.local_coord[1] for q in qubits]

        return lyt.Rect(
            rows=max(rows) - min(rows) + 1,
            cols=max(cols) - min(cols) + 1,
        )

    @abc.abstractmethod
    def default_layout(self) -> list[PhysicalQubit]:  # pragma: no cover
        """Return the physical qubits in their default layout."""
        ...


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
        return f"cpa.RotatedSurfaceCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        def transform(x: int, y: int) -> tuple[int, int]:
            return (
                (x + y) // 2,
                (x - y + 2 * (self.d - 1)) // 2,
            )

        qubits = []
        # Data Qubits
        for r in range(self.d):
            for c in range(self.d):
                local_coord = transform(2 * r, 2 * c)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=0)
                )
        # Interior weight 4 checks
        for r in range(self.d - 1):
            for c in range(self.d - 1):
                role = 1 if (r + c) % 2 == 0 else 2
                local_coord = transform(2 * r + 1, 2 * c + 1)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role)
                )
        # Boundary weight 2 checks
        for k in range(self.d - 1):
            if k % 2 == 0:
                # Top Z Boundary
                local_coord = transform(-1, 2 * k + 1)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=2)
                )
                # Right X Boundary
                local_coord = transform(2 * k + 1, 2 * self.d - 1)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=1)
                )
            else:
                # Bottom Z Boundary
                local_coord = transform(2 * self.d - 1, 2 * k + 1)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=2)
                )
                # Left X Boundary
                local_coord = transform(2 * k + 1, -1)
                qubits.append(
                    PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=1)
                )
        return qubits


class BBCodePatch(CSSCodePatch):
    """A Bivariate Bicyle code patch generated from the qldpc library"""

    def __init__(self, patch_id: int, l: int, m: int):
        # polynomial used in https://arxiv.org/abs/2606.25011v1
        orders = {x: l, y: m}
        poly_a = 1 + y**3 + x * y**2 + x * y**4
        poly_b = 1 + y**-3 + x**-1 * y**-2 + x**-1 * y**-4
        qldpc_code = codes.BBCode(orders, poly_a, poly_b)
        super().__init__(patch_id=patch_id, qldpc_code=qldpc_code)

    @property
    def qldpc_code(self) -> codes.BBCode:
        assert isinstance(self._qldpc_code, codes.BBCode)
        return self._qldpc_code

    def __repr__(self) -> str:
        args = [
            f"patch_id={self.patch_id!r}",
            f"l={self.qldpc_code.orders[0]!r}",
            f"m={self.qldpc_code.orders[1]!r}",
        ]
        return f"cpa.BBCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        """Produces a plaquette-based physical layout for the physical qubits in the code patch"""
        qubits = []
        l, m = self.qldpc_code.orders
        for index in range(self.n):
            half, col, row = map(int, np.unravel_index(index, (2, l, m)))
            if half == 0:  # L data + X check
                data_local_coord = (2 * row, 2 * col)
                meas_local_coord = (2 * row, 2 * col + 1)
                meas_role = 1
            else:  # R data + Z check
                data_local_coord = (2 * row + 1, 2 * col + 1)
                meas_local_coord = (2 * row + 1, 2 * col)
                meas_role = 2
            data_qubit = PhysicalQubit(patch_id=self.patch_id, local_coord=data_local_coord, role=0)
            meas_qubit = PhysicalQubit(
                patch_id=self.patch_id, local_coord=meas_local_coord, role=meas_role
            )
            qubits.extend([data_qubit, meas_qubit])
        return qubits
