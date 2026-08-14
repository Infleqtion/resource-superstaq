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

import typing

import cirq
from qldpc import codes
from qldpc.objects import Pauli


def _validate_id(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


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
        self._x_support = self._validate_support(x_support, "x_support")
        self._z_support = self._validate_support(z_support, "z_support")
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
        return self._x_support

    @property
    def z_support(self) -> frozenset[int]:
        return self._z_support

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, int]:
        return self.patch_id, self.logical_index

    @staticmethod
    def _validate_support(support: typing.Iterable[int], name: str) -> frozenset[int]:
        qubits = set()
        for qubit in support:
            if not isinstance(qubit, int) or isinstance(qubit, bool):
                raise TypeError(f"LogicalQubit {name} entries must be integers.")
            if qubit < 0:
                raise ValueError(f"LogicalQubit {name} entries must be nonnegative.")
            qubits.add(qubit)
        return frozenset(qubits)

    def __str__(self) -> str:
        return f"{self.patch_id}:{self.logical_index}"

    def __repr__(self) -> str:
        return f"codepatch.LogicalQubit({self.patch_id!r}, {self.logical_index!r})"


class CodePatch:
    """Common metadata for an error-correcting code patch."""

    def __init__(
        self,
        patch_id: int,
        n: int,
        k: int,
        d: int | float | None,
        num_measure_qubits: int,
        logical_qubits: typing.Iterable[LogicalQubit],
    ) -> None:
        self._patch_id = _validate_id(patch_id, "patch_id")
        self.n = n
        self.k = k
        self.d = d
        self.num_data_qubits = self.n
        self.num_measure_qubits = num_measure_qubits
        self.logical_qubits = tuple(logical_qubits)

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def num_physical_qubits(self) -> int:
        """The total number of data and measurement qubits in the code patch."""
        return self.num_data_qubits + self.num_measure_qubits


class RotatedSurfaceCodePatch(CodePatch):
    """A rotated surface-code patch backed by qLDPC metadata."""

    def __init__(self, patch_id: int, d: int) -> None:
        assert (d - 1) % 2 == 0, "CodePatches must be odd distance"
        self._qldpc_code = codes.SurfaceCode(d)
        n, k, code_distance = self._qldpc_code.get_code_params()
        super().__init__(
            patch_id=patch_id,
            n=int(n),
            k=int(k),
            d=code_distance,
            num_measure_qubits=int(self._qldpc_code.num_checks),
            logical_qubits=self._logical_qubits_from_qldpc_code(patch_id),
        )

    @property
    def qldpc_code(self) -> codes.SurfaceCode:
        return self._qldpc_code

    def num_x_stabs(self) -> int:
        """Return the number of X-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_x)

    def num_z_stabs(self) -> int:
        """Return the number of Z-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_z)

    def total_x_syndrome_cnots(self) -> int:
        """Return the data-check interactions needed to measure all X stabilizers."""
        return len(self.qldpc_code.matrix_x.nonzero()[0])

    def total_z_syndrome_cnots(self) -> int:
        """Return the data-check interactions needed to measure all Z stabilizers."""
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

    def __repr__(self) -> str:
        args = [
            f"patch_id={self.patch_id!r}",
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
        ]
        return f"codepatch.RotatedSurfaceCodePatch({', '.join(args)})"
