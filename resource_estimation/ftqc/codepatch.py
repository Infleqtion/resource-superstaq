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

from qldpc import codes
from qldpc.objects import Pauli


class LogicalQubit:
    """Metadata for a logical qubit primitive."""

    def __init__(
        self,
        x_support: typing.Iterable[int],
        z_support: typing.Iterable[int],
    ) -> None:
        self.x_support = self._validate_support(x_support, "x_support")
        self.z_support = self._validate_support(z_support, "z_support")
        if not self.x_support | self.z_support:
            raise ValueError("LogicalQubit supports must include at least one physical qubit.")

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


class CodePatch:
    """Metadata for a logical surface code patch."""

    def __init__(self, d: int) -> None:
        self._qldpc_code = codes.SurfaceCode(d)
        n, k, code_distance = self._qldpc_code.get_code_params()
        self.n = int(n)
        self.k = int(k)
        self.d = code_distance
        self.num_data_qubits = self.n
        self.num_measure_qubits = int(self._qldpc_code.num_checks)
        self.logical_qubits = self._logical_qubits_from_qldpc_code()

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def num_physical_qubits(self) -> int:
        """The total number of data and measurement qubits in the code patch."""
        return self.num_data_qubits + self.num_measure_qubits

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

    def _logical_qubits_from_qldpc_code(self) -> list[LogicalQubit]:
        logical_x = self.qldpc_code.get_logical_ops(Pauli.X)[0]
        logical_z = self.qldpc_code.get_logical_ops(Pauli.Z)[0]
        return [
            LogicalQubit(
                x_support=self._logical_op_support(logical_x),
                z_support=self._logical_op_support(logical_z),
            )
        ]

    @staticmethod
    def _logical_op_support(logical_op: typing.Iterable[int]) -> set[int]:
        return {index for index, value in enumerate(logical_op) if value}

    def __repr__(self) -> str:
        args = [
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
        ]
        return f"codepatch.CodePatch({', '.join(args)})"
