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
import pytest
from qldpc import codes

import resource_estimation.ftqc.codepatch as codepatch


def test_logical_qubit() -> None:
    qubit = codepatch.LogicalQubit(x_support={0, 2}, z_support={2, 3})

    assert qubit.x_support == frozenset({0, 2})
    assert qubit.z_support == frozenset({2, 3})


def test_logical_qubit_rejects_missing_supports() -> None:
    with pytest.raises(TypeError):
        codepatch.LogicalQubit()  # type: ignore[call-arg]


def test_logical_qubit_rejects_empty_supports() -> None:
    with pytest.raises(ValueError, match="supports must include"):
        codepatch.LogicalQubit(x_support=set(), z_support=set())


@pytest.mark.parametrize("support", [{-1}, {0, -1}])
def test_logical_qubit_rejects_negative_support_entries(support: set[int]) -> None:
    with pytest.raises(ValueError, match="entries must be nonnegative"):
        codepatch.LogicalQubit(x_support=support, z_support={0})


@pytest.mark.parametrize("support", [{1.5}, {True}])
def test_logical_qubit_rejects_noninteger_support_entries(support: set[object]) -> None:
    with pytest.raises(TypeError, match="entries must be integers"):
        codepatch.LogicalQubit(x_support=support, z_support={0})  # type: ignore[arg-type]


def test_code_patch_surface_metadata() -> None:
    patch = codepatch.CodePatch(d=7)

    assert patch.code_params == (49, 1, 7)
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.num_physical_qubits == 97
    assert patch.num_x_stabs() == 24
    assert patch.num_z_stabs() == 24
    assert patch.total_x_syndrome_cnots() == 84
    assert patch.total_z_syndrome_cnots() == 84
    assert len(patch.logical_qubits) == 1
    assert repr(patch) == "codepatch.CodePatch(d=7, n=49, k=1)"

    patch = codepatch.CodePatch(d=5)

    assert patch.code_params == (25, 1, 5)
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.total_x_syndrome_cnots() == 40
    assert patch.total_z_syndrome_cnots() == 40
    assert len(patch.logical_qubits) == 1
    assert repr(patch) == "codepatch.CodePatch(d=5, n=25, k=1)"


def test_code_patch_surface_stabilizer_metadata_matches_qldpc() -> None:
    for d in [3, 5, 7]:
        patch = codepatch.CodePatch(d=d)
        qldpc_code = codes.SurfaceCode(d)

        assert patch.num_x_stabs() == qldpc_code.num_checks_x
        assert patch.num_z_stabs() == qldpc_code.num_checks_z
        assert patch.total_x_syndrome_cnots() == len(qldpc_code.matrix_x.nonzero()[0])
        assert patch.total_z_syndrome_cnots() == len(qldpc_code.matrix_z.nonzero()[0])


def test_code_patch_logical_qubits_from_css_logical_support() -> None:
    patch = codepatch.CodePatch(d=3)

    assert len(patch.logical_qubits) == 1
    assert patch.logical_qubits[0].x_support == frozenset({6, 7, 8})
    assert patch.logical_qubits[0].z_support == frozenset({0, 4, 8})


def test_code_patch_requires_distance() -> None:
    with pytest.raises(TypeError):
        codepatch.CodePatch()  # type: ignore[call-arg]


@pytest.mark.parametrize("d", (3, 5, 7))
def test_surface_code_patch_counts_match_legacy(d: int) -> None:
    patch = codepatch.CodePatch(d=d)

    assert patch.num_data_qubits == d**2
    assert patch.num_measure_qubits == d**2 - 1
    assert patch.num_physical_qubits == 2 * d**2 - 1
    assert patch.num_x_stabs() == (d**2 - 1) // 2
    assert patch.num_z_stabs() == (d**2 - 1) // 2
    assert patch.total_x_syndrome_cnots() == 2 * d * (d - 1)
    assert patch.total_z_syndrome_cnots() == 2 * d * (d - 1)
