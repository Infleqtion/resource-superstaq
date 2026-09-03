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
import inspect

import cirq
import pytest
from qldpc import codes

import resource_estimation.ftqc.codepatch as codepatch
import resource_estimation.ftqc.lattice_surgery_primitives as lsp


def test_logical_qubit() -> None:
    qubit = codepatch.LogicalQubit(patch_id=3, logical_index=0, x_support={0, 2}, z_support={2, 3})

    assert isinstance(qubit, cirq.Qid)
    assert qubit.patch_id == 3
    assert qubit.logical_index == 0
    assert qubit.dimension == 2
    assert qubit.x_support == frozenset({0, 2})
    assert qubit.z_support == frozenset({2, 3})
    assert str(qubit) == "3:0"
    assert repr(qubit) == "codepatch.LogicalQubit(3, 0)"

    circuit = cirq.Circuit(cirq.H(qubit))
    assert circuit.all_qubits() == frozenset({qubit})


def test_logical_qubit_identity() -> None:
    qubit = codepatch.LogicalQubit(0, 0, x_support={0}, z_support={1})
    same_qubit = codepatch.LogicalQubit(0, 0, x_support={0}, z_support={1})
    other_index = codepatch.LogicalQubit(0, 1, x_support={0}, z_support={1})
    other_patch = codepatch.LogicalQubit(1, 0, x_support={0}, z_support={1})

    assert qubit == same_qubit
    assert hash(qubit) == hash(same_qubit)
    assert qubit != other_index
    assert qubit != other_patch


def test_logical_qubit_rejects_missing_supports() -> None:
    with pytest.raises(TypeError):
        codepatch.LogicalQubit()  # type: ignore[call-arg]


def test_logical_qubit_rejects_empty_supports() -> None:
    with pytest.raises(ValueError, match="supports must include"):
        codepatch.LogicalQubit(patch_id=0, logical_index=0, x_support=set(), z_support=set())


@pytest.mark.parametrize(
    ("patch_id", "logical_index", "error_type", "message"),
    [
        (-1, 0, ValueError, "patch_id must be nonnegative"),
        (0, -1, ValueError, "logical_index must be nonnegative"),
        ("patch", 0, TypeError, "patch_id must be an integer"),
        (0, True, TypeError, "logical_index must be an integer"),
    ],
)
def test_logical_qubit_rejects_invalid_ids(
    patch_id: object,
    logical_index: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        codepatch.LogicalQubit(
            patch_id=patch_id,  # type: ignore[arg-type]
            logical_index=logical_index,  # type: ignore[arg-type]
            x_support={0},
            z_support={1},
        )


@pytest.mark.parametrize("support", [{-1}, {0, -1}])
def test_logical_qubit_rejects_negative_support_entries(support: set[int]) -> None:
    with pytest.raises(ValueError, match="entry must be nonnegative"):
        codepatch.LogicalQubit(patch_id=0, logical_index=0, x_support=support, z_support={0})


@pytest.mark.parametrize("support", [{1.5}, {True}])
def test_logical_qubit_rejects_noninteger_support_entries(support: set[object]) -> None:
    with pytest.raises(TypeError, match="entry must be an integer"):
        codepatch.LogicalQubit(patch_id=0, logical_index=0, x_support=support, z_support={0})  # type: ignore[arg-type]


def test_code_patch_classes_are_abstract() -> None:
    assert inspect.isabstract(codepatch.CodePatch)
    assert inspect.isabstract(codepatch.CSSCodePatch)
    assert not inspect.isabstract(codepatch.RotatedSurfaceCodePatch)

    with pytest.raises(TypeError, match="abstract class CodePatch"):
        codepatch.CodePatch(patch_id=3, n=9, k=1, d=3)  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract class CSSCodePatch"):
        codepatch.CSSCodePatch(  # type: ignore[abstract]
            patch_id=3, qldpc_code=codes.SurfaceCode(3)
        )


def test_rotated_surface_code_patch_metadata() -> None:
    patch = codepatch.RotatedSurfaceCodePatch(patch_id=7, d=7)

    assert isinstance(patch, codepatch.CSSCodePatch)
    assert patch.patch_id == 7
    assert patch.code_params == (49, 1, 7)
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.num_logical_qubits == 1
    assert patch.num_physical_qubits == 97
    assert patch.num_x_stabilizers() == 24
    assert patch.num_z_stabilizers() == 24
    assert patch.total_x_check_weight() == 84
    assert patch.total_z_check_weight() == 84
    assert len(patch.logical_qubits) == 1
    assert repr(patch) == ("codepatch.RotatedSurfaceCodePatch(patch_id=7, d=7, n=49, k=1)")

    patch = codepatch.RotatedSurfaceCodePatch(patch_id=5, d=5)

    assert patch.code_params == (25, 1, 5)
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.total_x_check_weight() == 40
    assert patch.total_z_check_weight() == 40
    assert len(patch.logical_qubits) == 1
    assert repr(patch) == ("codepatch.RotatedSurfaceCodePatch(patch_id=5, d=5, n=25, k=1)")


def test_code_patch_surface_stabilizer_metadata_matches_qldpc() -> None:
    for d in [3, 5, 7]:
        patch = codepatch.RotatedSurfaceCodePatch(patch_id=d, d=d)
        qldpc_code = codes.SurfaceCode(d)

        assert patch.num_x_stabilizers() == qldpc_code.num_checks_x
        assert patch.num_z_stabilizers() == qldpc_code.num_checks_z
        assert patch.total_x_check_weight() == len(qldpc_code.matrix_x.nonzero()[0])
        assert patch.total_z_check_weight() == len(qldpc_code.matrix_z.nonzero()[0])


def test_code_patch_logical_qubits_from_css_logical_support() -> None:
    patch = codepatch.RotatedSurfaceCodePatch(patch_id=3, d=3)

    assert len(patch.logical_qubits) == 1
    assert patch.logical_qubits[0].patch_id == 3
    assert patch.logical_qubits[0].logical_index == 0
    assert patch.logical_qubits[0].x_support == frozenset({6, 7, 8})
    assert patch.logical_qubits[0].z_support == frozenset({0, 4, 8})


def test_rotated_surface_code_patches_have_distinct_logical_qubits() -> None:
    first_patch = codepatch.RotatedSurfaceCodePatch(patch_id=0, d=3)
    second_patch = codepatch.RotatedSurfaceCodePatch(patch_id=1, d=3)

    assert first_patch.logical_qubits[0] != second_patch.logical_qubits[0]


def test_code_patch_requires_distance() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'd'"):
        codepatch.RotatedSurfaceCodePatch(patch_id=0)  # type: ignore[call-arg]


@pytest.mark.parametrize("d", (3, 5, 7))
def test_surface_code_patch_counts_match_legacy(d: int) -> None:
    patch = codepatch.RotatedSurfaceCodePatch(patch_id=d, d=d)
    legacy_patch = lsp.RotatedCodePatch(d)

    assert patch.d == legacy_patch.d
    assert patch.num_data_qubits == legacy_patch.num_data_qubits
    assert patch.num_measure_qubits == legacy_patch.num_measure_qubits
    assert patch.num_physical_qubits == legacy_patch.num_physical_qubits
    assert patch.num_x_stabilizers() == (
        legacy_patch.num_x_stabs(full=True) + legacy_patch.num_x_stabs(full=False)
    )
    assert patch.num_z_stabilizers() == (
        legacy_patch.num_z_stabs(full=True) + legacy_patch.num_z_stabs(full=False)
    )
    assert patch.total_x_check_weight() == legacy_patch.total_x_syndrome_cnots()
    assert patch.total_z_check_weight() == legacy_patch.total_z_syndrome_cnots()
