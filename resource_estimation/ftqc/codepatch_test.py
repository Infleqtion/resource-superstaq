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

import resource_estimation.ftqc.codepatch as cpa
import resource_estimation.ftqc.lattice_surgery_primitives as lsp


@pytest.mark.parametrize(
    "patch_id, coord, role, info",
    [
        (0, (0, 0), 0, {"data": 0}),
        (0, (1, 1), 1, {"x": 0}),
        (1, (0, 1), 2, {"z": 1}),
        (2, (3, 3), 0, {"data": 2}),

    ]
)
def test_physical_qubit_initializations(
    patch_id: int,
    coord: tuple[int, int],
    role: int,
    info: dict[str, int]
) -> None:
    qubit = cpa.PhysicalQubit(
        patch_id=patch_id,
        local_coord=coord,
        role=role,
        info=info
    )
    assert qubit.patch_id == patch_id
    assert qubit.local_coord == coord
    assert qubit.role == role
    assert qubit.info == info
    assert qubit.dimension == 2

def test_physical_qubit_others() -> None:
    q1 = cpa.PhysicalQubit(0, (0, 1), 0)
    q2 = cpa.PhysicalQubit(0, (0, 1), 1)
    q3 = cpa.PhysicalQubit(1, (0, 0), 0)
    assert q1 < q2 < q3
    assert str(q1) == "0:(0, 1, 0)"
    assert str(q2) == "0:(0, 1, 1)"
    assert str(q3) == "1:(0, 0, 0)"
    assert repr(q1) == "cpa.PhysicalQubit(0, (0, 1), 0)"
    assert repr(q2) == "cpa.PhysicalQubit(0, (0, 1), 1)"
    assert repr(q3) == "cpa.PhysicalQubit(1, (0, 0), 0)"


def test_logical_qubit() -> None:
    qubit = cpa.LogicalQubit(patch_id=3, logical_index=0, x_support={0, 2}, z_support={2, 3})

    assert isinstance(qubit, cirq.Qid)
    assert qubit.patch_id == 3
    assert qubit.logical_index == 0
    assert qubit.dimension == 2
    assert qubit.x_support == frozenset({0, 2})
    assert qubit.z_support == frozenset({2, 3})
    assert str(qubit) == "3:0"
    assert repr(qubit) == "cpa.LogicalQubit(3, 0)"

    circuit = cirq.Circuit(cirq.H(qubit))
    assert circuit.all_qubits() == frozenset({qubit})


def test_logical_qubit_identity() -> None:
    qubit = cpa.LogicalQubit(0, 0, x_support={0}, z_support={1})
    same_qubit = cpa.LogicalQubit(0, 0, x_support={0}, z_support={1})
    other_index = cpa.LogicalQubit(0, 1, x_support={0}, z_support={1})
    other_patch = cpa.LogicalQubit(1, 0, x_support={0}, z_support={1})

    assert qubit == same_qubit
    assert hash(qubit) == hash(same_qubit)
    assert qubit != other_index
    assert qubit != other_patch


def test_logical_qubit_rejects_missing_supports() -> None:
    with pytest.raises(TypeError):
        cpa.LogicalQubit()  # type: ignore[call-arg]


def test_logical_qubit_rejects_empty_supports() -> None:
    with pytest.raises(ValueError, match="supports must include"):
        cpa.LogicalQubit(patch_id=0, logical_index=0, x_support=set(), z_support=set())


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
        cpa.LogicalQubit(
            patch_id=patch_id,  # type: ignore[arg-type]
            logical_index=logical_index,  # type: ignore[arg-type]
            x_support={0},
            z_support={1},
        )


@pytest.mark.parametrize("support", [{-1}, {0, -1}])
def test_logical_qubit_rejects_negative_support_entries(support: set[int]) -> None:
    with pytest.raises(ValueError, match="entry must be nonnegative"):
        cpa.LogicalQubit(patch_id=0, logical_index=0, x_support=support, z_support={0})


@pytest.mark.parametrize("support", [{1.5}, {True}])
def test_logical_qubit_rejects_noninteger_support_entries(support: set[object]) -> None:
    with pytest.raises(TypeError, match="entry must be an integer"):
        cpa.LogicalQubit(patch_id=0, logical_index=0, x_support=support, z_support={0})  # type: ignore[arg-type]


def test_code_patch_classes_are_abstract() -> None:
    assert inspect.isabstract(cpa.CodePatch)
    assert inspect.isabstract(cpa.CSSCodePatch)
    assert not inspect.isabstract(cpa.RotatedSurfaceCodePatch)

    with pytest.raises(TypeError, match="abstract class CodePatch"):
        cpa.CodePatch(patch_id=3, n=9, k=1, d=3)  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract class CSSCodePatch"):
        cpa.CSSCodePatch(  # type: ignore[abstract]
            patch_id=3, qldpc_code=codes.SurfaceCode(3)
        )


def test_rotated_surface_code_patch_metadata() -> None:
    patch = cpa.RotatedSurfaceCodePatch(patch_id=7, d=7)

    assert isinstance(patch, cpa.CSSCodePatch)
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
    assert repr(patch) == ("cpa.RotatedSurfaceCodePatch(patch_id=7, d=7, n=49, k=1)")

    patch = cpa.RotatedSurfaceCodePatch(patch_id=5, d=5)

    assert patch.code_params == (25, 1, 5)
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.total_x_check_weight() == 40
    assert patch.total_z_check_weight() == 40
    assert len(patch.logical_qubits) == 1
    assert repr(patch) == ("cpa.RotatedSurfaceCodePatch(patch_id=5, d=5, n=25, k=1)")


def test_code_patch_surface_stabilizer_metadata_matches_qldpc() -> None:
    for d in [3, 5, 7]:
        patch = cpa.RotatedSurfaceCodePatch(patch_id=d, d=d)
        qldpc_code = codes.SurfaceCode(d)

        assert patch.num_x_stabilizers() == qldpc_code.num_checks_x
        assert patch.num_z_stabilizers() == qldpc_code.num_checks_z
        assert patch.total_x_check_weight() == len(qldpc_code.matrix_x.nonzero()[0])
        assert patch.total_z_check_weight() == len(qldpc_code.matrix_z.nonzero()[0])


def test_code_patch_logical_qubits_from_css_logical_support() -> None:
    patch = cpa.RotatedSurfaceCodePatch(patch_id=3, d=3)

    assert len(patch.logical_qubits) == 1
    assert patch.logical_qubits[0].patch_id == 3
    assert patch.logical_qubits[0].logical_index == 0
    assert patch.logical_qubits[0].x_support == frozenset({6, 7, 8})
    assert patch.logical_qubits[0].z_support == frozenset({0, 4, 8})


def test_rotated_surface_code_patches_have_distinct_logical_qubits() -> None:
    first_patch = cpa.RotatedSurfaceCodePatch(patch_id=0, d=3)
    second_patch = cpa.RotatedSurfaceCodePatch(patch_id=1, d=3)

    assert first_patch.logical_qubits[0] != second_patch.logical_qubits[0]


def test_code_patch_requires_distance() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'd'"):
        cpa.RotatedSurfaceCodePatch(patch_id=0)  # type: ignore[call-arg]


@pytest.mark.parametrize("d", (3, 5, 7))
def test_surface_code_patch_counts_match_legacy(d: int) -> None:
    patch = cpa.RotatedSurfaceCodePatch(patch_id=d, d=d)
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

@pytest.mark.parametrize("d", (3, 5))
def test_rotated_codepatch_footprint(d: int) -> None:
    patch = cpa.RotatedSurfaceCodePatch(patch_id=9, d=d)
    qubits = patch.default_layout()

    assert len(qubits) == patch.num_physical_qubits == 2 * d**2 - 1
    assert {q.patch_id for q in qubits} == {9}
    assert len({q.local_coord for q in qubits}) == len(qubits)
    assert sum(q.role == 0 for q in qubits) == d**2
    assert sum(q.role == 1 for q in qubits) == (d**2 - 1) // 2
    assert sum(q.role == 2 for q in qubits) == (d**2 - 1) // 2
    assert (patch.footprint.rows, patch.footprint.cols) == (2 * d - 1, 2 * d - 1)

    if d == 3:
        by_coord = {q.local_coord: q.role for q in qubits}
        # Data, interior X/Z checks, and both kinds of boundary check.
        assert by_coord[(0, 2)] == 0
        assert by_coord[(1, 2)] == 1
        assert by_coord[(2, 1)] == 2
        assert by_coord[(0, 1)] == 2  # Top Z boundary
        assert by_coord[(3, 0)] == 1  # Right X boundary
        assert by_coord[(4, 3)] == 2  # Bottom Z boundary
        assert by_coord[(1, 4)] == 1  # Left X boundary


@pytest.mark.parametrize("d", (2, 4))
def test_rotated_codepatch_rejects_even_distance(d: int) -> None:
    with pytest.raises(ValueError, match="CodePatches must be odd distance"):
        cpa.RotatedSurfaceCodePatch(patch_id=0, d=d)


def test_bb_codepatch_metadata() -> None:
    patch = cpa.BBCodePatch(patch_id=8, l=2, m=3)
    assert isinstance(patch, cpa.CSSCodePatch)
    assert patch.patch_id == 8
    assert patch.n == patch.num_data_qubits == 12
    assert patch.num_measure_qubits == 12
    assert patch.num_physical_qubits == 24
    assert patch.num_logical_qubits == patch.k == len(patch.logical_qubits)
    assert all(q.patch_id == 8 for q in patch.logical_qubits)
    assert repr(patch) == "cpa.BBCodePatch(patch_id=8, l=2, m=3)"


def test_bb_codepatch_default_layout() -> None:
    patch = cpa.BBCodePatch(patch_id=8, l=2, m=3)
    qubits = patch.default_layout()
    assert len(qubits) == patch.num_physical_qubits == 24
    assert {q.patch_id for q in qubits} == {8}
    assert len({q.local_coord for q in qubits}) == len(qubits)
    assert sum(q.role == 0 for q in qubits) == 12
    assert sum(q.role == 1 for q in qubits) == 6
    assert sum(q.role == 2 for q in qubits) == 6
    assert (patch.footprint.rows, patch.footprint.cols) == (6, 4)
