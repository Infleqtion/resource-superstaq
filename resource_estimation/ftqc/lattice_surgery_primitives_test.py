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
from math import pi

import cirq
import pytest
from numpy.testing import assert_array_equal

import resource_estimation.ftqc.lattice_surgery_primitives as lsp


def test_merge() -> None:
    merge_gate = lsp.Merge(2, smooth=True)
    assert merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"

    merge_gate = lsp.Merge(2, smooth=False)
    assert not merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"


def test_split() -> None:
    partitions = [1, 2, 3, 4]
    split_gate = lsp.Split(partitions=partitions, smooth=True)
    assert split_gate.smooth
    assert split_gate.num_qubits() == 10
    assert str(split_gate) == "SPLIT"
    assert split_gate.partitions == partitions

    split_gate = lsp.Split(partitions=partitions, smooth=False)
    assert not split_gate.smooth
    assert split_gate.num_qubits() == 10
    assert str(split_gate) == "SPLIT"
    assert split_gate.partitions == partitions


def test_syndrome_extract() -> None:
    for i in [1, 2, 3, 4]:
        extraction_gate = lsp.SyndromeExtract(i, i * 2)
        assert extraction_gate.num_qubits() == i
        assert extraction_gate.rounds == i * 2
        assert str(extraction_gate) == f"SE({i * 2})"


def test_error_correct() -> None:
    error_correction_gate = lsp.ErrorCorrect(1)
    assert str(error_correction_gate) == "ERROR CORRECT"


def test_cultivate() -> None:
    theta = pi / 2
    cultivation_gate = lsp.Cultivate(theta=theta)
    assert cultivation_gate.theta == theta
    assert str(cultivation_gate) == "CULT(1.571)"


def test_magic_state_code_teleport() -> None:
    gate = lsp.MagicStateCodeTeleport()

    assert gate.num_qubits() == 1
    assert str(gate) == "T-CODE-TELEPORT"
    assert repr(gate) == "lsp.MagicStateCodeTeleport()"
    assert cirq.circuit_diagram_info(gate).wire_symbols == ("T-CODE-XFER",)
    assert gate._json_dict_() == {}
    assert gate._json_namespace_() == "lsp"
    assert lsp.custom_resolver("lsp.MagicStateCodeTeleport") is lsp.MagicStateCodeTeleport


def test_distil_rejects_invalid_resource() -> None:
    with pytest.raises(ValueError, match="Invalid resource"):
        lsp.Distil("S")  # type: ignore[arg-type]


def test_move() -> None:
    a, b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    alley_move = lsp.Move(None).on(a, b)
    assert str(alley_move) == "MOVE(q(0, 0), q(0, 1))"
    interact_move = lsp.Move("interact").on(a)
    assert str(interact_move) == "MOVE_IZ(q(0, 0))"
    measure_move = lsp.Move("measure").on(b)
    assert str(measure_move) == "MOVE_MZ(q(0, 1))"


def test_distil() -> None:
    gate = lsp.Distil("T")
    assert str(gate) == "DISTIL(T)"
    gate = lsp.Distil("CCZ")
    assert str(gate) == "DISTIL(CCZ)"
    with pytest.raises(ValueError, match="Invalid resource"):
        _ = lsp.Distil("Toffoli")


def test_logical_qubit() -> None:
    qubit = lsp.LogicalQubit(x_support={0, 2}, z_support={2, 3})

    assert qubit.label == "zero"
    assert qubit.x_support == frozenset({0, 2})
    assert qubit.z_support == frozenset({2, 3})
    assert qubit.num_qubits == 3


@pytest.mark.parametrize("label", ["zero", "one", "plus", "minus", "data"])
def test_logical_qubit_labels(label: lsp.LogicalQubitLabel) -> None:
    qubit = lsp.LogicalQubit(x_support={0}, z_support={0}, label=label)

    assert qubit.label == label


def test_logical_qubit_rejects_missing_supports() -> None:
    with pytest.raises(TypeError):
        lsp.LogicalQubit()  # type: ignore[call-arg]


def test_logical_qubit_rejects_empty_supports() -> None:
    with pytest.raises(ValueError, match="supports must include"):
        lsp.LogicalQubit(x_support=set(), z_support=set())


def test_logical_qubit_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="Logical qubit label must be one of"):
        lsp.LogicalQubit(x_support={0}, z_support={0}, label="bad")  # type: ignore[arg-type]


@pytest.mark.parametrize("support", [{-1}, {0, -1}])
def test_logical_qubit_rejects_negative_support_entries(support: set[int]) -> None:
    with pytest.raises(ValueError, match="entries must be nonnegative"):
        lsp.LogicalQubit(x_support=support, z_support={0})


@pytest.mark.parametrize("support", [{1.5}, {True}])
def test_logical_qubit_rejects_noninteger_support_entries(support: set[object]) -> None:
    with pytest.raises(TypeError, match="entries must be integers"):
        lsp.LogicalQubit(x_support=support, z_support={0})  # type: ignore[arg-type]


def test_code_patch_surface_metadata() -> None:
    pytest.importorskip("qldpc")

    patch = lsp.CodePatch("surface", d=7)

    assert patch.code_type == "surface"
    assert patch.code_params == (49, 1, 7)
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.num_physical_qubits == 97
    assert patch.num_x_stabs() == 24
    assert patch.num_z_stabs() == 24
    assert patch.total_x_syndrome_cnots() == 84
    assert patch.total_z_syndrome_cnots() == 84
    assert patch.patch_label == "compute"
    assert patch.is_qldpc_backed
    assert patch.is_binary
    assert patch.is_css
    assert patch.is_stabilizer_code
    assert patch.is_surface_code
    assert len(patch.logical_qubits) == 1
    assert [qubit.label for qubit in patch.logical_qubits] == ["zero"]
    assert (
        repr(patch) == "lsp.CodePatch(code_type='surface', d=7, n=49, k=1, patch_label='compute')"
    )

    patch = lsp.CodePatch("surface", d=5, patch_label="cultivate")

    assert patch.code_params == (25, 1, 5)
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.total_x_syndrome_cnots() == 40
    assert patch.total_z_syndrome_cnots() == 40
    assert patch.patch_label == "cultivate"
    assert patch.is_qldpc_backed
    assert len(patch.logical_qubits) == 1
    assert (
        repr(patch) == "lsp.CodePatch(code_type='surface', d=5, n=25, k=1, patch_label='cultivate')"
    )


def test_code_patch_surface_stabilizer_metadata_matches_qldpc() -> None:
    pytest.importorskip("qldpc")
    from qldpc import codes

    for d in [3, 5, 7]:
        patch = lsp.CodePatch("surface", d=d)
        qldpc_code = codes.SurfaceCode(d)

        assert patch.num_x_stabs() == qldpc_code.num_checks_x
        assert patch.num_z_stabs() == qldpc_code.num_checks_z
        assert patch.total_x_syndrome_cnots() == len(qldpc_code.matrix_x.nonzero()[0])
        assert patch.total_z_syndrome_cnots() == len(qldpc_code.matrix_z.nonzero()[0])


def test_code_patch_logical_qubits_from_css_logical_support() -> None:
    pytest.importorskip("qldpc")

    patch = lsp.CodePatch("surface", d=3)

    assert len(patch.logical_qubits) == 1
    assert patch.logical_qubits[0].label == "zero"
    assert patch.logical_qubits[0].num_qubits == 5
    assert patch.logical_qubits[0].x_support == frozenset({6, 7, 8})
    assert patch.logical_qubits[0].z_support == frozenset({0, 4, 8})

    patch = lsp.CodePatch("toric", d=2)

    assert len(patch.logical_qubits) == 2
    assert [qubit.label for qubit in patch.logical_qubits] == ["zero"] * 2
    assert [qubit.num_qubits for qubit in patch.logical_qubits] == [3, 3]
    assert [qubit.x_support for qubit in patch.logical_qubits] == [
        frozenset({1, 2}),
        frozenset({1, 3}),
    ]
    assert [qubit.z_support for qubit in patch.logical_qubits] == [
        frozenset({0, 2}),
        frozenset({0, 3}),
    ]


def test_code_patch_rejects_mismatched_distance() -> None:
    pytest.importorskip("qldpc")

    with pytest.raises(ValueError, match="does not match qLDPC code distance"):
        lsp.CodePatch("surface", 3, d=5)


def test_code_patch_requires_qldpc_code_type() -> None:
    pytest.importorskip("qldpc")

    with pytest.raises(ValueError, match="qLDPC code family not found"):
        lsp.CodePatch("color", d=3)

    with pytest.raises(ValueError, match="Patch label must be one of"):
        lsp.CodePatch("surface", d=3, patch_label="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        lsp.CodePatch()  # type: ignore[call-arg]


def test_code_patch_non_css_stabilizer_metadata_errors() -> None:
    qldpc = pytest.importorskip("qldpc")
    from qldpc import codes

    patch = lsp.CodePatch(codes.FiveQubitCode, patch_label="distil")

    assert qldpc.__version__
    assert patch.code_type == "FiveQubitCode"
    assert patch.code_params == (5, 1, 3)
    assert patch.num_data_qubits == 5
    assert patch.num_measure_qubits == 4
    assert patch.is_binary
    assert not patch.is_css
    assert patch.is_stabilizer_code
    assert not patch.is_surface_code
    assert patch.is_qldpc_backed
    assert patch.patch_label == "distil"
    assert len(patch.logical_qubits) == 1
    assert patch.logical_qubits[0].label == "zero"
    assert patch.logical_qubits[0].num_qubits == 5
    assert patch.logical_qubits[0].x_support == frozenset({1, 2, 4})
    assert patch.logical_qubits[0].z_support == frozenset({0, 1, 2, 3, 4})
    with pytest.raises(ValueError, match="X/Z stabilizer counts"):
        patch.num_x_stabs()
    with pytest.raises(ValueError, match="X/Z stabilizer counts"):
        patch.num_z_stabs()
    with pytest.raises(ValueError, match="X/Z stabilizer counts"):
        patch.total_x_syndrome_cnots()
    with pytest.raises(ValueError, match="X/Z stabilizer counts"):
        patch.total_z_syndrome_cnots()


def test_code_patch_qldpc_css_stabilizer_metadata() -> None:
    pytest.importorskip("qldpc")
    from qldpc import codes

    simplex = codes.SimplexCode(3)
    patch = lsp.CodePatch("hgp", simplex, simplex, d=4)

    assert patch.num_x_stabs() == 49
    assert patch.num_z_stabs() == 49
    assert patch.num_physical_qubits == 196
    assert len(patch.logical_qubits) == 18
    assert [qubit.label for qubit in patch.logical_qubits] == ["zero"] * 18
    assert [qubit.num_qubits for qubit in patch.logical_qubits] == [7] * 18


def test_code_patch_rejects_unequal_logical_qubit_supports() -> None:
    pytest.importorskip("qldpc")
    from qldpc.objects import Pauli

    class UnequalLogicalSupportCode:
        num_checks = 0

        def get_code_params(self) -> tuple[int, int, int]:
            return 4, 2, 2

        def get_logical_ops(self, pauli: object) -> list[list[int]]:
            if pauli == Pauli.X:
                return [[1, 0, 0, 0], [0, 1, 0, 0]]
            if pauli == Pauli.Z:
                return [[0, 1, 0, 0], [0, 0, 1, 1]]
            raise ValueError(f"Unexpected pauli: {pauli!r}")

    with pytest.raises(ValueError, match="must have the same size"):
        lsp.CodePatch(UnequalLogicalSupportCode)


def test_code_patch_qldpc_family_alias() -> None:
    pytest.importorskip("qldpc")

    patch = lsp.CodePatch("toric", d=2)

    assert patch.code_type == "toric"
    assert patch.code_params == (4, 2, 2)
    assert patch.patch_label == "compute"


def test_qldpc_family_resolution_fallbacks() -> None:
    class ExactNameCodes:
        CustomCode = object()

    class ExportedCodes:
        __all__ = ("Mixed_Case_Code",)

    assert lsp._resolve_qldpc_family_name("CustomCode", ExactNameCodes) == "CustomCode"
    assert lsp._resolve_qldpc_family_name("mixed case code", ExportedCodes) == "Mixed_Case_Code"


def test_code_patch_qldpc_compatibility_fallbacks() -> None:
    class LegacyCode:
        dimension = 1

        def __len__(self) -> int:
            return 5

        def get_distance_if_known(self) -> int:
            return 3

    class CheckMatrix:
        shape = (4, 9)

    class MatrixOnlyCode:
        matrix_x = CheckMatrix()

    assert lsp.CodePatch._metadata_from_qldpc_code(LegacyCode()) == (5, 1, 3)

    patch = object.__new__(lsp.CodePatch)
    patch._qldpc_code = MatrixOnlyCode()
    assert patch._qldpc_css_check_count("x") == 4

    patch.k = 2
    with pytest.raises(ValueError, match="returned 1 logical X operators"):
        patch._validate_logical_ops_count([[1, 0]], "X")

    patch.n = 3
    with pytest.raises(ValueError, match="must have length n=3 or 2n=6"):
        patch._logical_op_support([1, 0])


def test_code_patch_callable_factory() -> None:
    pytest.importorskip("qldpc")
    from qldpc import codes

    def custom_five_qubit_code() -> object:
        return codes.FiveQubitCode()

    patch = lsp.CodePatch(custom_five_qubit_code, patch_label="memory")

    assert patch.code_type == "custom_five_qubit_code"
    assert patch.code_params == (5, 1, 3)
    assert patch.patch_label == "memory"


def test_rotated_code_patch() -> None:
    with pytest.raises(AssertionError, match="CodePatches must be odd distance"):
        lsp.RotatedCodePatch(4)

    d = 3
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 3
    assert patch.rows == patch.cols == 5
    assert patch.num_physical_qubits == 17
    assert patch.num_data_qubits == 9
    assert patch.num_measure_qubits == 8
    assert patch.num_z_stabs(full=True) == 2
    assert patch.num_z_stabs(full=False) == 2
    assert patch.num_x_stabs(full=True) == 2
    assert patch.num_x_stabs(full=False) == 2
    assert patch.total_z_syndrome_cnots() == 12
    assert patch.total_x_syndrome_cnots() == 12

    d = 5
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 5
    assert patch.rows == patch.cols == 9
    assert patch.num_physical_qubits == 49
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.num_z_stabs(full=True) == 8
    assert patch.num_z_stabs(full=False) == 4
    assert patch.num_x_stabs(full=True) == 8
    assert patch.num_x_stabs(full=False) == 4
    assert patch.total_z_syndrome_cnots() == 40
    assert patch.total_x_syndrome_cnots() == 40

    d = 7
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 7
    assert patch.rows == patch.cols == 13
    assert patch.num_physical_qubits == 97
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.num_z_stabs(full=True) == 18
    assert patch.num_z_stabs(full=False) == 6
    assert patch.num_x_stabs(full=True) == 18
    assert patch.num_x_stabs(full=False) == 6
    assert patch.total_z_syndrome_cnots() == 84
    assert patch.total_x_syndrome_cnots() == 84


def test_buffer() -> None:
    d = 7
    smooth_buff = lsp.BufferCodePatch(d=d, smooth=True)
    rough_buff = lsp.BufferCodePatch(d=d, smooth=False)

    assert_array_equal(
        [
            smooth_buff.num_z_stabs(full=True),
            smooth_buff.num_x_stabs(full=True),
            rough_buff.num_z_stabs(full=True),
            rough_buff.num_x_stabs(full=True),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_buff.num_x_stabs(full=False),
            rough_buff.num_z_stabs(full=False),
        ],
        2,
    )
    assert_array_equal(
        [
            smooth_buff.num_z_stabs(full=False),
            rough_buff.num_x_stabs(full=False),
        ],
        0,
    )


def test_intermediate_patch() -> None:
    d = 7
    smooth_inter = lsp.IntermediatePatch(d=d, smooth=True)
    rough_inter = lsp.IntermediatePatch(d=d, smooth=False)
    assert_array_equal(
        [
            smooth_inter.num_z_stabs(full=True),
            smooth_inter.num_x_stabs(full=True),
            rough_inter.num_z_stabs(full=True),
            rough_inter.num_x_stabs(full=True),
        ],
        18,
    )
    assert_array_equal(
        [
            smooth_inter.num_x_stabs(full=False),
            rough_inter.num_z_stabs(full=False),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_inter.num_z_stabs(full=False),
            rough_inter.num_x_stabs(full=False),
        ],
        0,
    )


def test_endpoint_patch() -> None:
    d = 7
    smooth_end = lsp.EndpointPatch(d=d, smooth=True)
    rough_end = lsp.EndpointPatch(d=d, smooth=False)
    assert_array_equal(
        [
            smooth_end.num_z_stabs(full=True),
            smooth_end.num_x_stabs(full=True),
            rough_end.num_z_stabs(full=True),
            rough_end.num_x_stabs(full=True),
        ],
        18,
    )
    assert_array_equal(
        [
            smooth_end.num_x_stabs(full=False),
            rough_end.num_z_stabs(full=False),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_end.num_z_stabs(full=False),
            rough_end.num_x_stabs(full=False),
        ],
        3,
    )


def test_serialization() -> None:
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    factory_block = cirq.LineQubit.range(31)
    circuit = cirq.Circuit(
        [
            lsp.Merge(2, True).on(qubit_a, qubit_b),
            lsp.Split([1, 1], True).on(qubit_a, qubit_b),
            lsp.SyndromeExtract(1, 1).on(qubit_a),
            lsp.ErrorCorrect(1).on(qubit_b),
            lsp.Cultivate(1.0).on(qubit_a),
            lsp.Move(zone="interact").on_each(qubit_a, qubit_b),
            lsp.Move(zone=None).on(qubit_a, qubit_b),
            lsp.Move(zone="measure").on(qubit_a),
            lsp.Distil("T").on(*factory_block),
            lsp.Distil("CCZ").on(*factory_block[:23]),
        ]
    )
    json_str = cirq.to_json(circuit)
    new_circuit = cirq.read_json(
        json_text=json_str, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )
    cirq.testing.assert_json_roundtrip_works(
        circuit, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )

    circuit = cirq.Circuit(
        [
            lsp.Merge(2, True).on(qubit_a, qubit_b),
            lsp.Split([1, 1], True).on(qubit_a, qubit_b),
            lsp.SyndromeExtract(1, 1).on(qubit_a),
            lsp.ErrorCorrect(1).on(qubit_b),
            lsp.Distil("T").on(*factory_block),
            lsp.Distil("CCZ").on(*factory_block[:23]),
            lsp.Move(zone="interact").on_each(qubit_a, qubit_b),
            lsp.Move(zone=None).on(qubit_a, qubit_b),
            lsp.Move(zone="measure").on(qubit_a),
        ],
    )
    json_str = cirq.to_json(circuit)
    new_circuit = cirq.read_json(
        json_text=json_str,
        resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS],
    )
    assert new_circuit == circuit
    cirq.testing.assert_json_roundtrip_works(
        circuit,
        resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS],
    )


def test_repr() -> None:
    qa, qb = cirq.LineQubit.range(2)
    factory_block = cirq.LineQubit.range(31)
    merge = lsp.Merge(2, smooth=False).on(qa, qb)
    assert (
        repr(merge)
        == "lsp.Merge(num_qubits=2, smooth=False).on(cirq.LineQubit(0), cirq.LineQubit(1))"
    )

    split = lsp.Split([1, 1], smooth=False).on(qa, qb)
    assert (
        repr(split)
        == "lsp.Split(partitions=[1, 1], smooth=False).on(cirq.LineQubit(0), cirq.LineQubit(1))"
    )

    se = lsp.SyndromeExtract(1, 5).on(qa)
    assert repr(se) == "lsp.SyndromeExtract(num_qubits=1, rounds=5).on(cirq.LineQubit(0))"

    ec = lsp.ErrorCorrect(1).on(qa)
    assert repr(ec) == "lsp.ErrorCorrect(num_qubits=1).on(cirq.LineQubit(0))"

    cult = lsp.Cultivate(7).on(qa)
    assert repr(cult) == "lsp.Cultivate(theta=7).on(cirq.LineQubit(0))"

    dist_t = lsp.Distil("T").on(*factory_block)
    assert (
        repr(dist_t)
        == "lsp.Distil(T)(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3), cirq.LineQubit(4), cirq.LineQubit(5), cirq.LineQubit(6), cirq.LineQubit(7), cirq.LineQubit(8), cirq.LineQubit(9), cirq.LineQubit(10), cirq.LineQubit(11), cirq.LineQubit(12), cirq.LineQubit(13), cirq.LineQubit(14), cirq.LineQubit(15), cirq.LineQubit(16), cirq.LineQubit(17), cirq.LineQubit(18), cirq.LineQubit(19), cirq.LineQubit(20), cirq.LineQubit(21), cirq.LineQubit(22), cirq.LineQubit(23), cirq.LineQubit(24), cirq.LineQubit(25), cirq.LineQubit(26), cirq.LineQubit(27), cirq.LineQubit(28), cirq.LineQubit(29), cirq.LineQubit(30))"
    )

    dist_ccz = lsp.Distil("CCZ").on(*factory_block[:23])
    assert (
        repr(dist_ccz)
        == "lsp.Distil(CCZ)(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3), cirq.LineQubit(4), cirq.LineQubit(5), cirq.LineQubit(6), cirq.LineQubit(7), cirq.LineQubit(8), cirq.LineQubit(9), cirq.LineQubit(10), cirq.LineQubit(11), cirq.LineQubit(12), cirq.LineQubit(13), cirq.LineQubit(14), cirq.LineQubit(15), cirq.LineQubit(16), cirq.LineQubit(17), cirq.LineQubit(18), cirq.LineQubit(19), cirq.LineQubit(20), cirq.LineQubit(21), cirq.LineQubit(22))"
    )
    move = lsp.Move(zone="interact").on_each(qa, qb)
    assert (
        repr(move)
        == "[lsp.Move(zone=interact).on(cirq.LineQubit(0)), lsp.Move(zone=interact).on(cirq.LineQubit(1))]"
    )

    move = lsp.Move(zone=None).on(qa, qb)
    assert repr(move) == "lsp.Move(zone=None).on(cirq.LineQubit(0), cirq.LineQubit(1))"

    move = lsp.Move(zone="measure").on(qa)
    assert repr(move) == "lsp.Move(zone=measure).on(cirq.LineQubit(0))"


def test_patch_eq_and_hash() -> None:
    patch1 = lsp.RotatedCodePatch(3)
    patch2 = lsp.RotatedCodePatch(5)
    assert patch1 != patch2
    assert hash(patch1) == 3
    assert hash(patch2) == 5

    patch3 = lsp.BufferCodePatch(3, smooth=True)
    patch4 = lsp.BufferCodePatch(3, smooth=False)
    assert patch3 != patch4

    patch5 = lsp.IntermediatePatch(3, smooth=True)
    patch6 = lsp.IntermediatePatch(5, smooth=False)

    patch7 = lsp.EndpointPatch(3, smooth=True)
    patch8 = lsp.EndpointPatch(5, smooth=False)

    assert patch3 != patch4
    assert patch5 != patch6
    assert patch7 != patch8

    assert patch1 != patch3
