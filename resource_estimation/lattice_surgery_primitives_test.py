from math import pi

import cirq
import pytest
import resource_estimation.lattice_surgery_primitives as lsp
from numpy.testing import assert_array_equal


def test_merge():
    merge_gate = lsp.Merge(2, smooth=True)
    assert merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"

    merge_gate = lsp.Merge(2, smooth=False)
    assert not merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"


def test_split():
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


def test_syndrome_extract():
    for i in [1, 2, 3, 4]:
        extraction_gate = lsp.SyndromeExtract(i, i * 2)
        assert extraction_gate.num_qubits() == i
        assert extraction_gate.rounds == i * 2
        assert str(extraction_gate) == f"SE({i * 2})"


def test_error_correct():
    error_correction_gate = lsp.ErrorCorrect(1)
    assert str(error_correction_gate) == "ERROR CORRECT"


def test_cultivate():
    theta = pi / 2
    cultivation_gate = lsp.Cultivate(theta=theta)
    assert cultivation_gate.theta == theta
    assert str(cultivation_gate) == "CULT(1.571)"


def test_move():
    a, b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    alley_move = lsp.Move(None).on(a, b)
    assert str(alley_move) == "MOVE(q(0, 0), q(0, 1))"
    interact_move = lsp.Move("interact").on(a)
    assert str(interact_move) == "MOVE_IZ(q(0, 0))"
    measure_move = lsp.Move("measure").on(b)
    assert str(measure_move) == "MOVE_MZ(q(0, 1))"


def test_rotated_code_patch():
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


def test_buffer():
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


def test_intermediate_patch():
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


def test_endpoint_patch():
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


def test_serialization():
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
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
        ]
    )
    print(circuit)
    json_str = cirq.to_json(circuit)
    # print(json_str)
    new_circuit = cirq.read_json(
        json_text=json_str, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )
    print(new_circuit)
    print(new_circuit == circuit)
    cirq.testing.assert_json_roundtrip_works(
        circuit, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )


def test_repr():
    qa, qb = cirq.LineQubit.range(2)
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

    move = lsp.Move(zone="interact").on_each(qa, qb)
    assert (
        repr(move)
        == "[lsp.Move(zone=interact).on(cirq.LineQubit(0)), lsp.Move(zone=interact).on(cirq.LineQubit(1))]"
    )

    move = lsp.Move(zone=None).on(qa, qb)
    assert repr(move) == "lsp.Move(zone=None).on(cirq.LineQubit(0), cirq.LineQubit(1))"

    move = lsp.Move(zone="measure").on(qa)
    assert repr(move) == "lsp.Move(zone=measure).on(cirq.LineQubit(0))"


def test_patch_eq_and_hash():
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
