from __future__ import annotations

import cirq

from resource_estimation.compile_gateset.ppm_compiler import (
    CNOTMergeTag,
    ResourceStateTag,
    identify_cnot_merge_groups,
)


def _tagged_cnots(circuit: cirq.Circuit) -> list[tuple[cirq.Operation, CNOTMergeTag]]:
    tagged_cnots = []
    for operation in circuit.all_operations():
        tags = [tag for tag in operation.tags if isinstance(tag, CNOTMergeTag)]
        if tags:
            assert len(tags) == 1  # One merged group per CNOT
            tagged_cnots.append((operation, tags[0]))
    return tagged_cnots


def _assert_unitary_equivalent(actual: cirq.Circuit, expected: cirq.Circuit) -> None:
    qubit_order = sorted(actual.all_qubits() | expected.all_qubits())
    assert cirq.linalg.allclose_up_to_global_phase(
        actual.unitary(qubit_order=qubit_order),
        expected.unitary(qubit_order=qubit_order),
    )


def test_groups_adjacent_control_and_target_spiders() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)

    control_group = identify_cnot_merge_groups(cirq.Circuit(cirq.CNOT(q0, q1), cirq.CNOT(q0, q2)))
    assert [tag for _, tag in _tagged_cnots(control_group)] == [
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
    ]

    target_group = identify_cnot_merge_groups(cirq.Circuit(cirq.CNOT(q0, q2), cirq.CNOT(q1, q2)))
    assert [tag for _, tag in _tagged_cnots(target_group)] == [
        CNOTMergeTag(0, cirq.X),
        CNOTMergeTag(0, cirq.X),
    ]


def test_pushes_hadamard_through_opposite_spider() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    original = cirq.Circuit(cirq.CNOT(q0, q1), cirq.H(q0), cirq.CNOT(q2, q0))

    transformed = identify_cnot_merge_groups(original)

    tagged_cnots = _tagged_cnots(transformed)
    assert [tag for _, tag in tagged_cnots] == [
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
    ]
    assert [operation.untagged.qubits for operation, _ in tagged_cnots] == [
        (q0, q1),
        (q0, q2),
    ]
    _assert_unitary_equivalent(transformed, original)

    repeated = identify_cnot_merge_groups(transformed)
    assert repeated == transformed


def test_pushes_s_gate_through_z_spider() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    original = cirq.Circuit(cirq.CNOT(q0, q1), cirq.S(q0), cirq.CNOT(q0, q2))

    transformed = identify_cnot_merge_groups(original)

    assert [tag for _, tag in _tagged_cnots(transformed)] == [
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
    ]
    _assert_unitary_equivalent(transformed, original)


def test_supports_arbitrary_single_qubit_cliffords() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    original = cirq.Circuit(
        cirq.CNOT(q1, q0),
        (cirq.Y**0.5)(q0),
        cirq.CNOT(q0, q2),
    )

    transformed = identify_cnot_merge_groups(original)

    assert [tag for _, tag in _tagged_cnots(transformed)] == [
        CNOTMergeTag(0, cirq.X),
        CNOTMergeTag(0, cirq.X),
    ]
    assert any(
        isinstance(operation.gate, cirq.SingleQubitCliffordGate)
        for operation in transformed.all_operations()
    )
    _assert_unitary_equivalent(transformed, original)


def test_y_basis_and_tagged_operations_are_scan_barriers() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    y_barrier = identify_cnot_merge_groups(
        cirq.Circuit(cirq.CNOT(q0, q1), cirq.S(q1), cirq.CNOT(q2, q1))
    )
    assert len({tag.group_id for _, tag in _tagged_cnots(y_barrier)}) == 2

    resource_tag = ResourceStateTag(cirq.T, 0, 0, 1)
    annotation_barrier = identify_cnot_merge_groups(
        cirq.Circuit(
            cirq.CNOT(q0, q1),
            cirq.I(q0).with_tags(resource_tag),
            cirq.CNOT(q0, q2),
        )
    )
    assert len({tag.group_id for _, tag in _tagged_cnots(annotation_barrier)}) == 2


def test_surrounding_group_basis_takes_precedence_over_larger_candidate() -> None:
    p, q, r, s = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.CNOT(p, q),
        cirq.CNOT(q, r),
        cirq.CNOT(p, q),
        cirq.CNOT(s, r),
    )

    transformed = identify_cnot_merge_groups(circuit)

    tags_by_qubits = {
        operation.untagged.qubits: tag for operation, tag in _tagged_cnots(transformed)
    }
    surrounding_group = tags_by_qubits[(p, q)]
    assert surrounding_group == CNOTMergeTag(0, cirq.Z)
    assert tags_by_qubits[(q, r)].basis is cirq.Z
    assert tags_by_qubits[(q, r)].group_id != tags_by_qubits[(s, r)].group_id


def test_max_weight() -> None:
    p, q, r, s, t = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.CNOT(p, q), cirq.CNOT(p, r), cirq.CNOT(p, s), cirq.CNOT(p, t))

    transformed = identify_cnot_merge_groups(circuit)

    assert [tag for _, tag in _tagged_cnots(transformed)] == [
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
    ]

    transformed_capped = identify_cnot_merge_groups(circuit, max_weight=2)

    assert [tag for _, tag in _tagged_cnots(transformed_capped)] == [
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(0, cirq.Z),
        CNOTMergeTag(1, cirq.Z),
        CNOTMergeTag(1, cirq.Z),
    ]


def test_preserves_non_cnot_operations_and_handles_empty_circuit() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CZ(q0, q1), cirq.measure(q1))

    assert identify_cnot_merge_groups(circuit) == circuit
    assert identify_cnot_merge_groups(cirq.Circuit()) == cirq.Circuit()
