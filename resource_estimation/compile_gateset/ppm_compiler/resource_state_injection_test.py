from __future__ import annotations

import cirq
import sympy

from resource_estimation.compile_gateset.ppm_compiler import (
    ResourceStateBasis,
    ResourceStateTag,
    replace_resource_gates,
)


def test_replaces_t_and_ccz_with_resource_ancillas() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.T(q0), cirq.CCZ(q0, q1, q2))

    transformed = replace_resource_gates(circuit)

    t_ancilla = cirq.NamedQubit("T_0")
    ccz_ancillas = [cirq.NamedQubit(f"CCZ_{index}") for index in range(3)]
    expected = cirq.Circuit(
        cirq.Moment(
            [
                cirq.I(t_ancilla).with_tags(ResourceStateTag(cirq.T, 0, 0, 1)),
            ]
        ),
        cirq.Moment([cirq.CNOT(q0, t_ancilla)]),
        cirq.Moment(
            [
                cirq.I(ccz_ancillas[0]).with_tags(ResourceStateTag(cirq.CCZ, 1, 0, 3)),
                cirq.I(ccz_ancillas[1]).with_tags(ResourceStateTag(cirq.CCZ, 1, 1, 3)),
                cirq.I(ccz_ancillas[2]).with_tags(ResourceStateTag(cirq.CCZ, 1, 2, 3)),
            ]
        ),
        cirq.Moment(
            [
                cirq.CNOT(q0, ccz_ancillas[0]),
                cirq.CNOT(q1, ccz_ancillas[1]),
                cirq.CNOT(q2, ccz_ancillas[2]),
            ]
        ),
    )
    assert transformed == expected

    tags = [
        tag
        for operation in transformed.all_operations()
        for tag in operation.tags
        if isinstance(tag, ResourceStateTag)
    ]
    assert all(tag.basis is ResourceStateBasis.Z for tag in tags)


def test_toffoli_reverses_cnot_direction_on_target_resource_leg() -> None:
    control0, control1, target = cirq.LineQubit.range(3)

    transformed = replace_resource_gates(cirq.Circuit(cirq.TOFFOLI(control0, control1, target)))

    ancillas = [cirq.NamedQubit(f"TOFFOLI_{index}") for index in range(3)]
    assert cirq.CNOT(control0, ancillas[0]) in transformed.all_operations()
    assert cirq.CNOT(control1, ancillas[1]) in transformed.all_operations()
    assert cirq.CNOT(ancillas[2], target) in transformed.all_operations()
    assert cirq.CNOT(target, ancillas[2]) not in transformed.all_operations()

    tags = sorted(
        (
            tag
            for operation in transformed.all_operations()
            for tag in operation.tags
            if isinstance(tag, ResourceStateTag)
        ),
        key=lambda tag: tag.qubit_index,
    )
    assert [tag.basis for tag in tags] == [
        ResourceStateBasis.Z,
        ResourceStateBasis.Z,
        ResourceStateBasis.X,
    ]


def test_joint_measurement_teleportation() -> None:
    control0, control1, target = cirq.LineQubit.range(3)

    transformed = replace_resource_gates(
        cirq.Circuit(cirq.TOFFOLI(control0, control1, target)), use_joint_meas=True
    )

    ancillas = [cirq.NamedQubit(f"TOFFOLI_{index}") for index in range(3)]
    assert (
        cirq.measure_single_paulistring(cirq.PauliString(cirq.Z(control0), cirq.Z(ancillas[0])))
        in transformed.all_operations()
    )
    assert (
        cirq.measure_single_paulistring(cirq.PauliString(cirq.Z(control1), cirq.Z(ancillas[1])))
        in transformed.all_operations()
    )
    assert (
        cirq.measure_single_paulistring(cirq.PauliString(cirq.X(target), cirq.X(ancillas[2])))
        in transformed.all_operations()
    )

    tags = sorted(
        (
            tag
            for operation in transformed.all_operations()
            for tag in operation.tags
            if isinstance(tag, ResourceStateTag)
        ),
        key=lambda tag: tag.qubit_index,
    )
    assert [tag.basis for tag in tags] == [
        ResourceStateBasis.Z,
        ResourceStateBasis.Z,
        ResourceStateBasis.X,
    ]


def test_preserves_clifford_and_non_unitary_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.depolarize(0.1)(q1),
        cirq.measure(q0, key="result"),
    )

    assert replace_resource_gates(circuit) == circuit


def test_replaces_classically_controlled_gates() -> None:
    control, q0, q1 = cirq.LineQubit.range(3)
    condition = cirq.KeyCondition(cirq.MeasurementKey("enable"))
    circuit = cirq.Circuit(
        cirq.measure(control, key="enable"),
        cirq.S(q0).with_classical_controls(condition),
        cirq.CZ(q0, q1).with_classical_controls(condition),
    )

    transformed = replace_resource_gates(circuit)

    s_ancilla = cirq.NamedQubit("S_0")
    cz_ancillas = [cirq.NamedQubit(f"CZ_{index}") for index in range(2)]
    expected = cirq.Circuit(
        cirq.Moment([cirq.measure(control, key="enable")]),
        cirq.Moment([cirq.I(s_ancilla).with_tags(ResourceStateTag(cirq.S, 0, 0, 1, (condition,)))]),
        cirq.Moment([cirq.CNOT(q0, s_ancilla)]),
        cirq.Moment(
            [
                cirq.I(cz_ancillas[0]).with_tags(ResourceStateTag(cirq.CZ, 1, 0, 2, (condition,))),
                cirq.I(cz_ancillas[1]).with_tags(ResourceStateTag(cirq.CZ, 1, 1, 2, (condition,))),
            ]
        ),
        cirq.Moment(
            [
                cirq.CNOT(q0, cz_ancillas[0]),
                cirq.CNOT(q1, cz_ancillas[1]),
            ]
        ),
    )
    assert transformed == expected

    tags = [
        tag
        for operation in transformed.all_operations()
        for tag in operation.tags
        if isinstance(tag, ResourceStateTag)
    ]
    assert all(tag.is_classically_controlled for tag in tags)
    assert all(tag.control_keys == {cirq.MeasurementKey("enable")} for tag in tags)


def test_preserves_classically_controlled_non_unitary_operation() -> None:
    q = cirq.LineQubit(0)
    operation = cirq.reset(q).with_classical_controls("enable")
    circuit = cirq.Circuit(operation)

    assert replace_resource_gates(circuit) == circuit


def test_replaces_numeric_and_parameterized_rz_gates() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.rz(0.123)(q0), cirq.rz(theta)(q1))

    transformed = replace_resource_gates(circuit)

    tags = [
        tag
        for operation in transformed.all_operations()
        for tag in operation.tags
        if isinstance(tag, ResourceStateTag)
    ]
    assert tags == [
        ResourceStateTag(cirq.rz(0.123), 0, 0, 1),
        ResourceStateTag(cirq.rz(theta), 1, 0, 1),
    ]
    assert cirq.CNOT(q0, cirq.NamedQubit("RZ_0")) in transformed.all_operations()
    assert cirq.CNOT(q1, cirq.NamedQubit("RZ_1")) in transformed.all_operations()


def test_ancilla_names_do_not_collide_with_input_qubits() -> None:
    data = cirq.NamedQubit("data")
    existing = cirq.NamedQubit("T_0")
    circuit = cirq.Circuit(cirq.X(existing), cirq.T(data))

    transformed = replace_resource_gates(circuit)

    assert cirq.NamedQubit("T_0") in transformed.all_qubits()
    assert cirq.NamedQubit("T_1") in transformed.all_qubits()
    assert cirq.CNOT(data, cirq.NamedQubit("T_1")) in transformed.all_operations()


def test_empty_circuit_is_unchanged() -> None:
    assert replace_resource_gates(cirq.Circuit()) == cirq.Circuit()
