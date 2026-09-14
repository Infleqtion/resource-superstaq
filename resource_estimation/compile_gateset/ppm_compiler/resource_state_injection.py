from __future__ import annotations

import dataclasses
import enum
import re
from collections import defaultdict

import cirq


@dataclasses.dataclass(frozen=True)
class ResourceStateTag:
    """Metadata describing one qubit of an injected resource state.

    Tags with the same ``resource_id`` belong to the same resource state. The original gate is
    retained so that a later pass can replace the identity marker carrying this tag with the
    appropriate state preparation. A non-empty ``classical_controls`` tuple indicates that the
    original gate was classically controlled.

    Attributes:
        gate: The gate replaced by the resource state.
        resource_id: A circuit-local identifier shared by all qubits of this resource state.
        qubit_index: The position of this ancilla in the gate's ordered qubit tuple.
        num_qubits: The total number of qubits in the resource state.
        classical_controls: Conditions that controlled the original gate. Empty for an
            unconditionally applied gate.
        basis: Injection basis of this resource-state leg. A Z-basis leg uses the data qubit as
            the teleportation CNOT control; an X-basis leg reverses that direction.
    """

    gate: cirq.Gate
    resource_id: int
    qubit_index: int
    num_qubits: int
    classical_controls: tuple[cirq.Condition, ...] = ()
    basis: cirq.Pauli = cirq.Z

    @property
    def is_classically_controlled(self) -> bool:
        """Whether the resource state represents a classically controlled gate."""
        return bool(self.classical_controls)

    @property
    def control_keys(self) -> frozenset[cirq.MeasurementKey]:
        """Measurement keys referenced by the classical control conditions."""
        return frozenset(key for condition in self.classical_controls for key in condition.keys)


def _gate_name(gate: cirq.Gate) -> str:
    """Return a concise gate-family name suitable for an ancilla name."""
    displayed_gate = getattr(gate, "bloq", gate)
    match = re.match(r"[A-Za-z][A-Za-z0-9]*", str(displayed_gate))
    if match:
        return match.group().upper()

    class_name = type(displayed_gate).__name__
    for suffix in ("PowGate", "Gate"):
        if class_name.endswith(suffix):
            class_name = class_name.removesuffix(suffix)
            break
    return class_name.upper()


def _fresh_ancilla(
    gate: cirq.Gate,
    used_qubits: set[cirq.Qid],
    next_indices: defaultdict[str, int],
) -> cirq.NamedQubit:
    gate_name = _gate_name(gate)
    while True:
        index = next_indices[gate_name]
        next_indices[gate_name] += 1
        ancilla = cirq.NamedQubit(f"{gate_name}_{index}")
        if ancilla not in used_qubits:
            used_qubits.add(ancilla)
            return ancilla


def _is_non_clifford_unitary(operation: cirq.Operation) -> bool:
    """Return whether an operation should be replaced by a resource state."""
    if operation.gate is None or cirq.has_stabilizer_effect(operation):
        return False

    # Parameterized rotations do not report a unitary until their parameters are resolved, but
    # they still need a resource state. Non-unitary channels and annotations are left unchanged.
    return cirq.has_unitary(operation) or cirq.is_parameterized(operation)


def _resource_state_basis(gate: cirq.Gate, qubit_index: int, num_qubits: int) -> cirq.Pauli:
    """Return the injection basis for one ordered gate leg.

    Controls in CCZ/CZ/CCX/CX use Z-basis injection, targets use X-basis injections.
    Single qubit rotations use the rotation basis.
    """
    if isinstance(gate, (cirq.CXPowGate, cirq.CCXPowGate)):
        if qubit_index == num_qubits - 1:
            return cirq.X
        else:
            return cirq.Z
    elif isinstance(gate, cirq.XPowGate):
        return cirq.X
    elif isinstance(gate, (cirq.ZPowGate, cirq.CZPowGate, cirq.CCZPowGate)):
        return cirq.Z
    else:
        raise ValueError(f"{gate} is not a supported resource state gate.")


def _injection_details(
    operation: cirq.Operation,
) -> tuple[cirq.Gate, tuple[cirq.Condition, ...]] | None:
    """Return the gate and controls when an operation needs resource-state injection."""
    classical_controls = tuple(sorted(operation.classical_controls, key=repr))
    uncontrolled_operation = operation.without_classical_controls()

    if uncontrolled_operation.gate is None:
        return None

    if classical_controls:
        # Classically controlled Clifford gates are represented as resource-state injections so
        # that a later pass can turn them into delayed-choice injections. This also ensures that
        # a classically controlled non-Clifford unitary is not accidentally missed merely because
        # Cirq's ClassicallyControlledOperation has no gate of its own.
        if cirq.has_unitary(uncontrolled_operation) or cirq.is_parameterized(
            uncontrolled_operation
        ):
            return uncontrolled_operation.gate, classical_controls
        return None

    if _is_non_clifford_unitary(uncontrolled_operation):
        return uncontrolled_operation.gate, ()
    return None


def replace_resource_gates(
    circuit: cirq.AbstractCircuit, use_joint_meas: bool = False
) -> cirq.Circuit:
    """Replace resource state gates with tagged resource ancillas and teleportation gates.

    A gate on ``n`` qubits is replaced by ``n`` fresh ancillas. Each ancilla first receives a
    tagged identity operation, which is only a marker for a future resource-state-preparation
    pass. Z-basis resource legs use ``CNOT(data, ancilla)``; X-basis legs use
    ``CNOT(ancilla, data)``. For example, Toffoli's two controls are Z-basis legs and its target
    is an X-basis leg.

    Uncontrolled non-Clifford gates and classically controlled unitary gates (including Clifford
    gates) are replaced. The exact classical control conditions are stored on the resource-state
    tag rather than applied to the teleportation CNOTs. Uncontrolled Clifford gates and
    non-unitary operations are copied unchanged. Every input moment produces one output moment;
    moments containing replacements produce one additional CNOT moment.

    Args:
        circuit: Circuit whose injectable gates should be replaced.
        use_joint_meas: If true, uses joint ZZ/XX measurements instead of CNOT gates

    Returns:
        A new circuit containing the untouched operations, tagged identity resource-state
        markers, and teleportation operations.
    """
    used_qubits = set(circuit.all_qubits())
    next_ancilla_indices: defaultdict[str, int] = defaultdict(int)
    output_moments: list[cirq.Moment] = []
    resource_id = 0

    for moment in circuit:
        first_moment_operations: list[cirq.Operation] = []
        teleportation_operations: list[cirq.Operation] = []

        for operation in moment.operations:
            injection_details = _injection_details(operation)
            if injection_details is None:
                first_moment_operations.append(operation)
                continue

            gate, classical_controls = injection_details
            num_qubits = len(operation.qubits)
            for qubit_index, data_qubit in enumerate(operation.qubits):
                ancilla = _fresh_ancilla(gate, used_qubits, next_ancilla_indices)
                basis = _resource_state_basis(gate, qubit_index, num_qubits)
                tag = ResourceStateTag(
                    gate=gate,
                    resource_id=resource_id,
                    qubit_index=qubit_index,
                    num_qubits=num_qubits,
                    classical_controls=classical_controls,
                    basis=basis,
                )
                first_moment_operations.append(cirq.I(ancilla).with_tags(tag))
                if use_joint_meas:
                    pauli_string: cirq.PauliString[cirq.Qid] = (
                        cirq.PauliString(cirq.Z(data_qubit), cirq.Z(ancilla))
                        if basis is cirq.Z
                        else cirq.PauliString(cirq.X(data_qubit), cirq.X(ancilla))
                    )
                    teleportation_operations.append(cirq.measure_single_paulistring(pauli_string))

                else:
                    cnot_qubits = (
                        (data_qubit, ancilla) if basis is cirq.Z else (ancilla, data_qubit)
                    )
                    teleportation_operations.append(cirq.CNOT(*cnot_qubits))

            resource_id += 1

        output_moments.append(cirq.Moment(first_moment_operations))
        if teleportation_operations:
            output_moments.append(cirq.Moment(teleportation_operations))

    return cirq.Circuit(output_moments)
