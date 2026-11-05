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
"""Exact channel signatures using Cirq tableaus and a stabilizer partial trace.

Pauli controls do not branch, conditional S gates branch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import cirq

from .controls import control_parity
from .stabilizers import Signature, reduced_signature


@dataclass(frozen=True)
class _ConditionalS:
    control: cirq.LineQubit
    target: cirq.Qid
    exponent: Fraction


WeightedChannels = tuple[tuple[Fraction, Signature], ...]


def channel_signature(
    circuit: cirq.AbstractCircuit, inputs: Sequence[cirq.Qid], outputs: Sequence[cirq.Qid]
) -> Signature:
    """Return a single stabilizer signature, rejecting if we have a mixture afterwards"""
    # Get the corresponding channel for our circuit
    branches = channel_mixture(circuit, inputs, outputs)
    # Require a single signature from our mixture with total weight 1
    if len(branches) != 1 or branches[0][0] != 1:
        raise ValueError("Conditional S requires a weighted channel mixture")
    # Return only the signature, not its weight (which becomes relevant when corrections exist)
    return branches[0][1]


def channel_mixture(
    circuit: cirq.AbstractCircuit, inputs: Sequence[cirq.Qid], outputs: Sequence[cirq.Qid]
) -> WeightedChannels:
    """Simulate a coherent dilation of the normalized Choi state, then trace it.

    Supports unitary Clifford gates, first-use ancilla resets, PPMs, single-qubit
    Z readout, and affine-controlled Pauli and S/S-dagger corrections. Unlisted inputs are |0>.
    Cirq does quantum evolution and projection; weights and group comparison are exact.

    All circuit and boundary wires must be dimension-two Cirq qubits. Each boundary
    list must contain distinct qubits; these input contracts are not checked here.
    """
    # Get the translated circuit and which qubits behave as its inputs and outputs
    ops, total, keep = _compile_channel(circuit, inputs, outputs)
    # Evolve on our inputs with Cirq's tableau simulator
    branches = _run_tableau(ops, total)
    # Trace environment wires and combine identical reduced channel states.
    merged: dict[Signature, Fraction] = {}
    for weight, tableau in branches:
        signature = reduced_signature(tableau, keep)
        merged[signature] = merged.get(signature, Fraction(0)) + weight
    return tuple((weight, signature) for signature, weight in merged.items())


def _compile_channel(
    circuit: cirq.AbstractCircuit, inputs: Sequence[cirq.Qid], outputs: Sequence[cirq.Qid]
) -> tuple[list[cirq.Operation | _ConditionalS], int, list[int]]:
    """Allocate Choi/record wires and translate the circuit without evolving it."""
    # Gather circuit and boundary qubits.
    qubits = tuple(
        cirq.QubitOrder.DEFAULT.order_for(circuit.all_qubits() | set(inputs) | set(outputs))
    )

    # Build wires dictionary mapping qubit names to qubits
    wires = dict(zip(qubits, cirq.LineQubit.range(len(qubits))))

    # Get our standard qubits and operations ready
    ops: list[cirq.Operation | _ConditionalS] = []
    total = len(qubits)

    # Add fresh qubits as needed, as we can create new states and discard qubits post measurement.
    def fresh() -> cirq.LineQubit:
        nonlocal total
        q = cirq.LineQubit(total)
        total += 1
        return q

    # Build the Choi state by "wrapping" each of the inputs around with a bell state
    # Equality of Choi states gives us equality for every possible input via simulation.
    references = [fresh() for _ in inputs]
    for reference, q in zip(references, inputs):
        ops.extend([cirq.H(reference), cirq.CNOT(reference, wires[q])])
    records: list[cirq.LineQubit] = []
    outcomes: list[cirq.MeasurementKey] = []
    seen: set[cirq.Qid] = set()

    # Translate operation to work on the Choi state and make sure operations are well formed.
    for moment in circuit:
        available = tuple(outcomes)
        for operation in moment:
            gate = operation.untagged.gate
            # Classically controlled operators become either a controlled Pauli or a _ConditionalS marker
            if operation.classical_controls:
                ops.extend(
                    _translate_control(
                        operation.transform_qubits(lambda q: wires[q]), available, records, fresh
                    )
                )
            # Check resets happen on fresh, non-input qubits.
            elif isinstance(gate, cirq.ResetChannel):
                q = operation.qubits[0]
                if q in seen or q in inputs:
                    raise ValueError("Only first-use non-input resets are supported")
            # Measurements need a fresh qubit to store the measurement outcome onto, used for feed-forward
            # PPMs copy parity onto the new qubit.
            elif isinstance(gate, (cirq.PauliMeasurementGate, cirq.MeasurementGate)):
                record = fresh()
                ops.extend(
                    _translate_measurement(
                        operation.untagged.transform_qubits(lambda q: wires[q]), record
                    )
                )
                key = cirq.measurement_key_obj(operation)
                outcomes.append(key)
                records.append(record)
            # Catchall for unsupported operators
            elif (
                cirq.is_parameterized(operation)
                or not cirq.has_unitary(operation)
                or not cirq.has_stabilizer_effect(operation)
            ):
                raise ValueError(f"Unsupported operation: {operation!r}")
            # All other operators get translated to operate on the Choi state.
            else:
                ops.append(operation.untagged.transform_qubits(lambda q: wires[q]))
            seen.update(operation.qubits)

    keep = [q.x for q in references] + [wires[q].x for q in outputs]
    # ops is the operators, clifford operations plus conditional S to branch on.
    # total is all internal qubits allocated, counting our references, circuit wires, measurement records, ancillae.
    # Keep is the qubits in our final Choi state, giving us our final comparison.
    return ops, total, keep


def _translate_control(
    operation: cirq.Operation,
    available: Sequence[cirq.MeasurementKey],
    records: Sequence[cirq.LineQubit],
    fresh: Callable[[], cirq.LineQubit],
) -> list[cirq.Operation | _ConditionalS]:
    """Compute an affine condition, then emit a coherent Pauli or branching S step.

    Operation qubits have already been mapped to the internal tableau wires.
    """
    ops: list[cirq.Operation | _ConditionalS] = []
    controls = tuple(operation.classical_controls)
    if len(controls) != 1 or len(operation.qubits) != 1:
        raise ValueError("Only one affine condition on a single-qubit correction is supported")
    bits, constant = control_parity(controls[0], available)
    correction = operation.without_classical_controls().untagged
    gate = correction.gate
    if not isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)) or cirq.is_parameterized(
        gate
    ):
        raise ValueError("Only Pauli and S/S-dagger feed-forward is supported")
    exponent = Fraction(str(gate.exponent))
    phase = 4 * exponent * Fraction(str(gate.global_shift))
    is_s = isinstance(gate, cirq.ZPowGate) and exponent.denominator == 2
    if not is_s and (exponent.denominator != 1 or phase.denominator != 1 or int(phase) % 2):
        raise ValueError("Non-Clifford coherent correction")
    control = fresh()
    if constant:
        ops.append(cirq.X(control))
    ops.extend(cirq.CNOT(records[j], control) for j in sorted(bits))
    target = operation.qubits[0]
    if is_s:
        # Conditioning on this record parity makes S a fixed Clifford.
        # Its global shift is only a branch phase and vanishes on tracing records.
        ops.append(_ConditionalS(control, target, exponent))
        return ops
    if int(exponent) % 2:
        if isinstance(gate, cirq.XPowGate):
            ops.append(cirq.CNOT(control, target))
        elif isinstance(gate, cirq.ZPowGate):
            ops.append(cirq.CZ(control, target))
        else:
            ops.extend([cirq.S(target) ** -1, cirq.CNOT(control, target), cirq.S(target)])
    if int(phase) % 8:
        ops.append(cirq.ZPowGate(exponent=int(phase) / 4)(control))
    return ops


def _translate_measurement(
    operation: cirq.Operation, record: cirq.LineQubit
) -> list[cirq.Operation]:
    """Copy signed Pauli parity into a fresh record wire and restore the basis.

    Operation qubits have already been mapped to the internal tableau wires.
    """
    gate = operation.gate
    assert isinstance(gate, (cirq.PauliMeasurementGate, cirq.MeasurementGate))
    ops: list[cirq.Operation] = []
    if isinstance(gate, cirq.MeasurementGate):
        if len(operation.qubits) != 1 or gate.confusion_map:
            raise ValueError("Only single-qubit readouts without confusion are supported")
        # Represent the Z readout for our measurement as a Pauli observable
        observable = cirq.DensePauliString("Z", coefficient=-1 if gate.full_invert_mask()[0] else 1)
    else:
        observable = gate.observable()
    basis: list[cirq.Operation] = []
    # Provide the change of basis necessary to convert us to Z measurements, simplifying feed-forward
    for pauli, q in zip(observable, operation.qubits):
        if pauli == cirq.X:
            basis.append(cirq.H(q))
        elif pauli == cirq.Y:
            basis.extend([cirq.S(q) ** -1, cirq.H(q)])
    ops.extend(basis)
    # Provide a clifford that records the outcome parity
    ops.extend(
        cirq.CNOT(q, record) for pauli, q in zip(observable, operation.qubits) if pauli != cirq.I
    )
    # Flip the record if our observable is negative
    if observable.coefficient == -1:
        ops.append(cirq.X(record))
    # Undo the change of basis
    ops.extend(op**-1 for op in reversed(basis))
    return ops


def _run_tableau(
    ops: Sequence[cirq.Operation | _ConditionalS], total: int
) -> list[tuple[Fraction, cirq.CliffordTableau]]:
    """Evolve with Cirq, retaining all weighted outcomes of conditional S steps."""
    # Evolve our state, branching when met with a conditional S.
    # Initially we have one tableau of weight 1
    branches = [(Fraction(1), cirq.CliffordTableau(total))]
    qubit_order = cirq.LineQubit.range(total)
    # Steps are either conditional S gates or clifford operators
    for step in ops:
        updated = []
        for weight, tableau in branches:
            if isinstance(step, _ConditionalS):
                # Conditional S's require a split
                for probability, outcome, projected in _split_record(tableau, step.control.x):
                    if outcome:
                        state = cirq.CliffordTableauSimulationState(projected, qubits=qubit_order)
                        cirq.act_on(
                            cirq.ZPowGate(exponent=float(step.exponent))(step.target),
                            state,
                        )
                    updated.append((weight * probability, projected))
            else:
                # Standard operators update each branch.
                state = cirq.CliffordTableauSimulationState(tableau, qubits=qubit_order)
                cirq.act_on(step, state)
                updated.append((weight, tableau))
        branches = updated
    return branches


def _split_record(
    tableau: cirq.CliffordTableau, axis: int
) -> list[tuple[Fraction, int, cirq.CliffordTableau]]:
    """Resolve a control outcome, returning one or two weighted tableaus
    """
    pivot = next((j for j in range(tableau.n) if tableau.xs[tableau.n + j, axis]), None)
    flip = tableau.stabilizers()[pivot] if pivot is not None else None
    projected = tableau.copy()
    outcome = projected.measure([axis], seed=0)[0]
    if flip is None:
        return [(Fraction(1), outcome, projected)]
    other = projected.copy()
    qubits = cirq.LineQubit.range(tableau.n)
    state = cirq.CliffordTableauSimulationState(other, qubits=qubits)
    cirq.act_on(flip.on(*qubits), state)
    return [(Fraction(1, 2), outcome, projected), (Fraction(1, 2), outcome ^ 1, other)]
