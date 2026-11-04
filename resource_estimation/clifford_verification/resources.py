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
"""Recognizing compiler ResourceStateTag declarations for T/T-dagger states."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import cirq

from resource_estimation.compile_gateset.ppm_compiler.resource_state_injection import (
    ResourceStateTag,
)


class NonCliffordError(ValueError):
    """Clifford verifier called with non-clifford gate in circuit."""


class NotCertifiedError(ValueError):
    """Unable to certify equality due to resource-matching failure or circuit equality failure."""

# T state representative
@dataclass(frozen=True)
class TState:
    """Exact symbolic state (|0> + exp(+/-i*pi/4)|1>)/sqrt(2)."""

    dagger: bool = False

    def __post_init__(self) -> None:
        if type(self.dagger) is not bool:
            raise ValueError("dagger must be a boolean")


def reject_non_clifford(operation: cirq.Operation) -> None:
    """Reject non-Clifford data gates; resource declarations are tagged identities."""
    base = operation.without_classical_controls().untagged
    if (
        not cirq.is_parameterized(base)
        and cirq.has_unitary(base)
        and not cirq.has_stabilizer_effect(base)
    ):
        raise NonCliffordError(f"Verifier called with non-Cliffords: {operation!r}")

# Recognizes stand-in operators for T states
def resource_state(operation: cirq.Operation) -> TState | None:
    gate = operation.untagged.gate
    tag = next((tag for tag in operation.tags if isinstance(tag, ResourceStateTag)), None)
    if tag is not None:
        if (
            gate != cirq.I
            or len(operation.qubits) != 1
            or tag.num_qubits != 1
            or tag.qubit_index != 0
            or tag.classical_controls
            or tag.basis != cirq.Z
            or not isinstance(tag.gate, cirq.ZPowGate)
            or cirq.is_parameterized(tag.gate)
        ):
            raise ValueError(
                "Unsupported resource-state marker; expected an unconditional Z-basis T preparation"
            )
        exponent = Fraction(str(tag.gate.exponent)) % 2
        if exponent not in (Fraction(1, 4), Fraction(7, 4)):
            raise ValueError("Unsupported resource-state marker; expected T or T dagger")
        return TState(exponent == Fraction(7, 4))
    return None

# Split a circuit into two parts; resource states and cliffords
def extract_resources(
    circuit: cirq.AbstractCircuit,
) -> tuple[cirq.Circuit, tuple[tuple[cirq.Qid, TState], ...]]:
    seen: set[cirq.Qid] = set()
    resources: list[tuple[cirq.Qid, TState]] = []
    moments = []
    for index, moment in enumerate(circuit):
        operations = []
        for operation in moment:
            # Throw an error if handed a non clifford unitary
            try:
                reject_non_clifford(operation)
            except NonCliffordError as error:
                raise NonCliffordError(f"Moment {index}: {error}") from error

            # Check if operator is a ResourceState by utilizing ResourceStateTag
            state = resource_state(operation)
            
            # Check to make sure is not a StatePreparationChannel
            if state is None and isinstance(operation.untagged.gate, cirq.StatePreparationChannel):
                raise ValueError(
                    f"Moment {index}: unsupported state preparation; use a compiler ResourceStateTag"
                )

            # Currently only support single-qubit resources, but multi-qubit resources could be extended with proper bookkeeping
            if state is not None:
                if operation.classical_controls or len(operation.qubits) != 1:
                    raise ValueError(
                        f"Moment {index}: resource preparations must be unconditional and single-qubit"
                    )

                # Make sure resource prep is on a qubit we have yet to see
                qubit = operation.qubits[0]
                if qubit in seen:
                    raise ValueError(
                        f"Moment {index}: resource preparation must be the first use of {qubit}"
                    )
                resources.append((qubit, state))
                operations.append(cirq.I(qubit))
            else:
                operations.append(operation)
            seen.update(operation.qubits)
        moments.append(cirq.Moment(operations))

    # Sort by kind (T or T\dagger) then on the Cirq qubit order
    order = {q: j for j, q in enumerate(cirq.QubitOrder.DEFAULT.order_for(circuit.all_qubits()))}
    resources.sort(key=lambda item: (item[1].dagger, order[item[0]]))
    return cirq.Circuit(moments), tuple(resources)
