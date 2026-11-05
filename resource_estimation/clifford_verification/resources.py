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
"""Recognizing grouped compiler ResourceStateTag preparation declarations."""

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

@dataclass(frozen=True)
class ResourceState:
    """Symbolic preparation: a compiler gate and the basis of each ordered leg.

    Equality compares declarations, not state vectors. Resource IDs identify groups
    within one circuit and are deliberately excluded from this descriptor.
    """

    gate: cirq.Gate
    bases: tuple[cirq.Pauli, ...]


@dataclass(frozen=True)
class ResourceGroup:
    """One preparation and its wires, ordered by the compiler's qubit_index."""

    state: ResourceState
    qubits: tuple[cirq.Qid, ...]


def reject_non_clifford(operation: cirq.Operation) -> None:
    """Reject non-Clifford data gates; resource declarations are tagged identities."""
    base = operation.without_classical_controls().untagged
    if (
        not cirq.is_parameterized(base)
        and cirq.has_unitary(base)
        and not cirq.has_stabilizer_effect(base)
    ):
        raise NonCliffordError(f"Verifier called with non-Cliffords: {operation!r}")

def resource_state(operation: cirq.Operation) -> ResourceStateTag | None:
    """Recognize one leg of an unconditional compiler preparation declaration."""
    tag = next((tag for tag in operation.tags if isinstance(tag, ResourceStateTag)), None)
    if tag is None:
        return None
    if (
        operation.untagged.gate != cirq.I                       # Tags should only be on I gates (I believe - this may change)
        or len(operation.qubits) != 1                           # Tagged operators should be single-qubit operators, for multi-qubit resource states we want multiple single-qubit-operator tags
        or operation.classical_controls                         # Tagged operators should not be classically controlled
        or tag.classical_controls                               # '' ''
        or cirq.is_parameterized(tag.gate)                      # Future: treat matching adaptive corrections as opaque operation boundaries.
                                                                # Verify the surrounding segments while retaining the classical records used
                                                                # by each correction, and match correction targets and control conditions.
                                                                # This could avoid correction branching, but may reject valid rewrites that
                                                                # move operations across these boundaries.
                                                                # For example, we could no longer check
                                                                # --o-----   -----o--
                                                                #   |      =      |
                                                                # --S--Z--   --Z--S--
        or cirq.qid_shape(tag.gate) != (2,) * tag.num_qubits    # Every leg has been claimed and is one qubit
        or not 0 <= tag.qubit_index < tag.num_qubits            # Marker validity check
    ):
        raise ValueError("Unsupported resource-state marker; expected an unconditional qubit preparation")
    return tag


def _canonical_gate(gate: cirq.Gate) -> cirq.Gate:
    # Preserve the existing exact matching of T/T-dagger exponent aliases.
    # A global phase on a resource preparation does not change its state.
    if isinstance(gate, cirq.ZPowGate):
        exponent = Fraction(str(gate.exponent)) % 2
        if exponent in (Fraction(1, 4), Fraction(7, 4)):
            return cirq.T if exponent == Fraction(1, 4) else cirq.T**-1
    return gate


def extract_resources(
    circuit: cirq.AbstractCircuit,
) -> tuple[cirq.Circuit, tuple[ResourceGroup, ...]]:
    """Replace resource markers with open wires and collect complete preparations.

    The compiler declares one identity per leg. Legs may occur in different
    moments, but each must be the first use of its wire. Grouping preserves
    entanglement information that would be lost by matching legs independently.
    """
    seen: set[cirq.Qid] = set()
    groups: dict[int, dict[int, tuple[cirq.Qid, ResourceStateTag]]] = {}
    moments = []
    for index, moment in enumerate(circuit):
        operations = []
        for operation in moment:
            try:
                reject_non_clifford(operation)
            except NonCliffordError as error:
                raise NonCliffordError(f"Moment {index}: {error}") from error
            tag = resource_state(operation)
            if tag is None and isinstance(operation.untagged.gate, cirq.StatePreparationChannel): # Open end to extend this to work with StatePreparationChannels - ResourceStateTags were chosen as they are part of the PPM compiler
                raise ValueError(
                    f"Moment {index}: unsupported state preparation; use a compiler ResourceStateTag"
                )
            if tag is not None:
                qubit = operation.qubits[0]
                if qubit in seen:
                    raise ValueError(
                        f"Moment {index}: resource preparation must be the first use of {qubit}"
                    )
                legs = groups.setdefault(tag.resource_id, {})
                if tag.qubit_index in legs:
                    raise ValueError(f"Resource {tag.resource_id}: duplicate leg {tag.qubit_index}")
                legs[tag.qubit_index] = (qubit, tag)
                operations.append(cirq.I(qubit))
            else:
                operations.append(operation)
            seen.update(operation.qubits)
        moments.append(cirq.Moment(operations))

    resources = []
    for resource_id, legs in groups.items():
        first = next(iter(legs.values()))[1]
        gate = _canonical_gate(first.gate)
        if any(
            tag.num_qubits != first.num_qubits or _canonical_gate(tag.gate) != gate
            for _, tag in legs.values()
        ):
            raise ValueError(f"Resource {resource_id}: inconsistent preparation descriptors")
        if set(legs) != set(range(first.num_qubits)):
            raise ValueError(f"Resource {resource_id}: incomplete preparation")
        ordered = [legs[j] for j in range(first.num_qubits)]
        state = ResourceState(gate, tuple(tag.basis for _, tag in ordered))
        resources.append(ResourceGroup(state, tuple(q for q, _ in ordered)))

    # Pair whole groups deterministically; never reorder legs within a group.
    # Equal resources are paired by physical qubit order, without permutation search.
    order = {q: j for j, q in enumerate(cirq.QubitOrder.DEFAULT.order_for(circuit.all_qubits()))}
    resources.sort(
        key=lambda group: (
            repr(group.state.gate),
            tuple(str(basis) for basis in group.state.bases),
            tuple(order[q] for q in group.qubits),
        )
    )
    return cirq.Circuit(moments), tuple(resources)
