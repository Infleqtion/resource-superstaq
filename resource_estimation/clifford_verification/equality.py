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
"""Equality checking procedure for supported Cirq Cliffords with resource preparations, not a full decision procedure"""

from __future__ import annotations

import cirq

from .resources import NotCertifiedError, extract_resources
from .stabilizers import channel_mixtures_equal
from .tableau_channels import channel_mixture


def cliffords_equal(
    left: cirq.AbstractCircuit,
    right: cirq.AbstractCircuit,
) -> bool:
    """Check exact equality of two clifford circuits via simulation.

    Offers support for logical quantum channels of Cliffords, PPMs, single qubit measurement, and single-qubit Pauli and S/S-dagger corrections with affine conditions.
    S corrections use exact weighted branching; other measurements remain symbolic.
    Handles grouped resource-state preparation by separating out the resource states and checking they are
    symbolically equal after performing simulation on the remaining Clifford circuit. Failure to
    certify should not be taken as inequality of logical quantum channels for this reason.
    Only unconditional compiler ResourceStateTag markers are accepted as resource preparations.
    """
    # Pull out resource preparations so we can do clifford simulation on the remainder for left and right circuits
    left, left_resources = extract_resources(left)
    right, right_resources = extract_resources(right)

    # Union of qubits, including wires present on only one side.
    common = tuple(cirq.QubitOrder.DEFAULT.order_for(left.all_qubits() | right.all_qubits()))

    # Give the boundaries for the left and right circuits, as well as an indicator that is True when boundaries exclude circuit qubits e.g. resets/measurements are initial/terminal.
    left_in, left_out, left_marked = _infer_boundaries(left)
    right_in, right_out, right_marked = _infer_boundaries(right)

    if not left_marked and not right_marked and not left_resources and not right_resources:
        left_in = left_out = right_in = right_out = common

    # Collections of marked resource qubits
    left_resource_qubits = tuple(q for group in left_resources for q in group.qubits)
    right_resource_qubits = tuple(q for group in right_resources for q in group.qubits)

    # Logical inputs, excluding prepared ancillas and resource inputs.
    left_in = tuple(q for q in left_in if q not in left_resource_qubits)
    right_in = tuple(q for q in right_in if q not in right_resource_qubits)

    if len(left_in) != len(right_in) or len(left_out) != len(right_out):
        raise ValueError("Both circuits must have the same logical input and output counts")

    # Validate both Clifford remainders before reporting any resource mismatch.
    left_channel = channel_mixture(left, left_in + left_resource_qubits, left_out)
    right_channel = channel_mixture(right, right_in + right_resource_qubits, right_out)

    # Check that whole resource preparations match, including ordered leg bases
    if tuple(group.state for group in left_resources) != tuple(group.state for group in right_resources):
        raise NotCertifiedError("Not certified: resource-state preparations do not match")
    equal = channel_mixtures_equal(left_channel, right_channel)
    if not equal and (left_resources or right_resources):
        raise NotCertifiedError(
            "Not certified: open-resource channels differ. Supplied resource states may still make them equal"
        )
    return equal


# Allows us to have different physical wire counts while requiring us to match logical inputs/outputs
def _infer_boundaries(
    circuit: cirq.AbstractCircuit,
) -> tuple[tuple[cirq.Qid, ...], tuple[cirq.Qid, ...], bool]:
    """Identify inputs and outputs to the circuit by excluding initial resets and terminal cirq.measure readouts."""
    first: dict[cirq.Qid, cirq.Operation] = {}
    last: dict[cirq.Qid, cirq.Operation] = {}
    for operation in circuit.all_operations():
        for q in operation.qubits:
            first.setdefault(q, operation)
            last[q] = operation
    order = tuple(cirq.QubitOrder.DEFAULT.order_for(circuit.all_qubits()))
    # Inputs are qubits do not have their first operator as a reset.
    inputs = tuple(q for q in order if not isinstance(first[q].untagged.gate, cirq.ResetChannel))
    # Outputs are qubits that do not have their last operator as a measurement.
    outputs = tuple(q for q in order if not isinstance(last[q].untagged.gate, cirq.MeasurementGate))
    return inputs, outputs, inputs != order or outputs != order
