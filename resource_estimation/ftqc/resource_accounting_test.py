# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end resource ledgers for small circuits with hand-countable costs."""

from collections import Counter

import cirq
import pytest
import stim

import resource_estimation.ftqc as ftqc
import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.logical_operations import CSSLogicalOperations
from resource_estimation.ftqc.qldpc_surgery import (
    build_joint_logical_pauli_measurement_circuit,
)


@pytest.fixture(scope="module")
def steane_architecture() -> ftqc.DefaultMovement:
    patch = lsp.CodePatch("steane")
    logical_operations = CSSLogicalOperations.from_circuits(
        patch,
        h_circuit=stim.Circuit("H 0 1 2 3 4 5 6"),
        s_circuit=stim.Circuit("S_DAG 0 1 2 3 4 5 6"),
    )
    return ftqc.DefaultMovement(
        patch=patch,
        logical_operations=logical_operations,
        patch_span=4,
        idling=False,
        post_op_correction=False,
        syndrome_rounds=1,
        t_state_transfer_rounds=1,
    )


def _compile_without_factories(
    circuit: cirq.Circuit, architecture: ftqc.DefaultMovement
) -> tuple[ftqc.MovementLayout, cirq.Circuit]:
    layout = ftqc.MovementLayout(circuit, num_t_factories=0)
    return layout, ftqc.ft_compile(layout, architecture, verbose=0)


def test_single_steane_h_matches_hand_resource_ledger(
    steane_architecture: ftqc.DefaultMovement,
) -> None:
    """One initial syndrome round followed by one seven-qubit transversal H."""
    qubit = cirq.LineQubit(0)
    layout, compiled = _compile_without_factories(cirq.Circuit(cirq.H(qubit)), steane_architecture)
    operations = list(compiled.all_operations())

    assert [str(operation.gate) for operation in operations] == ["SE(1)", "H"]

    # This ledger deliberately records the current per-check CZ decomposition.  Each of the
    # three weight-four X checks has two ancilla basis changes and two basis changes on each
    # supported data qubit.  Each Z check has only the two ancilla basis changes.  Basis changes
    # shared between overlapping X checks are not folded together by the current serial model.
    x_check_basis_changes = 3 * (2 + 2 * 4)
    z_check_basis_changes = 3 * 2
    transversal_h_gates = 7
    assert x_check_basis_changes + z_check_basis_changes + transversal_h_gates == 43

    estimator = ftqc.ResourceEstimator(steane_architecture)
    assert estimator.serial_circuit_cost(compiled) == {
        cirq.MeasurementGate: 6,
        cirq.ResetChannel: 6,
        cirq.CZ: 24,
        cirq.PhasedXZGate: 43,
        cirq.QubitPermutationGate: 14,
    }
    assert estimator.parallel_circuit_cost(compiled) == {
        cirq.MeasurementGate: 1,
        cirq.ResetChannel: 1,
        cirq.CZ: 6,
        cirq.PhasedXZGate: 3,
        cirq.QubitPermutationGate: 14,
    }
    assert estimator.physical_qubits(compiled, layout=layout) == 13


def test_single_steane_cnot_matches_hand_resource_ledger(
    steane_architecture: ftqc.DefaultMovement,
) -> None:
    """Two parallel initial syndrome rounds, zone movement, and a transversal CNOT."""
    control, target = cirq.LineQubit.range(2)
    layout, compiled = _compile_without_factories(
        cirq.Circuit(cirq.CNOT(control, target)), steane_architecture
    )
    operations = list(compiled.all_operations())

    assert [str(operation.gate) for operation in operations] == [
        "SE(1)",
        "SE(1)",
        "MOVE_IZ",
        "MOVE_IZ",
        "CNOT",
        "MOVE_IZ",
        "MOVE_IZ",
    ]

    # Serial: two syndrome ledgers + four moves + seven CZs and fourteen one-qubit gates.
    estimator = ftqc.ResourceEstimator(steane_architecture)
    assert estimator.serial_circuit_cost(compiled) == {
        cirq.MeasurementGate: 12,
        cirq.ResetChannel: 12,
        cirq.CZ: 55,
        cirq.PhasedXZGate: 86,
        cirq.QubitPermutationGate: 32,
    }
    # The two syndrome rounds and the two moves in each direction occur in parallel by block.
    assert estimator.parallel_circuit_cost(compiled) == {
        cirq.MeasurementGate: 1,
        cirq.ResetChannel: 1,
        cirq.CZ: 7,
        cirq.PhasedXZGate: 4,
        cirq.QubitPermutationGate: 16,
    }
    assert estimator.physical_qubits(compiled, layout=layout) == 26


def test_surface_to_steane_adapter_matches_hand_resource_ledger(
    steane_architecture: ftqc.DefaultMovement,
) -> None:
    """Audit both qLDPC's raw circuit and the complete code-teleportation protocol."""
    resource = build_joint_logical_pauli_measurement_circuit(
        steane_architecture.cultivation_patch,
        steane_architecture.patch,
        basis="Z",
        rounds=1,
    )

    # Count raw physical target groups without using count_stim_resources(), which is part of
    # the production path under test.  Metadata, detectors, observables, and ticks have no
    # physical target groups in this ledger.
    raw_stim_counts: Counter[str] = Counter()
    for instruction in resource.circuit.flattened():
        if instruction.name in {"RX", "MX", "CX", "CZ"}:
            raw_stim_counts[instruction.name] += len(instruction.target_groups())

    assert raw_stim_counts == {
        "RX": 77,
        "MX": 77,
        "CX": 112,
        "CZ": 128,
    }
    assert len(resource.left_data_ids) == 49
    assert len(resource.right_data_ids) == 7
    assert len(resource.temporary_ids) == 11
    assert resource.circuit.num_qubits == 133
    assert 133 - 49 - 7 - 11 == 66  # syndrome/check qubits

    # RX = reset + H, MX = H + measurement, and CX/CZ both use one native CZ.
    qldpc_serial = Counter(
        {
            cirq.ResetChannel: raw_stim_counts["RX"],
            cirq.MeasurementGate: raw_stim_counts["MX"],
            cirq.PhasedXZGate: raw_stim_counts["RX"] + raw_stim_counts["MX"],
            cirq.CZ: raw_stim_counts["CX"] + raw_stim_counts["CZ"],
        }
    )
    assert qldpc_serial == {
        cirq.ResetChannel: 77,
        cirq.MeasurementGate: 77,
        cirq.PhasedXZGate: 154,
        cirq.CZ: 240,
    }

    target_plus_prep = Counter({cirq.ResetChannel: 7, cirq.PhasedXZGate: 7})
    target_syndrome_round = Counter(
        {
            cirq.MeasurementGate: 6,
            cirq.ResetChannel: 6,
            cirq.CZ: 24,
            cirq.PhasedXZGate: 36,
        }
    )
    source_logical_x_measurement = Counter({cirq.PhasedXZGate: 49, cirq.MeasurementGate: 49})
    two_target_syndrome_rounds = Counter(
        {gate: 2 * count for gate, count in target_syndrome_round.items()}
    )
    expected_serial = (
        qldpc_serial + target_plus_prep + two_target_syndrome_rounds + source_logical_x_measurement
    )

    # qLDPC contributes depths R=1, H=2, CZ=9, M=1.  The remaining depths come from target
    # preparation, two target syndrome rounds, and destructive source-X measurement.
    expected_parallel = Counter(
        {
            cirq.ResetChannel: 4,
            cirq.MeasurementGate: 4,
            cirq.CZ: 21,
            cirq.PhasedXZGate: 8,
        }
    )
    movement_layers = 2 * (expected_parallel[cirq.CZ] + expected_parallel[cirq.MeasurementGate])
    assert movement_layers == 50
    expected_serial[cirq.QubitPermutationGate] = movement_layers
    expected_parallel[cirq.QubitPermutationGate] = movement_layers

    assert expected_serial == {
        cirq.MeasurementGate: 138,
        cirq.ResetChannel: 96,
        cirq.CZ: 288,
        cirq.PhasedXZGate: 282,
        cirq.QubitPermutationGate: 50,
    }
    assert expected_parallel == {
        cirq.MeasurementGate: 4,
        cirq.ResetChannel: 4,
        cirq.CZ: 21,
        cirq.PhasedXZGate: 8,
        cirq.QubitPermutationGate: 50,
    }

    operation = lsp.MagicStateCodeTeleport()(cirq.LineQubit(0))
    assert steane_architecture.gate_cost(operation) == expected_serial
    assert steane_architecture.moment_cost(operation) == expected_parallel
    assert steane_architecture.t_factory_physical_qubits == 133
