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
import json
import os
from collections import Counter
from math import ceil, pi
import numpy as np

import cirq
import pytest
import resource_estimation.architecture as arch
import resource_estimation.estimate as est
import resource_estimation.lattice_surgery_primitives as lsp
from cirq_superstaq import ParallelRGate
from resource_estimation.stim_functions import cultivate, load_saved_cost


@pytest.fixture
def lattice_architecture():
    return arch.DefaultLattice()


@pytest.fixture
def movement_architecture():
    return arch.DefaultMovement()


def test_architecture_exceptions(lattice_architecture, movement_architecture):
    with pytest.raises(ValueError, match="Cultivation cost"):
        _ = lattice_architecture.cultivate_cost(lsp.Cultivate(1).on(cirq.GridQubit(0, 0)))


def test_inplace_exact(lattice_architecture: arch.DefaultLattice):
    # TODO: Brainstorm a better way to test this feature
    actual_op_cost = lattice_architecture.cultivate_cost(
        lsp.Cultivate(pi / 2).on(cirq.GridQubit(0, 0))
    )
    # Tests that the parallel gates are counted correctly
    se_moment_cost = Counter(
        arch._syndrome_extract_cost(rounds=4, num_logical_qubits=1, d=7)["moment_cost"]
    )
    expected_Y_moment_cost = Counter(
        {cirq.PhasedXZGate: 10, cirq.CZ: 10, cirq.MeasurementGate: 2, cirq.ResetChannel: 2}
    )  # Includes both pieces
    expected_moment_cost = expected_Y_moment_cost + se_moment_cost
    assert expected_moment_cost == actual_op_cost["moment_cost"]

    # Tests that the serial gates are counted correctly
    # It does continue the assumption that we can just use a syndrome extraction cycle to approximate the total cost
    se_gate_cost = Counter(
        arch._syndrome_extract_cost(rounds=4, num_logical_qubits=1, d=7)["gate_cost"]
    )
    expected_Y_gate_cost = Counter(
        arch._syndrome_extract_cost(rounds=4, num_logical_qubits=1, d=7)["gate_cost"]
    )
    expected_gate_cost = expected_Y_gate_cost + se_gate_cost + expected_Y_gate_cost
    expected_gate_cost += {cirq.CZ: 2 * (7 - 1)}
    assert actual_op_cost["gate_cost"] == expected_gate_cost


@pytest.mark.parametrize("arc", [arch.DefaultMovement(), arch.DefaultLattice()])
def test_illegal_gate(arc):
    illegal_gate = cirq.Rx(rads=2).on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match="Gate not recognized"):
        _ = arc.gate_cost(illegal_gate)


@pytest.mark.parametrize("d", (3, 5, 7))
def test_movement_gate_costs(d):
    # Check that all gate costs are correct for movment architectures
    arc = arch.DefaultMovement(d=d)
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)

    # Check Cultivate

    op = lsp.Cultivate(pi / 4).on(qubit_a)
    cost = arc.gate_cost(op)
    if d < 7:
        with pytest.warns(UserWarning, match="Returning result for d=7"):
            base_cost = cultivate(dsurface=d, fault_distance=3)
    else:
        base_cost = cultivate(dsurface=d, fault_distance=3)
    expected_cost = base_cost["serial"]
    # To account for movement we add the QubitPermutationGates to the base cost
    expected_cost[cirq.QubitPermutationGate] = 2 * (
        base_cost["parallel"].get(cirq.CZ, 0) + base_cost["parallel"].get(cirq.MeasurementGate, 0)
    )
    assert expected_cost == cost

    # Check Move
    op1, op2 = lsp.Move(zone="interact").on_each(qubit_a, qubit_b)
    gate_cost = arc.gate_cost(op1)
    assert gate_cost == {cirq.QubitPermutationGate: 1}
    gate_cost = arc.gate_cost(op2)
    assert gate_cost == {cirq.QubitPermutationGate: 1}

    # Check CNOT
    op = cirq.CNOT.on(qubit_a, qubit_b)
    cost = arc.gate_cost(op)
    assert cost == {
        cirq.CZ: arc.patch.num_data_qubits,
        cirq.PhasedXZGate: arc.patch.num_data_qubits
        * 2,  # Two physical Hadamards on the data qubits
    }, cost

    # Check Syndrome on single qubit
    op = lsp.SyndromeExtract(1, 1).on(qubit_a)
    cost = arc.gate_cost(op)
    assert cost == {
        cirq.CZ: arc.patch.total_z_syndrome_cnots() + arc.patch.total_x_syndrome_cnots(),
        cirq.QubitPermutationGate: 10,
        cirq.MeasurementGate: arc.patch.num_measure_qubits,
        cirq.ResetChannel: arc.patch.num_measure_qubits,
        cirq.PhasedXZGate: (
            (10 * arc.patch.num_x_stabs(full=True))  # 5 Hadamards on left and 5 Hadamards on right
            + (2 * arc.patch.num_z_stabs(full=True))  # 1 Hadamard on left and 1 Hadamard on right
            + (
                6 * arc.patch.num_x_stabs(full=False)
            )  # 3 Hadamards on left and 3 Hadamards on right
            + (2 * arc.patch.num_z_stabs(full=False))  # 1 Hadamard on left and 1 Hadamard on right
        ),
    }

    # Check Syndrome on two qubits
    op = lsp.SyndromeExtract(2, 1).on(qubit_a, qubit_b)
    cost = arc.gate_cost(op)
    assert cost == {
        cirq.CZ: 2 * (arc.patch.total_z_syndrome_cnots() + arc.patch.total_x_syndrome_cnots()),
        cirq.QubitPermutationGate: 10,
        cirq.MeasurementGate: arc.patch.num_measure_qubits * 2,
        cirq.ResetChannel: arc.patch.num_measure_qubits * 2,
        cirq.PhasedXZGate: 2
        * (
            (10 * arc.patch.num_x_stabs(full=True))  # 5 Hadamards on left and 5 Hadamards on right
            + (2 * arc.patch.num_z_stabs(full=True))  # 1 Hadamard on left and 1 Hadamard on right
            + (
                6 * arc.patch.num_x_stabs(full=False)
            )  # 3 Hadamards on left and 3 Hadamards on right
            + (2 * arc.patch.num_z_stabs(full=False))  # 1 Hadamard on left and 1 Hadamard on right
        ),
    }

    # Check S gate
    op = cirq.S.on(qubit_a)
    cost = arc.gate_cost(op)
    expected_cost = Counter(arc.gate_cost(lsp.SyndromeExtract(1, 1).on(qubit_a)))
    expected_cost += Counter(
        {cirq.CZ: (d - 1) ** 2, cirq.PhasedXZGate: d, cirq.QubitPermutationGate: 2}
    )
    assert cost == expected_cost

    # Check simple gates
    ops_with_expectations = [
        (cirq.I.on(qubit_a), {}),
        (cirq.Z.on(qubit_a), {}),
        (cirq.X.on(qubit_a), {}),
        (
            cirq.MeasurementGate(1).on(qubit_a),
            {
                cirq.MeasurementGate: arc.patch.num_data_qubits,
            },
        ),
        (
            cirq.H.on(qubit_a),
            {
                cirq.PhasedXZGate: arc.patch.num_data_qubits,
                cirq.QubitPermutationGate: 1,
            },
        ),
    ]
    for op, expectation in ops_with_expectations:
        cost = arc.gate_cost(op)
        assert cost == expectation


@pytest.mark.parametrize("d", (3, 5, 7))
def test_lattice_gate_costs(d):
    # Test that gate costs are exact for lattice architectures

    arc = arch.DefaultLattice(d=d)
    qubit_a, qubit_b, qubit_c = (
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
    )

    # Check Merge on two qubits
    op = lsp.Merge(2, smooth=True).on(qubit_a, qubit_b)
    cost = arc.gate_cost(op)
    expected_cz_cost = d * (
        2 * (d - 1) ** 2 * 4 + 2 * (3 * (d // 2)) * 2 + 1 * (2 * (d - 1)) * 4 + 1 * 2 * 2
    )
    num_full_z_stabs = (d - 1) ** 2 + d - 1  # Two full patches plus one buffer zone
    num_full_x_stabs = num_full_z_stabs  # Only partials will be different
    num_partial_z_stabs = 2 * (d // 2)  # No partials in buffer zone since merge is smooth
    num_partial_x_stabs = (
        2 * num_partial_z_stabs + 2
    )  # Two extras in the buffer zone since merge is smooth

    expected_single_qubit_cost = d * (  # d rounds repetition
        2 * num_full_z_stabs
        + 2 * num_partial_z_stabs
        + 8 * num_full_x_stabs
        + 6 * num_partial_x_stabs
    )
    expected_measurement_cost = d * (
        2
        * (  # Measures from endpoint patches (2 endpoints)
            (d - 1) ** 2  # 1 from each full stabilizer
            + (3 * (d // 2))  # 1 from each partial stabilizer
        )
        + 1
        * (  # Measurements from buffer zones (1 buffer zone)
            2 * (d - 1) + 1 * 2  # 1 from each full stabilizer  # 1 from each partial stabilizer
        )
    )
    expected_reset_cost = expected_measurement_cost

    assert cost[cirq.CZ] == expected_cz_cost
    assert cost[cirq.MeasurementGate] == expected_measurement_cost
    assert cost[cirq.ResetChannel] == expected_reset_cost
    assert cost[cirq.PhasedXZGate] == expected_single_qubit_cost

    # Check Merge on three qubits
    op = lsp.Merge(3, smooth=True).on(qubit_a, qubit_b, qubit_c)
    cost = arc.gate_cost(op)
    expected_cz_cost += d * (4 * (d - 1) ** 2 + 4 * (d - 1) * 2 + +2 * (d // 2) * 2 + 4)
    expected_measurement_cost += d * ((d - 1) ** 2 + (d - 1) * 2 + 2 * (d // 2) + 2)
    expected_reset_cost = expected_measurement_cost
    expected_single_qubit_cost += d * (
        5 * (d - 1) ** 2  # Extra Rzs from full stabilizers in intermediate patch
        + 2 * (d - 1)  # Rzs from full Z stabilizers in buffer
        + 8 * (d - 1)  # Rzs from full X stabilizers in buffer
        + 12  # Rzs from partial X stabilizers in buffer
        + 6 * 2 * (d // 2)  # Rzs from partial X stabilizers in intermediate patch
    )
    assert cost[cirq.CZ] == expected_cz_cost
    assert cost[cirq.MeasurementGate] == expected_measurement_cost
    assert cost[cirq.ResetChannel] == expected_reset_cost
    assert cost[cirq.PhasedXZGate] == expected_single_qubit_cost

    # Check Split
    op = lsp.Split([1, 1], smooth=True).on(qubit_a, qubit_b)
    cost = arc.gate_cost(op)
    expected_cost = {
        cirq.MeasurementGate: 2 * arc.d + 1,
        cirq.PhasedXZGate: ceil((2 * arc.d + 1) / 2),
    }
    assert cost == expected_cost

    op = lsp.Split([1, 1], smooth=False).on(qubit_a, qubit_b)
    cost = arc.gate_cost(op)
    expected_cost = {cirq.MeasurementGate: 2 * d + 1}
    assert cost == expected_cost

    # Check Cultivate

    op = lsp.Cultivate(pi / 4).on(qubit_a)
    cost = arc.gate_cost(op)
    if d < 7:
        with pytest.warns(UserWarning, match="Returning result for d=7"):
            cost = arc.gate_cost(op)
            expected_cost = cultivate(dsurface=d, fault_distance=3)["serial"]
    else:
        cost = arc.gate_cost(op)
        expected_cost = cultivate(dsurface=d, fault_distance=3)["serial"]
    assert expected_cost == cost

    # Check Syndrome Extract on one qubit
    op = lsp.SyndromeExtract(1, arc.d).on(qubit_a)
    cost = arc.gate_cost(op)
    full_stabilizers = (d - 1) ** 2
    partial_stabilizers = 4 * (d // 2)
    czs_per_syndrome_extract = (4 * full_stabilizers) + (2 * partial_stabilizers)
    single_qubit_gates_per_syndrome_extract = 12 * (full_stabilizers // 2) + 8 * (
        partial_stabilizers // 2
    )
    measures_per_syndrome_extract = full_stabilizers + partial_stabilizers
    resets_per_syndrome_extract = measures_per_syndrome_extract
    # Where do these numbers come from?
    # GR Rz GR C C C C GR Rz GR M  <---X Stabilizer (full)
    # GR    GR Z | | | GR    GR
    # GR    GR   Z | | GR    GR
    # GR    GR     Z | GR    GR
    # GR    GR       Z GR    GR
    # GR    GR         GR    GR
    # GR Rz GR C C C C GR Rz GR M  <---Z Stabilizer (full)
    # GR Rz GR Z | | | GR Rz GR
    # GR Rz GR   Z | | GR Rz GR
    # GR Rz GR     Z | GR Rz GR
    # GR Rz GR       Z GR Rz GR
    # GR Rz GR C C     GR Rz GR M  <---X Stabilizer (partial)
    # GR    GR Z |     GR    GR
    # GR    GR   Z     GR    GR
    # GR Rz GR C C     GR Rz GR M  <---Z Stabilizer (partial)
    # GR Rz GR Z |     GR Rz GR
    # GR Rz GR   Z     GR Rz GR
    # For the whole stabilizer, there are only 4 GR gates
    # The number of measurements is equal to the total number of stabilizers
    # The number of CZ gates is equal to 4 times the number of full stabilizers plus 3 times the number of partial stabilizers
    # The number of Rz gates is equal to (10 + 2) times the number of full stabilizers plus (6 + 2) times the number of partial stabilizers
    # If we assume `d` rounds of error correction, we simply multiply these numbers by `d`
    expected_cost = {
        cirq.CZ: czs_per_syndrome_extract * d,
        cirq.MeasurementGate: measures_per_syndrome_extract * d,
        cirq.PhasedXZGate: single_qubit_gates_per_syndrome_extract * d,
        cirq.ResetChannel: resets_per_syndrome_extract * d,
    }
    assert cost == expected_cost

    # Check simple gates
    ops_with_expectations = [
        (cirq.I.on(qubit_a), {}),
        (cirq.Z.on(qubit_a), {}),
        (cirq.X.on(qubit_a), {}),
        (
            cirq.MeasurementGate(1).on(qubit_a),
            {cirq.MeasurementGate: arc.patch.num_data_qubits},
        ),
    ]
    for op, expectation in ops_with_expectations:
        cost = arc.gate_cost(op)
        assert expectation == cost


def test_self_returns(movement_architecture, lattice_architecture):
    # TODO: There are no self-returns anymore so this function is not well named
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    ops_and_expectations = [
        (cirq.ResetChannel().on(qubit_a), {cirq.ResetChannel: 97}),
        (lsp.ErrorCorrect(1).on(qubit_a), {}),
        (lsp.ErrorCorrect(2).on(qubit_a, qubit_b), {}),
    ]
    for arc in movement_architecture, lattice_architecture:
        for op, expectation in ops_and_expectations:
            cost = arc.gate_cost(op)
            assert expectation == cost


@pytest.mark.parametrize("d", (3, 5, 7))
def test_against_cultiv(d):
    # Test Syndrome Extract
    # Set up memory circuit
    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/../data/cultivate_costs.json", "r"
    ) as f:
        saved_resources = json.load(f)

    d_count = load_saved_cost(dsurface=d, op_key="memory_d_rounds")["serial"]
    # Remove the Logical Measurement operation
    print(d_count)
    d_count[cirq.MeasurementGate] -= d**2

    s_count = load_saved_cost(dsurface=d, op_key="memory_1_round")["serial"]
    # Remove the Logical Measurement operation
    s_count[cirq.MeasurementGate] -= d**2

    syndrome_estimate = Counter(
        arch.DefaultLattice(d=d).syndrome_extract_cost(
            lsp.SyndromeExtract(1, d).on(cirq.LineQubit(0))
        )["gate_cost"]
    )
    assert d_count[cirq.CZ] == syndrome_estimate[cirq.CZ]
    assert d_count[cirq.MeasurementGate] == syndrome_estimate[cirq.MeasurementGate]

    single_syndrome_estimate = {k: v / d for k, v in syndrome_estimate.items()}
    assert s_count[cirq.CZ] == single_syndrome_estimate[cirq.CZ]
    assert s_count[cirq.MeasurementGate] == single_syndrome_estimate[cirq.MeasurementGate]

    official_cnot_resources = load_saved_cost(dsurface=d, op_key="cnot")["serial"]
    # Correct for optimizations/quirks of Cultiv
    # There are 4 + 2dsurface syndrome extractions just from idling
    official_cnot_resources[cirq.CZ] -= (4 + 2 * d) * s_count[cirq.CZ]
    # Removing contributions from idling
    official_cnot_resources[cirq.MeasurementGate] -= (4 + 2 * d) * s_count[cirq.MeasurementGate]

    # Optimization from measuring the whole ancilla patch in the second Split
    official_cnot_resources[cirq.MeasurementGate] += 2 * d**2 - 1
    official_cnot_resources[cirq.MeasurementGate]

    Estimator = est.ResourceEstimator(
        arc=arch.DefaultLattice(d=d, idling=False, post_op_correction=True)
    )
    cnot_qubits = [cirq.GridQubit(0, 1), cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)]
    low_level_circuit = cirq.Circuit(
        [
            lsp.Merge(2, smooth=True).on(cnot_qubits[0], cnot_qubits[1]),
            lsp.Split([1, 1], smooth=True).on(cnot_qubits[0], cnot_qubits[1]),
            lsp.Merge(2, smooth=False).on(cnot_qubits[1], cnot_qubits[2]),
            lsp.Split([1, 1], smooth=False).on(cnot_qubits[1], cnot_qubits[2]),
        ]
    )
    circuit_cost = Counter(Estimator.serial_circuit_cost(low_level_circuit))
    assert circuit_cost[cirq.CZ] == official_cnot_resources[cirq.CZ]


def test_movement_moment_costs(movement_architecture):
    # Test that all primitives have moment costs
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)

    # Check Move
    op1, op2 = lsp.Move(zone="interact").on_each(qubit_a, qubit_b)
    moment_cost = movement_architecture.moment_cost(op1)
    op_time = movement_architecture.op_time(op1)
    assert moment_cost == {cirq.QubitPermutationGate: 1}
    assert op_time == 500
    moment_cost = movement_architecture.moment_cost(op2)
    assert moment_cost == {cirq.QubitPermutationGate: 1}
    assert op_time == 500

    # Check S Gate
    op = cirq.S.on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {
        cirq.MeasurementGate: 1,
        cirq.CZ: 5,
        cirq.PhasedXZGate: 3,
        cirq.QubitPermutationGate: 12,
        cirq.ResetChannel: 1,
    }

    # Check Sydrome Extraction
    op = lsp.SyndromeExtract(1, rounds=1).on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {
        cirq.MeasurementGate: 1,
        cirq.CZ: 4,
        cirq.PhasedXZGate: 2,
        cirq.QubitPermutationGate: 10,
        cirq.ResetChannel: 1,
    }

    # Check CNOT
    op = cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    cost = movement_architecture.moment_cost(op)
    assert cost == {
        cirq.CZ: 1,
        cirq.PhasedXZGate: 2,
    }

    # Check H Gate
    op = cirq.H.on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {cirq.PhasedXZGate: 1, cirq.QubitPermutationGate: 1}

    op = cirq.X.on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {}

    op = cirq.Z.on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {}

    op = cirq.ResetChannel().on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {cirq.ResetChannel: 1}

    op = cirq.MeasurementGate(1).on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {cirq.MeasurementGate: 1}

    op = cirq.I.on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {}

    op = lsp.ErrorCorrect(1).on(cirq.GridQubit(0, 0))
    cost = movement_architecture.moment_cost(op)
    assert cost == {}

    op = cirq.Rx(rads=7).on(cirq.GridQubit(0, 0))
    with pytest.raises(ValueError, match="Gate not recognized"):
        _ = movement_architecture.moment_cost(op)


def test_lattice_moment_costs(lattice_architecture):
    # Test that all primitives have correct moment costs
    op = lsp.Cultivate(pi / 4).on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op=op)
    pass

    op = cirq.H.on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {
        cirq.CZ: 2 * (4 * lattice_architecture.patch.d),
        cirq.PhasedXZGate: 2 * (2 * lattice_architecture.patch.d) + 1,
        cirq.MeasurementGate: 2 * (lattice_architecture.patch.d) + 2,
        cirq.ResetChannel: 2 * lattice_architecture.patch.d + 2,
    }

    op = lsp.SyndromeExtract(1, lattice_architecture.patch.d).on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {
        cirq.CZ: 4 * lattice_architecture.patch.d,
        cirq.PhasedXZGate: 2 * lattice_architecture.patch.d,
        cirq.MeasurementGate: lattice_architecture.patch.d,
        cirq.ResetChannel: lattice_architecture.patch.d,
    }

    op = lsp.Merge(2).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {
        cirq.CZ: 4 * lattice_architecture.patch.d,
        cirq.PhasedXZGate: 2 * lattice_architecture.patch.d,
        cirq.MeasurementGate: lattice_architecture.patch.d,
        cirq.ResetChannel: lattice_architecture.patch.d,
    }

    op = cirq.X.on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {}

    op = cirq.Z.on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {}

    op = cirq.ResetChannel().on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {cirq.ResetChannel: 1}

    op = cirq.MeasurementGate(1).on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {cirq.MeasurementGate: 1}

    op = cirq.I.on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {}

    op = lsp.Split([1, 1]).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {}

    op = lsp.ErrorCorrect(1).on(cirq.GridQubit(0, 0))
    cost = lattice_architecture.moment_cost(op)
    assert cost == {}

    with pytest.raises(ValueError, match="Gate not recognized"):
        _ = lattice_architecture.gate_cost(cirq.Rx(rads=7).on(cirq.GridQubit(0, 0)))


def test_timing(movement_architecture, lattice_architecture):
    # This test should break first when we introduce real gate times
    gates_with_time = [
        (cirq.PhasedXZGate, 5.0),
        (cirq.CZ, 0.27),
        (cirq.MeasurementGate, 1000.0),
        (cirq.I, 0.0),
        (cirq.QubitPermutationGate, 500.0),
        (cirq.ResetChannel, 400.0),
    ]
    for gate, time in gates_with_time:
        if gate in movement_architecture.phys_gate_times:
            assert movement_architecture.phys_gate_times.get(gate) == time
        if gate in lattice_architecture.phys_gate_times:
            assert lattice_architecture.phys_gate_times.get(gate) == time

    # Confirm expected errors
    with pytest.raises(ValueError, match="Gate not recognized"):
        movement_architecture.op_time(lsp.Merge(2).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    with pytest.raises(ValueError, match="Gate not recognized"):
        lattice_architecture.op_time(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))


def test_classmethods():
    movement_input_dict = {
        "movement": True,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
    }
    mv_arc = arch.Architecture.from_dict(movement_input_dict)
    assert isinstance(mv_arc, arch.DefaultMovement)

    lattice_input_dict = {
        "movement": False,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
    }
    ls_arch = arch.Architecture.from_dict(lattice_input_dict)
    assert isinstance(ls_arch, arch.DefaultLattice)

    movement_input_dict = {
        "movement": True,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
        "gate_times": {cirq.QubitPermutationGate: 100},
    }
    mv_arc = arch.Architecture.from_dict(movement_input_dict)
    assert mv_arc.phys_gate_times[cirq.QubitPermutationGate] == 100

    movement_input_dict = {
        "movement": True,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
        "gate_times": {"QubitPermutationGate": 99},
    }
    mv_arc = arch.Architecture.from_dict(movement_input_dict)
    assert mv_arc.phys_gate_times[cirq.QubitPermutationGate] == 99

    lattice_input_dict = {
        "movement": False,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
        "gate_times": {cirq.CZ: 100},
    }
    ls_arc = arch.Architecture.from_dict(lattice_input_dict)
    assert ls_arc.phys_gate_times[cirq.CZ] == 100

    lattice_input_dict = {
        "movement": False,
        "idling": True,
        "post_op_correction": True,
        "d": 7,
        "cultivation_repetition": 1,
        "cultivation_fault_distance": 3,
        "syndrome_rounds": 1,
        "gate_times": {"CZ": 99},
    }
    ls_arc = arch.Architecture.from_dict(lattice_input_dict)
    assert ls_arc.phys_gate_times[cirq.CZ] == 99

    ls_arc = arch.Architecture.from_json("data/lattice_test.json")
    assert ls_arc.phys_gate_times[cirq.CZ] == 99

    mv_arc = arch.Architecture.from_json("data/movement_test.json")
    assert mv_arc.phys_gate_times[cirq.CZ] == 99

    with pytest.raises(ValueError, match="Gate time"):
        input_dict = {
            "movement": False,
            "idling": True,
            "post_op_correction": True,
            "d": 7,
            "cultivation_repetition": 1,
            "syndrome_rounds": 1,
            "gate_times": {"CNOT": 99},
        }
        _ = arch.Architecture.from_dict(input_dict)


def test_dual_species_with_movement():
    # HM never pays for Measurement
    # HM often pays to move for CZ
    # - Transversal CNOT
    # - Fold transversal S
    # - Folded Cultivation
    # The exception is Gidney Cultivation
    # This has all nearest-neighbor CZs, so no moves
    d = 7
    hm = arch.DualSpeciesMovement(
        d=d, syndrome_rounds=1, cultivation_repetition=1, idling=False, post_op_correction=True
    )
    mv = arch.DefaultMovement(
        d=d, syndrome_rounds=1, cultivation_repetition=1, idling=False, post_op_correction=True
    )
    ls = arch.DefaultLattice(
        d=d,
        syndrome_rounds=1,
        cultivation_repetition=1,
        idling=False,
        post_op_correction=True,
    )

    assert hm._cultivate_t_cost == ls._cultivate_t_cost
    assert hm._cnot_cost == mv._cnot_cost
    assert hm._measure_cost == ls._measure_cost
    op = lsp.SyndromeExtract(1, rounds=1).on(cirq.GridQubit(0, 0))
    assert (
        hm.syndrome_extract_cost(op)["moment_cost"] == ls.syndrome_extract_cost(op)["moment_cost"]
    )
    assert hm.syndrome_extract_cost(op)["gate_cost"] == ls.syndrome_extract_cost(op)["gate_cost"]

    hm_folded = arch.DualSpeciesMovement(
        d=d, cultivation_repetition=1, idling=False, post_op_correction=True, fold_cultiv=True
    )
    mv_folded = arch.DefaultMovement(
        d=d, cultivation_repetition=1, idling=False, post_op_correction=True, fold_cultiv=True
    )
    assert hm_folded._cnot_cost == mv_folded._cnot_cost
    assert hm_folded._measure_cost == ls._measure_cost

    # Check that partial penalty is consistent with expection of paying one move per CZ
    folded_with_full_penalty = mv_folded._cultivate_t_cost["moment_cost"]
    folded_with_partial_penalty = hm_folded._cultivate_t_cost["moment_cost"]
    assert (
        folded_with_partial_penalty[cirq.QubitPermutationGate] == folded_with_full_penalty[cirq.CZ]
    )
    # Check that the rest of the moments are the same
    del folded_with_full_penalty[cirq.QubitPermutationGate]
    del folded_with_partial_penalty[cirq.QubitPermutationGate]
    assert folded_with_partial_penalty == folded_with_full_penalty


@pytest.mark.parametrize("fold", (True, False))
def test_mzo(fold):
    # MZO always pays two Moves per measure
    # - Syndrome Extract
    # - Cultiving (folded or unfolded)
    # MZO normally does not pay the cost to move for CZ
    # The exception is folded cultivation, where it pays 1 Move per CZ
    # Transversal CNOT costs the same for MZO, SSM, and DSM

    mzo = arch.MeasureZonesOnly(d=7, fold_cultiv=fold)
    ssm = arch.DefaultMovement(d=7, fold_cultiv=fold)
    dsnm = arch.DefaultLattice(d=7, syndrome_rounds=1)

    # Part 1: Syndrome Extraction
    se_op = lsp.SyndromeExtract(num_qubits=1, rounds=1).on(cirq.GridQubit(0, 0))
    mzo_se = mzo.syndrome_extract_cost(se_op)["moment_cost"]
    ssm_se = ssm.syndrome_extract_cost(se_op)["moment_cost"]
    dsnm_se = dsnm.syndrome_extract_cost(se_op)["moment_cost"]

    # First check that moves are correct two times over
    mzo_se_moves = mzo_se[cirq.QubitPermutationGate]
    ssm_se_moves = ssm_se[cirq.QubitPermutationGate]
    dsnm_se_measures = dsnm_se[cirq.MeasurementGate]
    move_diff = ssm_se_moves - mzo_se_moves
    # Confirm that the difference is explained by the lack of CZ penalties
    assert move_diff == 2 * ssm_se[cirq.CZ]
    # Confirm that the difference is all attributable to Measurement penalties
    assert mzo_se_moves == 2 * dsnm_se_measures

    # As an extra check confirm that all other values are the same
    del mzo_se[cirq.QubitPermutationGate]
    del ssm_se[cirq.QubitPermutationGate]
    assert mzo_se == ssm_se
    assert mzo_se == dsnm_se

    # Part 2: Transversal CNOT
    assert mzo._cnot_cost == ssm._cnot_cost

    # Part 3: T state cultivation
    mzo_t_cult = mzo._cultivate_t_cost["moment_cost"]
    ssm_t_cult = ssm._cultivate_t_cost["moment_cost"]

    mzo_cult_moves = mzo_t_cult[cirq.QubitPermutationGate]
    ssm_cult_moves = ssm_t_cult[cirq.QubitPermutationGate]
    move_diff = ssm_cult_moves - mzo_cult_moves
    cz_movement_penalty = 1 if fold else 0

    # Confirm that movements are attributable to the CZ movement penalty
    assert move_diff == (2 - cz_movement_penalty) * ssm_t_cult[cirq.CZ]

    del mzo_t_cult[cirq.QubitPermutationGate]
    del ssm_t_cult[cirq.QubitPermutationGate]
    assert mzo_t_cult == ssm_t_cult


def test_string_representations():
    ssm = arch.DefaultMovement(
        idling=True, post_op_correction=True, d=9, cultivation_repetition=10, syndrome_rounds=1
    )
    assert str(ssm) == "SingleSpeciesMovement(d=9, cr=10, fd=3, sr=1, fold=False)"

    dsnm = arch.DefaultLattice(
        idling=True, post_op_correction=True, d=9, cultivation_repetition=10, syndrome_rounds=9
    )
    assert str(dsnm) == "DualSpeciesNoMovement(d=9, cr=10, fd=3, sr=9)"

    dsm = arch.DualSpeciesMovement(
        idling=True, post_op_correction=True, d=11, cultivation_repetition=17, syndrome_rounds=1
    )
    assert str(dsm) == "DualSpeciesMovement(d=11, cr=17, fd=3, sr=1, fold=False)"

    mzo = arch.MeasureZonesOnly(
        idling=True, post_op_correction=True, d=19, cultivation_repetition=7, syndrome_rounds=5
    )
    assert str(mzo) == "ReadoutZonesOnly(d=19, cr=7, fd=3, sr=5, fold=False)"

    sc = arch.Superconductor(
        idling=True, post_op_correction=True, d=13, cultivation_repetition=99, syndrome_rounds=14
    )
    assert str(sc) == "Superconductor(d=13, cr=99, fd=3, sr=14)"

    ssm_fold = arch.DefaultMovement(fold_cultiv=True)
    assert str(ssm_fold) == "SingleSpeciesMovement(d=7, cr=1, fd=3, sr=1, fold=True)"

    dsnm = arch.DefaultLattice()
    assert str(dsnm) == "DualSpeciesNoMovement(d=7, cr=1, fd=3)"


def test_folded_architecture():
    folded_movement = arch.DefaultMovement(fold_cultiv=True)
    normal_movement = arch.DefaultMovement(fold_cultiv=False)

    folded_cultivation_time = folded_movement._cultivate_t_cost["op_time"]
    normal_cultivation_time = normal_movement._cultivate_t_cost["op_time"]

    assert folded_cultivation_time < normal_cultivation_time


def test_convert_globals_to_phasedxz():
    """
    Confirm that the conversion function works as expected
    """
    sc = arch.Superconductor()
    example1 = {
        "gate_cost": {ParallelRGate: 2, cirq.Rz: 3},
        "moment_cost": {
            ParallelRGate: 13,
        },
    }
    expected = {"gate_cost": {cirq.PhasedXZGate: 3}, "moment_cost": {}, "op_time": 0.0}
    actual = arch.convert_globals_to_phasedxz(architecture=sc, cost_with_globals=example1)
    assert expected == actual

    example2 = {
        "gate_cost": {cirq.MeasurementGate: 5},
        "moment_cost": {cirq.Rz: 5, ParallelRGate: 9},
    }
    expected = {
        "gate_cost": {cirq.MeasurementGate: 5},
        "moment_cost": {cirq.PhasedXZGate: 5},
        "op_time": 0.02 * 5,
    }
    actual = arch.convert_globals_to_phasedxz(architecture=sc, cost_with_globals=example2)
    assert expected == actual


def test_logical_move():
    arc = arch.DualSpeciesMovement()
    one_hop = lsp.Move(zone=None).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))
    two_hop = lsp.Move(zone=None).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1))

    one_cost = arc.op_time(one_hop)
    two_cost = arc.op_time(two_hop)

    assert two_cost == 2 * one_cost


def test_y_cult_on_movement():
    ssm = arch.DefaultMovement(d=11)
    mzo = arch.MeasureZonesOnly(d=11)
    dsm = arch.DualSpeciesMovement(d=11)
    cost1 = ssm.cultivate_cost(lsp.Cultivate(np.pi / 2).on(cirq.GridQubit(0, 0)))
    cost2 = mzo.cultivate_cost(lsp.Cultivate(np.pi / 2).on(cirq.GridQubit(0, 0)))
    cost3 = dsm.cultivate_cost(lsp.Cultivate(np.pi / 2).on(cirq.GridQubit(0, 0)))
    assert cost3["op_time"] < cost2["op_time"] < cost1["op_time"]
