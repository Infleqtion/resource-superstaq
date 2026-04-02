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
from math import pi
import cirq
import pytest
import resource_estimation.architecture as arch
import resource_estimation.estimate as est
import resource_estimation.lattice_surgery_primitives as lsp
from numpy import isclose


@pytest.fixture
def lattice_estimator():
    return est.ResourceEstimator(
        arc=arch.DefaultLattice(
            d=5,
            idling=True,
            post_op_correction=1,
            cultivation_repetition=1,
            syndrome_rounds=None,
        )
    )


@pytest.fixture
def movement_estimator():
    return est.ResourceEstimator(
        arc=arch.DefaultMovement(
            d=5,
            idling=True,
            post_op_correction=1,
            cultivation_repetition=1,
            syndrome_rounds=None,
        )
    )


@pytest.mark.parametrize(
    "estimator",
    [
        est.ResourceEstimator(
            arc=arch.DefaultMovement(
                d=5,
                idling=True,
                post_op_correction=1,
                cultivation_repetition=1,
                syndrome_rounds=None,
            )
        ),
        est.ResourceEstimator(
            arc=arch.DefaultLattice(
                d=5,
                idling=True,
                post_op_correction=1,
                cultivation_repetition=1,
                syndrome_rounds=None,
            )
        ),
    ],
)
def test_all_primitives(estimator):
    dummy_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    circuit = cirq.Circuit()
    circuit += [cirq.I.on(q) for q in dummy_qubits]
    circuit += [cirq.Z.on(q) for q in dummy_qubits]
    circuit += [cirq.X.on(q) for q in dummy_qubits]
    circuit += [cirq.H.on(q) for q in dummy_qubits]
    circuit += [cirq.MeasurementGate(9, key="terminal").on(*dummy_qubits)]
    circuit += [lsp.SyndromeExtract(1, 1).on(q) for q in dummy_qubits]
    circuit += [lsp.ErrorCorrect(1).on(q) for q in dummy_qubits]
    arc = estimator.arc
    if arc.movement:
        circuit += [cirq.CNOT.on(dummy_qubits[i], dummy_qubits[i + 1]) for i in range(8)]
        circuit += [cirq.S.on(q) for q in dummy_qubits]
    else:
        circuit += [
            lsp.Merge(2, smooth=True).on(*dummy_qubits[:2]),
            lsp.Split([1, 1], smooth=True).on(*dummy_qubits[:2]),
            lsp.Merge(2, smooth=False).on(*dummy_qubits[1:3]),
            lsp.Split([1, 1], smooth=False).on(*dummy_qubits[1:3]),
        ]
    circuit += [lsp.Cultivate(pi / 4).on(q) for q in dummy_qubits]

    # At least verify that there is no randomness in these estimates
    # Still TODO: Make this test better
    with pytest.warns(UserWarning, match="Returning result for d=7"):
        c1 = estimator.serial_circuit_cost(circuit)
        t1 = estimator.serial_circuit_time(circuit)
        c2 = estimator.serial_circuit_cost(circuit)
        t2 = estimator.serial_circuit_time(circuit)
    for key in c1.keys():
        assert c1[key] == c2[key]
    assert isclose(t1, t2, atol=0.00001)


def test_parallel_circuit_cost(lattice_estimator, movement_estimator):
    # TODO: This test could (should?) be considerably more thorough than the coverage requirement would imply
    qubit_a, qubit_b, qubit_c, qubit_d = (
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
    )
    # Check that one round of Syndrome Extraction is less than one Merge
    circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on(qubit_a),
        lsp.Merge(2, smooth=True).on(qubit_b, qubit_c),
    )
    estimated_moment_cost = lattice_estimator.parallel_circuit_cost(circuit=circuit)
    expected_moment_cost = lattice_estimator.arc.moment_cost(lsp.Merge(2).on(qubit_b, qubit_c))
    assert estimated_moment_cost == expected_moment_cost

    # Check that d rounds of Syndrome Extraction is equal to one Merge
    circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, lattice_estimator.arc.d).on(qubit_a),
        lsp.Merge(2, smooth=True).on(qubit_b, qubit_c),
    )
    estimated_moment_cost = lattice_estimator.parallel_circuit_cost(circuit=circuit)
    expected_moment_cost = lattice_estimator.arc.moment_cost(
        lsp.SyndromeExtract(1, lattice_estimator.arc.d).on(qubit_a)
    )
    assert estimated_moment_cost == expected_moment_cost

    # Test parallel CNOT gates get counted as parallel
    circuit = cirq.Circuit(cirq.CNOT.on(qubit_a, qubit_b), cirq.CNOT.on(qubit_c, qubit_d))
    estimated_moment_cost = movement_estimator.parallel_circuit_cost(circuit=circuit)
    expected_moment_cost = movement_estimator.arc.moment_cost(cirq.CNOT.on(qubit_a, qubit_b))
    assert estimated_moment_cost == expected_moment_cost

    estimated_moment_cost = movement_estimator.parallel_circuit_cost(circuit=circuit, pretty=True)
    assert estimated_moment_cost == {
        "CZ": 1,
        "PhasedXZGate": 2,
    }


def test_self_returns(movement_estimator, lattice_estimator):
    # TODO: There are no self-returns anymore so this function is not well named
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [lsp.ErrorCorrect(2).on(qubit_a, qubit_b), cirq.ResetChannel().on(qubit_a)]
    )
    cost = movement_estimator.serial_circuit_cost(circuit=circuit, pretty=True)
    assert cost == {
        "ResetChannel": 49,
    }

    circuit = cirq.Circuit(
        [
            lsp.ErrorCorrect(1).on_each(qubit_a, qubit_b),
            cirq.ResetChannel().on_each(qubit_a, qubit_b),
        ]
    )
    cost = lattice_estimator.serial_circuit_cost(circuit=circuit, pretty=True)
    assert cost == {
        "ResetChannel": 2 * 49,
    }


def test_error_handling(lattice_estimator, movement_estimator):
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    # Check Bad Lattice Surgery Circuit
    bad_circuit = cirq.Circuit([lsp.Cultivate(pi / 2).on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b)])
    with pytest.raises(ValueError, match="incompatible"):
        _ = lattice_estimator.serial_circuit_cost(bad_circuit)

    # Check Bad Movement Circuit
    bad_circuit = cirq.Circuit(
        [
            cirq.S.on(qubit_a),
            cirq.Rx(rads=1 / 3).on(qubit_b),
            cirq.CNOT.on(qubit_a, qubit_b),
        ]
    )
    with pytest.raises(ValueError, match="incompatible"):
        _ = movement_estimator.serial_circuit_cost(bad_circuit)


# TODO: Might be worth having one or two more example tests for the critical path algorithm
def test_critical_path():
    q0, q1 = cirq.LineQubit.range(2)
    c1 = cirq.Circuit()
    c1 += cirq.S.on(q0)
    c1 += cirq.CNOT.on(q0, q1)
    c1 += cirq.S.on(q1)
    c2 = cirq.Circuit()
    c2 += cirq.S.on(q0)
    c2 += cirq.S.on(q0)
    c2 += cirq.CNOT.on(q0, q1)
    arc = arch.DefaultMovement()
    estim = est.ResourceEstimator(arc)
    # Should be identical aside from floating point errors
    assert isclose(estim.serial_circuit_time(c1), estim.serial_circuit_time(c2), atol=1e-5)

    qa, qb = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [
            cirq.S.on(qa),
            cirq.H.on(qa),
            cirq.S.on(qa),
            cirq.H.on(qa),
            cirq.Z.on(qa),
            cirq.S.on(qa),
            cirq.Z.on(qb),
            cirq.CNOT.on(qa, qb),
            cirq.Z.on(qa),
            cirq.S.on(qa),
            cirq.S.on(qb),
            cirq.H.on(qb),
            cirq.H.on(qb),
        ]
    )
    with pytest.warns(UserWarning, match="very expensive"):
        cp = estim.critical_path(circuit)
    expected = [
        cirq.S(cirq.GridQubit(0, 0)),
        cirq.H(cirq.GridQubit(0, 0)),
        cirq.S(cirq.GridQubit(0, 0)),
        cirq.H(cirq.GridQubit(0, 0)),
        cirq.Z(cirq.GridQubit(0, 0)),
        cirq.S(cirq.GridQubit(0, 0)),
        cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        cirq.S(cirq.GridQubit(0, 1)),
        cirq.H(cirq.GridQubit(0, 1)),
        cirq.H(cirq.GridQubit(0, 1)),
    ]
    assert cp == expected
    assert estim.parallel_circuit_time(circuit=circuit) == estim.parallel_circuit_time(
        circuit=cirq.Circuit(expected)
    )


def test_physical_qubit_count(lattice_estimator):
    test_circuit = cirq.Circuit(
        [
            cirq.I.on(cirq.GridQubit(0, 0)),
            lsp.SyndromeExtract(1, rounds=7).on(cirq.GridQubit(1, 0)),
        ]
    )
    expected_num_physical_qubits = 98  # 2 * (2 * d**2 - 1)
    num_physical_qubits = lattice_estimator.physical_qubits(test_circuit)
    assert num_physical_qubits == expected_num_physical_qubits
