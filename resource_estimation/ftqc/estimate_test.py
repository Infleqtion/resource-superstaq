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
from __future__ import annotations

import typing
from math import pi
from unittest import mock

import cirq
import networkx as nx
import numpy as np
import pytest
from numpy import isclose

import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.estimate as est
import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc import ResourceEstimator, ccz_8_to_1, distil_15_to_1
from resource_estimation.typing import GateKey


@pytest.fixture
def lattice_estimator() -> est.ResourceEstimator:
    return est.ResourceEstimator(
        arc=arch.DefaultLattice(
            d=5,
            idling=True,
            post_op_correction=True,
            cultivation_repetition=1,
            syndrome_rounds=None,
        ),
    )


@pytest.fixture
def movement_estimator() -> est.ResourceEstimator:
    return est.ResourceEstimator(
        arc=arch.DefaultMovement(
            d=5,
            idling=True,
            post_op_correction=True,
            cultivation_repetition=1,
            distillation_repetition=1,
            syndrome_rounds=None,
        ),
    )


@pytest.mark.parametrize(
    "estimator",
    [
        est.ResourceEstimator(
            arc=arch.DefaultMovement(
                d=5,
                idling=True,
                post_op_correction=True,
                cultivation_repetition=1,
                distillation_repetition=1,
                syndrome_rounds=None,
            ),
        ),
        est.ResourceEstimator(
            arc=arch.DefaultLattice(
                d=5,
                idling=True,
                post_op_correction=True,
                cultivation_repetition=1,
                syndrome_rounds=None,
            ),
        ),
    ],
)
def test_all_primitives(estimator: est.ResourceEstimator) -> None:
    dummy_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    factory_block = [cirq.GridQubit(4, i) for i in range(31)]
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
        circuit += [lsp.Distil("T").on(*factory_block)]
        circuit += [lsp.Distil("CCZ").on(*factory_block[:23])]
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
    assert c1 == c2
    assert np.isclose(t1, t2, atol=0.00001)


def test_parallel_circuit_cost(
    lattice_estimator: est.ResourceEstimator, movement_estimator: est.ResourceEstimator
) -> None:
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
        lsp.SyndromeExtract(1, lattice_estimator.arc.d).on(qubit_a),
    )
    assert estimated_moment_cost == expected_moment_cost

    # Test parallel CNOT gates get counted as parallel
    circuit = cirq.Circuit(cirq.CNOT.on(qubit_a, qubit_b), cirq.CNOT.on(qubit_c, qubit_d))
    estimated_moment_cost = movement_estimator.parallel_circuit_cost(circuit=circuit)
    expected_moment_cost = movement_estimator.arc.moment_cost(cirq.CNOT.on(qubit_a, qubit_b))
    assert estimated_moment_cost == expected_moment_cost

    estimated_moment_cost = movement_estimator.parallel_circuit_cost(circuit=circuit)
    assert estimated_moment_cost == {
        cirq.CZ: 1,
        cirq.PhasedXZGate: 2,
    }


def test_self_returns(
    movement_estimator: est.ResourceEstimator, lattice_estimator: est.ResourceEstimator
) -> None:
    # TODO: There are no self-returns anymore so this function is not well named
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [lsp.ErrorCorrect(2).on(qubit_a, qubit_b), cirq.ResetChannel().on(qubit_a)],
    )
    cost = movement_estimator.serial_circuit_cost(circuit=circuit)
    assert cost == {
        cirq.ResetChannel: 49,
    }

    circuit = cirq.Circuit(
        [
            lsp.ErrorCorrect(1).on_each(qubit_a, qubit_b),
            cirq.ResetChannel().on_each(qubit_a, qubit_b),
        ],
    )
    cost = lattice_estimator.serial_circuit_cost(circuit=circuit)
    assert cost == {
        cirq.ResetChannel: 2 * 49,
    }


def test_error_handling(
    lattice_estimator: est.ResourceEstimator, movement_estimator: est.ResourceEstimator
) -> None:
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
        ],
    )
    with pytest.raises(ValueError, match="incompatible"):
        _ = movement_estimator.serial_circuit_cost(bad_circuit)


# TODO: Might be worth having one or two more example tests for the critical path algorithm
def test_critical_path() -> None:
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
    assert np.isclose(estim.serial_circuit_time(c1), estim.serial_circuit_time(c2), atol=1e-5)

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
        ],
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
        circuit=cirq.Circuit(expected),
    )

    # Test that critical path for distillation circuits are as expected
    # Critical paths are currently the same for both distillation circuits
    t_15_to_1 = distil_15_to_1()
    ccz_distilled = ccz_8_to_1()
    expected_types: list[GateKey] = [lsp.Cultivate, cirq.CNOT, cirq.S, cirq.H, cirq.MeasurementGate]
    with pytest.warns(UserWarning, match="very expensive"):
        path1 = estim.critical_path(t_15_to_1)
        path2 = estim.critical_path(ccz_distilled)
        assert all(op in cirq.GateFamily(expected) for op, expected in zip(path1, expected_types))
        assert all(op in cirq.GateFamily(expected) for op, expected in zip(path2, expected_types))


@mock.patch("resource_estimation.ftqc.architecture.randint")
def test_dynamic_T_resource_counts(mock_randint: mock.MagicMock) -> None:
    arc = arch.DefaultMovement()
    qubit = cirq.GridQubit(0, 0)
    static_correction = cirq.S.on(qubit)
    dynamic_correction = lsp.ResourceCorrection("T").on(qubit)
    mock_randint.side_effect = [1, 0, 1, 0]
    static_s_gate_cost = arc.gate_cost(static_correction)
    assert arc.gate_cost(dynamic_correction) == {}
    assert arc.gate_cost(static_correction) == static_s_gate_cost

    assert arc.gate_cost(dynamic_correction) == static_s_gate_cost
    assert arc.gate_cost(static_correction) == static_s_gate_cost

    static_correction_time = arc.op_time(static_correction)
    assert arc.op_time(dynamic_correction) == 0.0
    assert arc.op_time(static_correction) == static_correction_time

    assert arc.op_time(dynamic_correction) == static_correction_time
    assert arc.op_time(static_correction) == static_correction_time


@mock.patch("resource_estimation.ftqc.architecture.randint")
def test_dynamic_CCZ_resource_counts(mock_randint: mock.MagicMock) -> None:
    arc = arch.DefaultMovement()
    qubit_a, qubit_b, qubit_c = (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
    correction_circuit = cirq.Circuit()
    # The circuit excludes syndrome extraction costs since those are handled during compilation, so
    # the CCZ correction cost should be identical to just doing these gates with no syndrome
    # extraction
    se_gate = lsp.SyndromeExtract(num_qubits=1, rounds=arc.rounds)
    correction_circuit.append(cirq.H.on_each(*(qubit_a, qubit_b, qubit_c)))
    correction_circuit.append(se_gate.on_each(*(qubit_a, qubit_b, qubit_c)))
    correction_circuit.append(cirq.X.on_each(*(qubit_a, qubit_b, qubit_c)))
    correction_circuit.append(cirq.CNOT.on(qubit_a, qubit_b))
    correction_circuit.append(se_gate.on_each(*(qubit_a, qubit_b)))
    correction_circuit.append(cirq.CNOT.on(qubit_a, qubit_c))
    correction_circuit.append(se_gate.on_each(*(qubit_a, qubit_c)))
    correction_circuit.append(cirq.CNOT.on(qubit_b, qubit_c))
    correction_circuit.append(se_gate.on_each(*(qubit_b, qubit_c)))
    correction_circuit.append(cirq.H.on_each(*(qubit_a, qubit_b, qubit_c)))
    dynamic_op: cirq.Operation = lsp.ResourceCorrection("CCZ").on(qubit_a, qubit_b, qubit_c)
    mock_randint.side_effect = [1, 0, 1, 0, 1, 0]
    est = ResourceEstimator(arc)
    normal_correction_cost = est.serial_circuit_cost(correction_circuit)
    operation_dynamic_cost = arc.gate_cost(dynamic_op)
    assert {} == operation_dynamic_cost
    # Should be correction applied on second flip
    operation_dynamic_cost = arc.gate_cost(dynamic_op)
    assert normal_correction_cost == operation_dynamic_cost

    normal_correction_time = est.parallel_circuit_time(correction_circuit)
    operation_dynamic_time = arc.op_time(dynamic_op)
    assert 0.0 == operation_dynamic_time
    operation_dynamic_time = arc.op_time(dynamic_op)
    assert isclose(normal_correction_time, operation_dynamic_time)

    parallel_correction_cost = est.parallel_circuit_cost(correction_circuit)
    moment_dynamic_cost = arc.moment_cost(dynamic_op)
    assert {} == moment_dynamic_cost
    # Should be correction applied on second flip
    moment_dynamic_cost = arc.moment_cost(dynamic_op)
    assert parallel_correction_cost == moment_dynamic_cost


def test_physical_qubit_count(lattice_estimator: est.ResourceEstimator) -> None:
    test_circuit = cirq.Circuit(
        [
            cirq.I.on(cirq.GridQubit(0, 0)),
            lsp.SyndromeExtract(1, rounds=7).on(cirq.GridQubit(1, 0)),
        ],
    )
    expected_num_physical_qubits = 98  # 2 * (2 * d**2 - 1)
    num_physical_qubits = lattice_estimator.physical_qubits(test_circuit)
    assert num_physical_qubits == expected_num_physical_qubits


def local_pauli(pauli: cirq.Pauli, qubit_index: int = 0) -> cirq.PauliString[cirq.LineQubit]:
    return cirq.PauliString(pauli(cirq.LineQubit(qubit_index)))


def test_reaction_depth_empty_and_clifford_only_circuits_are_zero() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit()) == 0
    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.H(qubit))) == 0


@pytest.mark.parametrize("factory_op", [cirq.T, cirq.S])
def test_reaction_depth_uses_default_single_vertex_factories(factory_op: cirq.Gate) -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()
    circuit = cirq.Circuit(factory_op(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert isinstance(reaction_tree, nx.DiGraph)
    assert reaction_tree.graph["operations"] == tuple(circuit.all_operations())
    assert set(reaction_tree.nodes) == {(0, 0)}
    assert reaction_tree.nodes[(0, 0)]["dependency_paulis"] == (cirq.PauliString(cirq.Z(qubit)),)
    assert reaction_tree.nodes[(0, 0)]["depth"] == 1
    assert reaction_depth_estimator.reaction_depth(circuit) == 1


def test_default_ccz_dynamics_schedule_expanded_toffolis() -> None:
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(1, 6)
    toffolis = (
        (q1, q2, q3),
        (q3, q4, q5),
        (q1, q2, q3),
        (q3, q4, q5),
    )
    operations = tuple(
        operation
        for control_a, control_b, target in toffolis
        for operation in (
            cirq.H(target),
            cirq.CCZ(control_a, control_b, target),
            cirq.H(target),
        )
    )
    circuit = cirq.Circuit(cirq.Moment(operation) for operation in operations)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    ccz_operation_indices = (1, 4, 7, 10)
    assert set(reaction_tree.nodes) == {
        (operation_index, vertex_index)
        for operation_index in ccz_operation_indices
        for vertex_index in range(3)
    }

    z_paulis: dict[cirq.Qid, cirq.PauliString[cirq.Qid]] = {
        qubit: cirq.PauliString(cirq.Z(qubit)) for qubit in (q1, q2, q3)
    }
    first_ccz_dynamics = (
        ((z_paulis[q1],), (z_paulis[q2] * z_paulis[q3],)),
        ((z_paulis[q2],), (z_paulis[q1] * z_paulis[q3],)),
        ((z_paulis[q3],), (z_paulis[q1] * z_paulis[q2],)),
    )
    for vertex_index, (dependency_paulis, outputs) in enumerate(first_ccz_dynamics):
        assert reaction_tree.nodes[(1, vertex_index)]["dependency_paulis"] == dependency_paulis
        assert reaction_tree.nodes[(1, vertex_index)]["outputs"] == outputs

    assert set(reaction_tree.edges) == {
        ((1, 0), (4, 0)),
        ((1, 1), (4, 0)),
        ((1, 0), (10, 0)),
        ((1, 1), (10, 0)),
        ((4, 1), (7, 2)),
        ((4, 2), (7, 2)),
        ((7, 0), (10, 0)),
        ((7, 1), (10, 0)),
    }
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_depth_uses_explicit_non_auto_corrected_t_factory() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={cirq.T: False},
    )

    reaction_tree = reaction_depth_estimator.reaction_tree(cirq.Circuit(cirq.T(qubit)))

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit))) == 1
    assert set(reaction_tree.nodes) == {(0, 0)}
    assert reaction_tree.nodes[(0, 0)]["outputs"] == (
        cirq.PauliString(cirq.X(qubit)),
        cirq.PauliString(cirq.Z(qubit)),
    )
    assert reaction_tree.nodes[(0, 0)]["dependency_paulis"] == (
        cirq.PauliString(cirq.X(qubit)),
        cirq.PauliString(cirq.Z(qubit)),
    )


@pytest.mark.parametrize("output_pauli", [cirq.X, cirq.Y, cirq.Z])
def test_non_auto_corrected_factory_always_creates_dependencies(
    output_pauli: cirq.Pauli,
) -> None:
    qubit = cirq.LineQubit(0)
    source_gate = cirq.XPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={source_gate: True, cirq.T: False},
        factory_reaction_dynamics={
            (source_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (local_pauli(output_pauli),),
                ),
            )
        },
    )
    circuit = cirq.Circuit(source_gate(qubit), cirq.T(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {((0, 0), (1, 0))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_non_auto_corrected_factory_ignores_disjoint_outputs() -> None:
    source_qubit, target_qubit = cirq.LineQubit.range(2)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={cirq.T: False},
    )
    circuit = cirq.Circuit(cirq.T(source_qubit), cirq.T(target_qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == set()
    assert reaction_depth_estimator.reaction_depth(circuit) == 1


def test_default_s_factory_is_auto_corrected() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()
    circuit = cirq.Circuit(cirq.T(qubit), cirq.S(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert reaction_depth_estimator.factories[cirq.S]
    assert set(reaction_tree.edges) == set()
    assert reaction_depth_estimator.reaction_depth(circuit) == 1


def test_reaction_depth_factory_dict_keys_define_factory_gates() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(factories={})

    with pytest.raises(
        ValueError, match="non-Clifford operation without factory reaction dynamics"
    ):
        reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit)))


@pytest.mark.parametrize("factories", [{cirq.S: False}, {cirq.TOFFOLI: True}])
def test_reaction_depth_rejects_undefined_factory_corrections(
    factories: dict[GateKey, bool],
) -> None:
    with pytest.raises(ValueError, match="No factory reaction dynamics are defined"):
        est.ReactionDepthEstimator(factories=factories)


def test_reaction_depth_rejects_nonlocal_dependency_pauli_at_construction() -> None:
    dependency_qubit = cirq.LineQubit(1)
    with pytest.raises(ValueError, match="Reaction Pauli .* must use only operation-local qubits"):
        est.ReactionDepthEstimator(
            factory_reaction_dynamics={
                (cirq.T, True): (
                    est.ReactionDynamics(
                        (cirq.PauliString(cirq.X(dependency_qubit)),),
                        (local_pauli(cirq.Z),),
                    ),
                )
            },
        )


def test_reaction_tree_does_not_add_edges_for_commuting_factory_outputs() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()
    circuit = cirq.Circuit(cirq.T(qubit), cirq.T(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.nodes) == {(0, 0), (1, 0)}
    assert set(reaction_tree.edges) == set()
    assert reaction_depth_estimator.reaction_depth(circuit) == 1


def test_reaction_tree_adds_edges_for_anticommuting_propagated_outputs() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()
    circuit = cirq.Circuit(cirq.T(qubit), cirq.H(qubit), cirq.T(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.nodes) == {(0, 0), (2, 0)}
    assert set(reaction_tree.edges) == {((0, 0), (2, 0))}
    assert reaction_tree.nodes[(0, 0)]["depth"] == 1
    assert reaction_tree.nodes[(2, 0)]["depth"] == 2
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_tree_adds_edges_for_anticommuting_y_outputs() -> None:
    qubit = cirq.LineQubit(0)
    source_gate = cirq.XPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={source_gate: True, cirq.T: True},
        factory_reaction_dynamics={
            (source_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (local_pauli(cirq.Y),),
                ),
            )
        },
    )
    circuit = cirq.Circuit(source_gate(qubit), cirq.T(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {((0, 0), (1, 0))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_tree_supports_multiple_vertices_per_factory_operation() -> None:
    qubit = cirq.LineQubit(0)
    custom_gate = cirq.XPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={custom_gate: True},
        factory_reaction_dynamics={
            (custom_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (local_pauli(cirq.X),),
                ),
                est.ReactionDynamics(
                    (local_pauli(cirq.X),),
                    (local_pauli(cirq.Z),),
                ),
            )
        },
    )
    circuit = cirq.Circuit(custom_gate(qubit), custom_gate(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.nodes) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(reaction_tree.edges) == {((0, 0), (1, 0)), ((0, 1), (1, 1))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_tree_omits_transitive_dependencies() -> None:
    qubit = cirq.LineQubit(0)
    custom_gate = cirq.XPowGate(exponent=0.25)
    sink_gate = cirq.YPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={custom_gate: True, sink_gate: True},
        factory_reaction_dynamics={
            (custom_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (local_pauli(cirq.X),),
                ),
            ),
            (sink_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.X),),
                    (),
                ),
            ),
        },
    )
    circuit = cirq.Circuit(
        custom_gate(qubit),
        custom_gate(qubit),
        custom_gate(qubit),
        sink_gate(qubit),
    )

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {((0, 0), (1, 0)), ((1, 0), (2, 0))}
    assert nx.descendants(reaction_tree, (0, 0)) == {(1, 0), (2, 0)}
    assert reaction_depth_estimator.reaction_depth(circuit) == 3


def test_reaction_tree_omits_covered_vertex_in_multi_vertex_factory() -> None:
    qubit = cirq.LineQubit(0)
    source_gate = cirq.XPowGate(exponent=0.25)
    target_gate = cirq.YPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={source_gate: True, target_gate: True},
        factory_reaction_dynamics={
            (source_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (local_pauli(cirq.X),),
                ),
            ),
            (target_gate, True): (
                est.ReactionDynamics((local_pauli(cirq.Z),), ()),
                est.ReactionDynamics((local_pauli(cirq.X),), ()),
            ),
        },
    )
    circuit = cirq.Circuit(source_gate(qubit), source_gate(qubit), target_gate(qubit))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {((0, 0), (1, 0)), ((1, 0), (2, 0))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 3


def test_reaction_tree_keeps_irreducible_dependencies() -> None:
    qubit = cirq.LineQubit(0)
    source_gate = cirq.XPowGate(exponent=0.25)
    sink_gate = cirq.YPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={source_gate: True, sink_gate: True},
        factory_reaction_dynamics={
            (source_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.X),),
                    (local_pauli(cirq.X),),
                ),
            ),
            (sink_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.Z),),
                    (),
                ),
            ),
        },
    )
    circuit = cirq.Circuit(
        source_gate(qubit),
        source_gate(qubit),
        sink_gate(qubit),
        sink_gate(qubit),
    )

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {
        ((0, 0), (2, 0)),
        ((0, 0), (3, 0)),
        ((1, 0), (2, 0)),
        ((1, 0), (3, 0)),
    }
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_tree_supports_multi_qubit_pauli_anticommutation() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    custom_gate = cirq.CZPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={custom_gate: True},
        factory_reaction_dynamics={
            (custom_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.X) * local_pauli(cirq.X, qubit_index=1),),
                    (local_pauli(cirq.Z) * local_pauli(cirq.X, qubit_index=1),),
                ),
            )
        },
    )
    circuit = cirq.Circuit(custom_gate(q0, q1), custom_gate(q0, q1))

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert set(reaction_tree.edges) == {((0, 0), (1, 0))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


def test_reaction_tree_localizes_dynamics_to_operation_qubits() -> None:
    first_qubit, second_qubit = cirq.NamedQubit.range(2, prefix="q")
    custom_gate = cirq.XPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={custom_gate: True},
        factory_reaction_dynamics={
            (custom_gate, True): (
                est.ReactionDynamics(
                    (local_pauli(cirq.X),),
                    (local_pauli(cirq.Y),),
                ),
            )
        },
    )
    circuit = cirq.Circuit(
        cirq.Moment(custom_gate(first_qubit)),
        cirq.Moment(custom_gate(second_qubit)),
        cirq.Moment(custom_gate(first_qubit)),
    )

    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert reaction_tree.nodes[(0, 0)]["outputs"] == (cirq.PauliString(cirq.Y(first_qubit)),)
    assert reaction_tree.nodes[(1, 0)]["outputs"] == (cirq.PauliString(cirq.Y(second_qubit)),)
    assert set(reaction_tree.edges) == {((0, 0), (2, 0))}
    assert reaction_depth_estimator.reaction_depth(circuit) == 2


@pytest.mark.parametrize("concrete_pauli_is_output", [False, True])
def test_reaction_depth_rejects_concrete_qubits_in_dynamics_at_construction(
    concrete_pauli_is_output: bool,
) -> None:
    operation_qubit = cirq.NamedQubit("operation")
    custom_gate = cirq.XPowGate(exponent=0.25)
    concrete_pauli: cirq.PauliString[cirq.Qid] = cirq.PauliString(cirq.X(operation_qubit))
    # invalid input is cast explicitly to catch error and satisfy type checker
    invalid_pauli = typing.cast(cirq.PauliString[cirq.LineQubit], concrete_pauli)
    with pytest.raises(ValueError, match="must use only operation-local qubits"):
        est.ReactionDepthEstimator(
            factories={custom_gate: True},
            factory_reaction_dynamics={
                (custom_gate, True): (
                    est.ReactionDynamics(
                        (local_pauli(cirq.X),) if concrete_pauli_is_output else (invalid_pauli,),
                        (invalid_pauli,) if concrete_pauli_is_output else (local_pauli(cirq.Z),),
                    ),
                )
            },
        )


def test_reaction_depth_rejects_non_factory_non_clifford() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    with pytest.raises(
        ValueError, match="non-Clifford operation without factory reaction dynamics"
    ):
        reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.TOFFOLI(q0, q1, q2)))


def test_reaction_tree_rejects_non_factory_non_clifford() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    with pytest.raises(
        ValueError, match="non-Clifford operation without factory reaction dynamics"
    ):
        reaction_depth_estimator.reaction_tree(cirq.Circuit(cirq.TOFFOLI(q0, q1, q2)))
