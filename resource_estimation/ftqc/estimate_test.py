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
import networkx as nx
import pytest
from numpy import isclose

import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.estimate as est
import resource_estimation.ftqc.lattice_surgery_primitives as lsp


@pytest.fixture
def lattice_estimator() -> est.ResourceEstimator:
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
def movement_estimator() -> est.ResourceEstimator:
    return est.ResourceEstimator(
        arc=arch.DefaultMovement(
            d=5,
            idling=True,
            post_op_correction=1,
            cultivation_repetition=1,
            distillation_repetition=1,
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
                distillation_repetition=1,
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
def test_all_primitives(estimator) -> None:
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
        circuit += [lsp.Distil().on(*factory_block)]
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


def test_parallel_circuit_cost(lattice_estimator, movement_estimator) -> None:
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


def test_self_returns(movement_estimator, lattice_estimator) -> None:
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


def test_error_handling(lattice_estimator, movement_estimator) -> None:
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


def test_physical_qubit_count(lattice_estimator) -> None:
    test_circuit = cirq.Circuit(
        [
            cirq.I.on(cirq.GridQubit(0, 0)),
            lsp.SyndromeExtract(1, rounds=7).on(cirq.GridQubit(1, 0)),
        ]
    )
    expected_num_physical_qubits = 98  # 2 * (2 * d**2 - 1)
    num_physical_qubits = lattice_estimator.physical_qubits(test_circuit)
    assert num_physical_qubits == expected_num_physical_qubits


def test_reaction_depth_uses_default_auto_corrected_t_factory() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit))) == {
        qubit: {"X": 0, "Z": 1}
    }


def test_reaction_depth_uses_default_s_factory() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.S(qubit))) == {
        qubit: {"X": 0, "Z": 1}
    }


def test_reaction_depth_uses_explicit_non_auto_corrected_t_factory() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={cirq.T: False},
    )

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit))) == {
        qubit: {"X": 1, "Z": 1}
    }


def test_reaction_depth_factory_dict_keys_define_factory_gates() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(factories={})

    with pytest.raises(ValueError, match="non-Clifford operation without a factory dynamic"):
        reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit)))


@pytest.mark.parametrize("factories", [{cirq.S: True}, {cirq.CCZ: True}])
def test_reaction_depth_rejects_undefined_factory_corrections(
    factories,
) -> None:
    with pytest.raises(ValueError, match="No reaction-depth factory dynamic is defined"):
        est.ReactionDepthEstimator(factories=factories)


def test_reaction_depth_rejects_wrong_arity_factory_dynamic() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        reaction_dynamics={(cirq.T, True): (est.ReactionDynamics(1, "X", 0, "Z", 1),)},
    )

    with pytest.raises(IndexError):
        reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit)))


def test_reaction_depth_uses_custom_factory_dynamics() -> None:
    qubit = cirq.LineQubit(0)
    custom_gate = cirq.ZPowGate(exponent=0.25)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={custom_gate: True},
        reaction_dynamics={(custom_gate, True): (est.ReactionDynamics(0, "X", 0, "Z", 2),)},
    )

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(custom_gate.on(qubit))) == {
        qubit: {"X": 0, "Z": 2}
    }


def test_reaction_depth_custom_dynamics_override_is_instance_local() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(
        reaction_dynamics={(cirq.T, True): (est.ReactionDynamics(0, "X", 0, "Z", 5),)},
    )

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit))) == {
        qubit: {"X": 0, "Z": 5}
    }
    assert est.ReactionDepthEstimator().reaction_depth(cirq.Circuit(cirq.T(qubit))) == {
        qubit: {"X": 0, "Z": 1}
    }


@pytest.mark.parametrize(
    "circuit",
    [
        cirq.Circuit(cirq.T(cirq.LineQubit(0)), cirq.H(cirq.LineQubit(0))),
        cirq.Circuit(
            cirq.T(cirq.LineQubit(0)),
            cirq.H(cirq.LineQubit(0)),
            cirq.S(cirq.LineQubit(0)),
        ),
        cirq.Circuit(
            cirq.T(cirq.LineQubit(0)),
            cirq.T(cirq.LineQubit(1)),
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
        ),
    ],
)
def test_reaction_tree_final_vertices_track_reaction_depth_qubits(
    circuit: cirq.Circuit,
) -> None:
    reaction_depth_estimator = est.ReactionDepthEstimator()
    reaction_depth = reaction_depth_estimator.reaction_depth(circuit)
    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)
    final_time = len(reaction_tree.graph["operations"])

    assert isinstance(reaction_tree, nx.DiGraph)
    assert reaction_tree.graph["operations"] == tuple(circuit.all_operations())
    for qubit in reaction_depth:
        assert ("X", qubit, final_time) in reaction_tree.nodes
        assert ("Z", qubit, final_time) in reaction_tree.nodes
        assert {
            basis: reaction_tree.nodes[(basis, qubit, final_time)]["depth"] for basis in ("X", "Z")
        } == reaction_depth[qubit]


def test_reaction_tree_adds_zero_weight_edges_for_unchanged_nodes() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.T(q0), cirq.T(q1))

    reaction_tree = est.ReactionDepthEstimator().reaction_tree(circuit)
    final_time = len(reaction_tree.graph["operations"])

    assert reaction_tree.number_of_nodes() == 2 * len(circuit.all_qubits()) * (
        len(tuple(circuit.all_operations())) + 1
    )
    assert all(
        (basis, qubit, final_time) in reaction_tree.nodes
        for qubit in (q0, q1)
        for basis in ("X", "Z")
    )
    assert {
        (("X", q0, 0), ("X", q0, 1), 0),
        (("X", q1, 0), ("X", q1, 1), 0),
        (("Z", q1, 0), ("Z", q1, 1), 0),
        (("Z", q0, 1), ("Z", q0, 2), 0),
        (("X", q1, 1), ("X", q1, 2), 0),
    }.issubset(
        (source, target, data["weight"]) for source, target, data in reaction_tree.edges(data=True)
    )


def test_reaction_tree_rejects_non_factory_non_clifford() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    with pytest.raises(ValueError, match="non-Clifford operation without a factory dynamic"):
        reaction_depth_estimator.reaction_tree(cirq.Circuit(cirq.CCZ(q0, q1, q2)))


def test_reaction_tree_tracks_pauli_product_factory_regression() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    pauli_product = cirq.PauliStringPhasor(
        cirq.PauliString({q0: cirq.Z, q1: cirq.Z}),
        exponent_neg=0.25,
        exponent_pos=-0.25,
    )
    pauli_product_dynamics = (
        est.ReactionDynamics(0, "Z", 0, "Z", 0),
        est.ReactionDynamics(0, "X", 0, "Z", 1),
        est.ReactionDynamics(1, "Z", 1, "Z", 0),
        est.ReactionDynamics(1, "X", 1, "Z", 1),
    )
    circuit = cirq.Circuit(
        cirq.Moment([pauli_product]),
        cirq.Moment([cirq.H(q0)]),
        cirq.Moment([cirq.T(q0)]),
        cirq.Moment([cirq.T(q1)]),
        cirq.Moment([cirq.H(q1)]),
        cirq.Moment([pauli_product]),
    )
    reaction_depth_estimator = est.ReactionDepthEstimator(
        factories={cirq.T: True, pauli_product.gate: True},
        reaction_dynamics={(pauli_product.gate, True): pauli_product_dynamics},
    )
    reaction_depth = reaction_depth_estimator.reaction_depth(circuit)
    reaction_tree = reaction_depth_estimator.reaction_tree(circuit)

    assert reaction_tree.graph["operations"] == tuple(circuit.all_operations())
    assert reaction_tree.graph["operations"][2] == cirq.T(q0)
    assert reaction_tree.graph["operations"][5] == pauli_product
    final_time = len(reaction_tree.graph["operations"])
    assert {
        qubit: {
            basis: reaction_tree.nodes[(basis, qubit, final_time)]["depth"] for basis in ("X", "Z")
        }
        for qubit in reaction_depth
    } == reaction_depth
    assert (
        max(
            reaction_tree.nodes[(basis, qubit, final_time)]["depth"]
            for qubit in reaction_depth
            for basis in ("X", "Z")
        )
        == 2
    )
    assert {
        (
            ("X", q0, 0),
            ("Z", q0, 1),
            1,
        ),
        (
            ("Z", q0, 1),
            ("X", q0, 2),
            0,
        ),
        (
            ("X", q0, 2),
            ("Z", q0, 3),
            1,
        ),
        (
            ("Z", q1, 4),
            ("X", q1, 5),
            0,
        ),
        (
            ("X", q1, 5),
            ("Z", q1, 6),
            1,
        ),
    }.issubset(
        (source, target, data["weight"]) for source, target, data in reaction_tree.edges(data=True)
    )


def test_reaction_depth_propagates_kept_primitive_cliffords() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(qubit), cirq.H(qubit))) == {
        qubit: {"X": 1, "Z": 0}
    }


def test_reaction_depth_splits_y_from_s_clifford() -> None:
    qubit = cirq.LineQubit(0)
    reaction_depth_estimator = est.ReactionDepthEstimator(factories={cirq.T: True})

    assert reaction_depth_estimator.reaction_depth(
        cirq.Circuit(cirq.T(qubit), cirq.H(qubit), cirq.S(qubit))
    ) == {qubit: {"X": 1, "Z": 1}}


def test_reaction_depth_propagates_cnot_clifford_products() -> None:
    control, target = cirq.LineQubit.range(2)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(
        cirq.Circuit(
            cirq.T(control),
            cirq.T(target),
            cirq.H(control),
            cirq.CNOT(control, target),
        )
    ) == {
        control: {"X": 1, "Z": 1},
        target: {"X": 1, "Z": 1},
    }


def test_reaction_depth_clears_source_axes_when_clifford_moves_them() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    assert reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.T(q0), cirq.SWAP(q0, q1))) == {
        q0: {"X": 0, "Z": 0},
        q1: {"X": 0, "Z": 1},
    }


def test_reaction_depth_rejects_non_factory_non_clifford() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    reaction_depth_estimator = est.ReactionDepthEstimator()

    with pytest.raises(ValueError, match="non-Clifford operation without a factory dynamic"):
        reaction_depth_estimator.reaction_depth(cirq.Circuit(cirq.CCZ(q0, q1, q2)))
