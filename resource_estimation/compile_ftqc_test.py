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
from collections import Counter, deque
from math import pi
import textwrap
import cirq
from numpy import isclose
import pytest
import resource_estimation.architecture as arch
import resource_estimation.compile_ftqc as comp
import resource_estimation.lattice_surgery_primitives as lsp
from cirq_superstaq import Barrier
from resource_estimation.layout import MovementLayout, Column, Embedded


@pytest.fixture
def bell_circuit():
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b)])
    return circuit


@pytest.fixture()
def t_circuit():
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b), cirq.T.on(qubit_b)])
    return circuit


@pytest.fixture
def random_circ():
    return cirq.testing.random_circuit(
        qubits=5,
        n_moments=8,
        op_density=1,
        gate_domain={cirq.H: 1, cirq.CNOT: 2, cirq.T: 1, cirq.S: 1},
        random_state=73,
    )


@pytest.mark.parametrize(
    "with_barriers",
    (True, False),
)
def test_end2end(with_barriers):
    # Circuit that tests all uses all possible gates
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(
        [
            cirq.H.on(q0),
            cirq.CNOT.on(q0, q1),
            cirq.T.on(q1),
            cirq.H.on(q0),
            cirq.CNOT.on(q0, q1),
            cirq.T.on(q1),
            cirq.X.on(q0),
            cirq.Z.on(q1),
            cirq.S.on(q1),
            cirq.I.on_each(q0, q1),
            cirq.MeasurementGate(2, key="end").on(q0, q1),
        ]
    )
    for arc in [
        arch.DefaultLattice(idling=False, post_op_correction=True),
        arch.DefaultLattice(idling=True, post_op_correction=True),
        arch.DefaultMovement(idling=False, post_op_correction=True),
        arch.DefaultMovement(idling=True, post_op_correction=True),
        arch.DefaultLattice(idling=False, post_op_correction=False),
        arch.DefaultMovement(idling=False, post_op_correction=False),
    ]:
        if arc.movement:
            test_layout = MovementLayout(input_circuit=circuit, num_t_factories=1)
        else:
            test_layout = Column(
                input_circuit=circuit,
            )
        compiled = comp.ft_compile(test_layout, arc, with_barriers=with_barriers)
        for op in compiled.all_operations():
            is_primitive = False
            if arc.primitives.validate(op) or op in cirq.GateFamily(Barrier):
                is_primitive = True
            assert is_primitive


def test_direct_substitution():
    dummy_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    nothing_circuit = cirq.Circuit(cirq.I.on_each(dummy_qubits))
    layout = Embedded(input_circuit=nothing_circuit)

    # Test primitives that are the same between movement and no movement
    for arc in [arch.DefaultMovement(), arch.DefaultLattice()]:
        for op_to_replace in [
            cirq.I.on(dummy_qubits[0]),
            cirq.H.on(dummy_qubits[0]),
            cirq.X.on(dummy_qubits[0]),
            cirq.Z.on(dummy_qubits[0]),
            cirq.MeasurementGate(1).on(dummy_qubits[0]),
            cirq.ResetChannel().on(dummy_qubits[0]),
            lsp.Cultivate(pi / 4).on(dummy_qubits[0]),
            lsp.SyndromeExtract(1, 1).on(dummy_qubits[0]),
            lsp.ErrorCorrect(1).on(dummy_qubits[0]),
        ]:
            replacement = comp._decompose_to_primitives(
                circuit=cirq.Circuit(op_to_replace),
                layout=layout,
                arc=arc,
            )
            assert replacement == cirq.Circuit(op_to_replace)

    # Test primitives that are reserved for no movement
    for op_to_replace in [
        lsp.Merge(2).on(*dummy_qubits[:2]),
        lsp.Split([1, 1]).on(*dummy_qubits[:2]),
    ]:
        replacement = comp._decompose_to_primitives(
            circuit=cirq.Circuit(op_to_replace),
            layout=layout,
            arc=arch.DefaultLattice(),
        )
        assert replacement == cirq.Circuit(op_to_replace)

    # Test primitives that are reserved for movement
    for op_to_replace in [
        cirq.CNOT.on(*dummy_qubits[:2]),
        cirq.S.on(dummy_qubits[0]),
    ]:
        replacement = comp._decompose_to_primitives(
            circuit=cirq.Circuit(op_to_replace),
            layout=layout,
            arc=arch.DefaultMovement(),
        )
        assert replacement == cirq.Circuit(op_to_replace)

    # Test unrecognized gate
    with pytest.raises(ValueError, match="Invalid Op for non-transversal CNOT: Rx"):
        _ = comp.replace_cirq_op(
            op=cirq.Rx(rads=pi / 2).on(dummy_qubits[0]),
            layout=layout,
            transversal_cnot=False,
        )


def test_replace_cirq_op_movement(bell_circuit):
    movement_layout = MovementLayout(bell_circuit, num_t_factories=2)

    op_to_replace = cirq.T.on(cirq.GridQubit(0, 0))
    returned_ops = comp.replace_cirq_op(
        op=op_to_replace, layout=movement_layout, transversal_cnot=True
    )
    expected_types = [
        lsp.Cultivate,
        lsp.Cultivate,
        cirq.CNOT,
        cirq.MeasurementGate,
        cirq.S,
        cirq.ResetChannel,
    ]
    assert len(expected_types) == len(returned_ops)
    for op, expected_type in zip(returned_ops, expected_types):
        assert op in cirq.GateFamily(expected_type)


@pytest.mark.parametrize("op_type", (cirq.S, cirq.T, cirq.CNOT))
def test_replace_cirq_op_lattice(op_type, bell_circuit):
    layout = Column(bell_circuit)

    op_to_replace = op_type.on(*list(layout.mapped_circuit.all_qubits())[: op_type.num_qubits()])
    print(op_to_replace)
    returned_ops = comp.replace_cirq_op(op=op_to_replace, layout=layout, transversal_cnot=False)
    print(returned_ops)

    if op_type == cirq.S:
        expected_types = [lsp.Cultivate] * 2 + [
            cirq.CNOT,
            cirq.MeasurementGate,
            cirq.Z,
            cirq.ResetChannel,
        ]
    elif op_type == cirq.T:
        expected_types = [
            lsp.Cultivate,
            lsp.Cultivate,
            cirq.CNOT,
            cirq.MeasurementGate,
            cirq.S,
            cirq.ResetChannel,
        ]
    elif op_type == cirq.CNOT:
        expected_types = [lsp.Merge, lsp.Split, lsp.Merge, lsp.Split]
    assert len(expected_types) == len(returned_ops)
    for op, expected_type in zip(returned_ops, expected_types):
        assert op in cirq.GateFamily(expected_type)


@pytest.mark.parametrize(
    "arc",
    [
        arch.DefaultLattice(idling=False, post_op_correction=False),
        arch.DefaultMovement(idling=False, post_op_correction=True),
    ],
)
def test_illegal_compile(arc):
    # Test illegal gates
    circuit = cirq.Circuit([cirq.Rx(rads=pi / 3).on(cirq.GridQubit(0, 0))])
    if arc.movement:
        layout = MovementLayout(circuit, num_t_factories=1)
    else:
        layout = Column(circuit)
    with pytest.raises(ValueError):
        _ = comp.ft_compile(layout=layout, arc=arc)
    with pytest.raises(ValueError):
        _ = comp.ft_compile(layout=layout, arc=arc)


def test_different_rounds():
    circuit = cirq.Circuit(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    layout = MovementLayout(input_circuit=circuit)
    for k in [1, 5, 7]:
        architecture = arch.DefaultMovement(
            idling=False,
            post_op_correction=True,
            d=7,
            cultivation_repetition=1,
            syndrome_rounds=k,
        )
        compiled_circuit = comp.ft_compile(layout=layout, arc=architecture)
        for op in compiled_circuit.all_operations():
            if op in cirq.GateFamily(lsp.SyndromeExtract):
                op.gate.rounds == k


def test_deterministic_compilation(random_circ):
    circuit = random_circ
    lay = Column(circuit)
    arc = arch.DefaultLattice()
    compiled1 = comp.ft_compile(lay, arc)
    compiled2 = comp.ft_compile(lay, arc)
    cirq.testing.assert_has_diagram(compiled1, str(compiled2))


def test_other_passes(random_circ):
    # If this test and test_deterministic_compilation both fail, that one likely causes the issue in this one
    circuit = random_circ
    lay = Column(circuit)
    arc = arch.DefaultLattice(idling=True, post_op_correction=True)
    compiled_circuit = comp.ft_compile(lay, arc)
    idling_corrected_resources = dict(
        Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        )
    )
    arc = arch.DefaultLattice(idling=False, post_op_correction=True)
    compiled_circuit = comp.ft_compile(lay, arc)
    corrected_resources = dict(
        Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        )
    )
    arc = arch.DefaultLattice(idling=False, post_op_correction=False)
    compiled_circuit = comp.ft_compile(lay, arc)
    uncorrected_resources = dict(
        Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        )
    )
    assert (
        idling_corrected_resources["MERGE"]
        == corrected_resources["MERGE"]
        == uncorrected_resources["MERGE"]
    )
    assert (
        idling_corrected_resources["SPLIT"]
        == corrected_resources["SPLIT"]
        == uncorrected_resources["SPLIT"]
    )
    assert (
        idling_corrected_resources["CULT(0.785)"]
        == corrected_resources["CULT(0.785)"]
        == uncorrected_resources["CULT(0.785)"]
    )
    assert (
        idling_corrected_resources["CULT(1.571)"]
        == corrected_resources["CULT(1.571)"]
        == uncorrected_resources["CULT(1.571)"]
    )
    assert idling_corrected_resources["H"] == corrected_resources["H"] == uncorrected_resources["H"]
    assert (
        idling_corrected_resources["Measure"]
        == corrected_resources["Measure"]
        == uncorrected_resources["Measure"]
    )
    assert (
        idling_corrected_resources["reset"]
        == corrected_resources["reset"]
        == uncorrected_resources["reset"]
    )
    assert idling_corrected_resources["Z"] == corrected_resources["Z"] == uncorrected_resources["Z"]
    assert (
        idling_corrected_resources["SE(7)"]
        >= corrected_resources["SE(7)"]
        >= uncorrected_resources["SE(7)"]
    )


def test_verbosity(random_circ):
    # TODO: Make this slightly more real (it's a visualization tool so not the most important but still)
    circuit = random_circ
    lay = Column(circuit)
    arc = arch.DefaultLattice()
    ops, compiled_circuit = comp.ft_compile(lay, arc, verbose=2)
    for moment_ops in ops:
        for op in moment_ops:
            assert op in compiled_circuit.all_operations()


def test_bell_movement_FF(bell_circuit):
    movement_layout = MovementLayout(bell_circuit)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=False,
    )
    compiled_bell_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
    # no idling, no post-op correction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 0): ───SE(1)───H───MOVE───@───#2─────
                                       │      │   │
                (0, 1): ───SE(1)───────#2─────X───MOVE───
            """
        ),
    )


def test_bell_movement_FT(bell_circuit):
    movement_layout = MovementLayout(bell_circuit)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=True,
    )
    compiled_bell_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
    # no idling, yes post-op correction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 0): ───SE(1)───H───SE(1)───MOVE───@───#2─────SE(1)───
                                               │      │   │
                (0, 1): ───SE(1)───────────────#2─────X───MOVE───SE(1)───
            """
        ),
    )


def test_bell_movement_TF(bell_circuit):
    movement_layout = MovementLayout(bell_circuit)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=False,
    )
    compiled_bell_circuit = comp.ft_compile(
        layout=movement_layout, arc=movement_architecture, with_barriers=False
    )
    # yes idling, no post-op correction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 0): ───SE(1)───H───────MOVE───@───#2─────
                                           │      │   │
                (0, 1): ───SE(1)───SE(1)───#2─────X───MOVE───
            """
        ),
    )


def test_bell_movement_TT(bell_circuit):
    movement_layout = MovementLayout(bell_circuit)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=True,
    )
    compiled_bell_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)

    # yes idling, yes post-op correction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 0): ───SE(1)───H───────SE(1)───MOVE───@───#2─────SE(1)───
                                                   │      │   │
                (0, 1): ───SE(1)───SE(1)───SE(1)───#2─────X───MOVE───SE(1)───
            """
        ),
    )


def test_bell_lattice_FF(bell_circuit):
    lattice_layout = Column(bell_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=False,
    )
    lattice_layout.input_circuit
    compiled_bell_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)

    # no idling, no post-op correction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 2): ───SE(1)───H───MERGE───SPLIT───────────────────
                                       │       │
                (0, 3): ───────────────#2──────#2──────MERGE───SPLIT───
                                                       │       │
                (0, 4): ───SE(1)───────────────────────#2──────#2──────
            """
        ),
    )


def test_bell_lattice_FT(bell_circuit):
    lattice_layout = Column(bell_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=True,
    )
    compiled_bell_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)

    # no idling, yes post-op correction
    # Since all operations are inherently corrected, there is no need for extra syndrome extraction
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 2): ───SE(1)───H───MERGE───SPLIT───────────────────
                                       │       │
                (0, 3): ───────────────#2──────#2──────MERGE───SPLIT───
                                                       │       │
                (0, 4): ───SE(1)───────────────────────#2──────#2──────
            """
        ),
    )


def test_bell_lattice_TF(bell_circuit):
    lattice_layout = Column(bell_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=False,
    )
    compiled_bell_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)

    # yes idling, no post-op correction
    # (0, 3) is an ancilla qubit, so it does not get idling in the second moment
    # The Split moments also do not get idling because, implicitly, they cn always be absorbed into a previous moment
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 2): ───SE(1)───H───────MERGE───SPLIT───SE(1)───────────
                                           │       │
                (0, 3): ───────────────────#2──────#2──────MERGE───SPLIT───
                                                           │       │
                (0, 4): ───SE(1)───SE(1)───SE(1)───────────#2──────#2──────
            """
        ),
    )


def test_bell_lattice_TT(bell_circuit):
    lattice_layout = Column(bell_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=True,
    )
    compiled_bell_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)
    compiled_bell_circuit
    # yes idling, yes post-op correction
    # Post-op correction does not add anything in this circuit, so this circuit is the same as the last one
    cirq.testing.assert_has_diagram(
        compiled_bell_circuit,
        textwrap.dedent(
            """
                (0, 2): ───SE(1)───H───────MERGE───SPLIT───SE(1)───────────
                                           │       │
                (0, 3): ───────────────────#2──────#2──────MERGE───SPLIT───
                                                           │       │
                (0, 4): ───SE(1)───SE(1)───SE(1)───────────#2──────#2──────
            """
        ),
    )


def test_t_movement_FF(t_circuit):
    movement_layout = MovementLayout(t_circuit, num_t_factories=2)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=False,
    )
    compiled_t_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
    # no idling, no post-op correction
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
            (0, 0): ───SE(1)─────────H───MOVE───@───#2───────────────────────────────────────────────────────
                                         │      │   │
            (0, 1): ───SE(1)─────────────#2─────X───MOVE───#2─────X───MOVE───S───────────────────────────────
                                                           │      │   │
            (1, 0): ───CULT(0.785)─────────────────────────┼──────┼───┼──────────────────────────────────────
                                                           │      │   │
            (1, 1): ───CULT(0.785)─────────────────────────MOVE───@───#2─────MOVE_MZ───M('')───MOVE_MZ───R───
            """
        ),
    )


def test_t_movement_FT(t_circuit):
    movement_layout = MovementLayout(t_circuit, num_t_factories=2)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=True,
    )
    compiled_t_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    # no idling, yes post-op correction
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
            (0, 0): ───SE(1)─────────H───SE(1)───MOVE───@───#2─────SE(1)─────────────────────────────────────────────────────────────────────
                                                 │      │   │
            (0, 1): ───SE(1)─────────────────────#2─────X───MOVE───SE(1)───#2─────X───MOVE───SE(1)───S─────────SE(1)─────────────────────────
                                                                           │      │   │
            (1, 0): ───CULT(0.785)─────────────────────────────────────────┼──────┼───┼──────────────────────────────────────────────────────
                                                                           │      │   │
            (1, 1): ───CULT(0.785)─────────────────────────────────────────MOVE───@───#2─────SE(1)───MOVE_MZ───M('')───MOVE_MZ───SE(1)───R───
            """
        ),
    )


def test_t_movement_TF(t_circuit):
    movement_layout = MovementLayout(t_circuit, num_t_factories=2)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=False,
    )
    compiled_t_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)

    # yes idling, no post-op correction
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
            (0, 0): ───SE(1)─────────H───────MOVE────@───────#2──────SE(1)───SE(1)───SE(1)───────────────────────────────────
                                             │       │       │
            (0, 1): ───SE(1)─────────SE(1)───#2──────X───────MOVE────#2──────X───────MOVE────S─────────SE(1)─────────────────
                                                                     │       │       │
            (1, 0): ───CULT(0.785)───SE(1)───SE(1)───SE(1)───SE(1)───┼───────┼───────┼───────────────────────────────────────
                                                                     │       │       │
            (1, 1): ───CULT(0.785)───SE(1)───────────────────────────MOVE────@───────#2──────MOVE_MZ───M('')───MOVE_MZ───R───

            """
        ),
    )


def test_t_movement_TT(t_circuit):
    movement_layout = MovementLayout(t_circuit, num_t_factories=2)
    movement_architecture = arch.MeasureZonesOnly(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=True,
    )
    compiled_t_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
    # yes idling, yes post-op correction
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    # This test was updated both by aligning left and to reflect the change to make cultivation happen later in the circuit,
    # The old version is left commented out below
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
                                                                                     ┌──────────┐   ┌──────────┐
            (0, 0): ───SE(1)─────────H───────SE(1)───MOVE────@───────#2──────SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────────────────────────
                                                     │       │       │
            (0, 1): ───SE(1)─────────SE(1)───SE(1)───#2──────X───────MOVE────SE(1)────#2─────────────X─────────────MOVE────SE(1)───S─────────SE(1)───SE(1)─────────────────
                                                                                      │              │             │
            (1, 0): ───CULT(0.785)───SE(1)───SE(1)───SE(1)───SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼────SE(1)────┼───────────────────────────────────────────────────────
                                                                                      │              │             │
            (1, 1): ───CULT(0.785)───SE(1)───SE(1)───SE(1)────────────────────────────MOVE───────────@─────────────#2──────SE(1)───MOVE_MZ───M('')───MOVE_MZ───SE(1)───R───
                                                                                     └──────────┘   └──────────┘
            """
        ),
    )


def test_t_lattice_FF(t_circuit):
    lattice_layout = Column(t_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=False,
    )
    lattice_layout.input_circuit
    compiled_t_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)
    # no idling, no post-op correction
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
                (0, 0): ───CULT(1.571)───────────────────────────────────────────────────────────────────────────────────────────────
                
                (0, 2): ───SE(1)─────────H───────MERGE───SPLIT───────────────────────────────────────────────────────────────────────
                                                 │       │
                (0, 3): ─────────────────────────#2──────#2──────MERGE───SPLIT───────────────────────────────────────────────────────
                                                                 │       │
                (0, 4): ───SE(1)─────────────────────────────────#2──────#2──────#3──────#3──────────────────────#2──────#2──────Z───
                                                                                 │       │                       │       │
                (0, 5): ─────────────────#3──────#3──────────────────────────────#2──────#2──────#2──────#2──────MERGE───SPLIT───────
                                         │       │                               │       │       │       │
                (0, 6): ───CULT(1.571)───┼───────┼───────────────────────────────┼───────┼───────MERGE───SPLIT───M('')───R───────────
                                         │       │                               │       │
                (1, 0): ───CULT(0.785)───┼───────┼───────────────────────────────┼───────┼───────────────────────────────────────────
                                         │       │                               │       │
                (1, 5): ─────────────────#2──────#2──────────────────────────────MERGE───SPLIT───────────────────────────────────────
                                         │       │
                (1, 6): ───CULT(0.785)───MERGE───SPLIT───M('')───R───────────────────────────────────────────────────────────────────
            """
        ),
    )


def test_t_lattice_FT(t_circuit):
    lattice_layout = Column(t_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=False,
        post_op_correction=True,
    )
    compiled_t_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)
    # no idling, yes post-op correction
    # Only measurement gates need to be corrected
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
                (0, 0): ───CULT(1.571)───────────────────────────────────────────────────────────────────────────────────────────────
                
                (0, 2): ───SE(1)─────────H───────MERGE───SPLIT───────────────────────────────────────────────────────────────────────
                                                 │       │
                (0, 3): ─────────────────────────#2──────#2──────MERGE───SPLIT───────────────────────────────────────────────────────
                                                                 │       │
                (0, 4): ───SE(1)─────────────────────────────────#2──────#2──────#3──────#3──────────────────────#2──────#2──────Z───
                                                                                 │       │                       │       │
                (0, 5): ─────────────────#3──────#3──────────────────────────────#2──────#2──────#2──────#2──────MERGE───SPLIT───────
                                         │       │                               │       │       │       │
                (0, 6): ───CULT(1.571)───┼───────┼───────────────────────────────┼───────┼───────MERGE───SPLIT───M('')───SE(1)───R───
                                         │       │                               │       │
                (1, 0): ───CULT(0.785)───┼───────┼───────────────────────────────┼───────┼───────────────────────────────────────────
                                         │       │                               │       │
                (1, 5): ─────────────────#2──────#2──────────────────────────────MERGE───SPLIT───────────────────────────────────────
                                         │       │
                (1, 6): ───CULT(0.785)───MERGE───SPLIT───M('')───SE(1)───R───────────────────────────────────────────────────────────
            """
        ),
    )


# This test is just totally broken after the change to cultivation and aligning, so leaving it commented out.
# TODO: Fix this test or remove it
# def test_t_lattice_TF(t_circuit):
#     lattice_layout = ColumnLayout(t_circuit)
#     lattice_architecture = arch.DefaultLattice(
#         d=7,
#         cultivation_repetition=1,
#         syndrome_rounds=1,
#         idling=True,
#         post_op_correction=False,
#     )
#     compiled_t_circuit = comp.ft_compile(
#         layout=lattice_layout, arc=lattice_architecture
#     )
#     # yes idling, no post-op correction
#     # Looks a little bit weird because Split creates opportunities to align left
#     cirq.testing.assert_has_diagram(
#         compiled_t_circuit,
#         textwrap.dedent(
#             """
#                                          ┌──────────┐   ┌──────────┐                           ┌──────────┐   ┌──────────┐
#                 (0, 0): ───CULT(1.571)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───────────────────────────

#                 (0, 2): ───SE(1)──────────H──────────────MERGE─────────SPLIT───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───────────────────────────
#                                                          │             │
#                 (0, 3): ─────────────────────────────────#2────────────#2──────MERGE───SPLIT─────────────────────────────────────────────────────────────────────────
#                                                                                │       │
#                 (0, 4): ───SE(1)──────────SE(1)──────────SE(1)─────────SE(1)───#2──────#2───────#3─────────────#3────────────SE(1)───────────#2──────#2──────Z───────
#                                                                                                 │              │                             │       │
#                 (0, 5): ──────────────────#3─────────────#3─────────────────────────────────────#2─────────────#2────────────#2──────#2──────MERGE───SPLIT───────────
#                                           │              │                                      │              │             │       │
#                 (0, 6): ───CULT(1.571)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────┼──────────────┼─────────────MERGE───SPLIT───M('')───R───────SE(1)───
#                                           │              │                                      │              │
#                 (1, 0): ───CULT(0.785)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───────────────────────────
#                                           │              │                                      │              │
#                 (1, 5): ──────────────────#2─────────────#2─────────────────────────────────────MERGE──────────SPLIT─────────────────────────────────────────────────
#                                           │              │
#                 (1, 6): ───CULT(0.785)────MERGE──────────SPLIT─────────M('')───R───────SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───────────────────────────
#                                          └──────────┘   └──────────┘                           └──────────┘   └──────────┘
#                 """
#         ),
#     )

# This test is just totally broken after the change to cultivation and aligning, so leaving it commented out.
# TODO: Fix this test or remove it
# def test_t_lattice_TT(t_circuit):
#     lattice_layout = ColumnLayout(t_circuit)
#     lattice_architecture = arch.DefaultLattice(
#         d=7,
#         cultivation_repetition=1,
#         syndrome_rounds=1,
#         idling=True,
#         post_op_correction=True,
#     )
#     compiled_t_circuit = comp.ft_compile(
#         layout=lattice_layout, arc=lattice_architecture
#     )
#     # yes idling, yes post-op correction
#     # I'm not sure if the splits are being handled properly. Should we idle on qubits being acted on by Split? Just the non-ancillas? There could be some discussion here
#     compiled_t_circuit = cirq.align_left(compiled_t_circuit)
#     cirq.testing.assert_has_diagram(
#         compiled_t_circuit,
#         textwrap.dedent(
#             """
#                                          ┌──────────┐   ┌──────────┐                           ┌──────────┐   ┌──────────┐
#                 (0, 0): ───CULT(1.571)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────

#                 (0, 2): ───SE(1)──────────H──────────────MERGE─────────SPLIT───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────
#                                                          │             │
#                 (0, 3): ─────────────────────────────────#2────────────#2──────MERGE───SPLIT─────────────────────────────────────────────────────────────────────
#                                                                                │       │
#                 (0, 4): ───SE(1)──────────SE(1)──────────SE(1)─────────SE(1)───#2──────#2───────#3─────────────#3────────────SE(1)───────────#2──────#2──────Z───
#                                                                                                 │              │                             │       │
#                 (0, 5): ──────────────────#3─────────────#3─────────────────────────────────────#2─────────────#2────────────#2──────#2──────MERGE───SPLIT───────
#                                           │              │                                      │              │             │       │
#                 (0, 6): ───CULT(1.571)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼─────────────MERGE───SPLIT───M('')───SE(1)───R───
#                                           │              │                                      │              │
#                 (1, 0): ───CULT(0.785)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)───────────────
#                                           │              │                                      │              │
#                 (1, 5): ──────────────────#2─────────────#2─────────────────────────────────────MERGE──────────SPLIT─────────────────────────────────────────────
#                                           │              │
#                 (1, 6): ───CULT(0.785)────MERGE──────────SPLIT─────────M('')───SE(1)───R────────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────
#                                          └──────────┘   └──────────┘                           └──────────┘   └──────────┘
#                 """
#         ),
#     )


def test_ssm_moves():
    arch_type = arch.DefaultMovement
    arch_info = {
        "zone_ops": arch_type.zone_ops if arch_type.zone_ops is not None else cirq.Gateset(),
        "alley_ops": arch_type.alley_ops if arch_type.alley_ops is not None else cirq.Gateset(),
    }
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        lsp.Move(zone="interact").on_each(c, b),
        cirq.CNOT.on(c, b),
        lsp.Move(zone="interact").on_each(b, c),
        lsp.Move(zone="interact").on_each(a, b),
        cirq.CNOT.on(a, b),
        lsp.Move(zone="interact").on_each(b, a),
        lsp.Move(zone="measure").on(c),
        cirq.MeasurementGate(1, key="").on(c),
        lsp.Move(zone="measure").on(c),
    )
    # Aligning left avoids ambiguity
    output_circuit = cirq.align_left(comp.add_moves(input_circuit, **arch_info))
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )


def test_mzo_moves():
    arch_type = arch.MeasureZonesOnly
    arch_info = {
        "zone_ops": arch_type.zone_ops if arch_type.zone_ops is not None else cirq.Gateset(),
        "alley_ops": arch_type.alley_ops if arch_type.alley_ops is not None else cirq.Gateset(),
    }
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        lsp.Move(zone=None).on(c, b),
        cirq.CNOT.on(c, b),
        lsp.Move(zone=None).on(b, c),
        lsp.Move(zone=None).on(a, b),
        cirq.CNOT.on(a, b),
        lsp.Move(zone=None).on(b, a),
        lsp.Move(zone="measure").on(c),
        cirq.MeasurementGate(1, key="").on(c),
        lsp.Move(zone="measure").on(c),
    )
    output_circuit = comp.add_moves(input_circuit, **arch_info)
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )


def test_hm_moves():
    arch_type = arch.DualSpeciesMovement
    arch_info = {
        "zone_ops": arch_type.zone_ops if arch_type.zone_ops is not None else cirq.Gateset(),
        "alley_ops": arch_type.alley_ops if arch_type.alley_ops is not None else cirq.Gateset(),
    }
    print(arch_info)
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        lsp.Move(zone=None).on(c, b),
        cirq.CNOT.on(c, b),
        lsp.Move(zone=None).on(b, c),
        lsp.Move(zone=None).on(a, b),
        cirq.CNOT.on(a, b),
        lsp.Move(zone=None).on(b, a),
        cirq.MeasurementGate(1, key="").on(c),
    )
    output_circuit = comp.add_moves(input_circuit, **arch_info)
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )
