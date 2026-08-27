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
import collections
import textwrap
from math import pi

import cirq
import cirq_superstaq as css
import pytest

import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.compile_ftqc as comp
import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.layout import (
    Column,
    Embedded,
    MovementDistillery,
    MovementLayout,
)
from resource_estimation.typing import GateKey, _require_gate_operation


@pytest.fixture
def bell_circuit() -> cirq.Circuit:
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b)])
    return circuit


@pytest.fixture
def t_circuit() -> cirq.Circuit:
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b), cirq.T.on(qubit_b)])
    return circuit


@pytest.fixture
def random_circ() -> cirq.Circuit:
    return cirq.testing.random_circuit(
        qubits=5,
        n_moments=8,
        op_density=1,
        gate_domain={cirq.H: 1, cirq.CNOT: 2, cirq.T: 1, cirq.S: 1},
        random_state=73,
    )


@pytest.fixture
def random_circ2() -> cirq.Circuit:
    return cirq.testing.random_circuit(
        qubits=4,
        n_moments=7,
        op_density=0.8,
        gate_domain={cirq.H: 1, cirq.CNOT: 2, cirq.T: 1, cirq.S: 1},
        random_state=73,
    )


@pytest.mark.parametrize(
    "with_barriers",
    (True, False),
)
def test_end2end(with_barriers: bool) -> None:
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
        ],
    )
    test_layout: MovementLayout | Column
    for arc in [
        arch.DefaultLattice(idling=False, post_op_correction=True),
        arch.DefaultLattice(idling=True, post_op_correction=True),
        arch.DefaultMovement(idling=False, post_op_correction=True),
        arch.DefaultMovement(idling=True, post_op_correction=True),
        arch.DefaultLattice(idling=False, post_op_correction=False),
        arch.DefaultMovement(idling=False, post_op_correction=False),
    ]:
        if arc.movement:
            test_layout = MovementLayout(
                input_circuit=circuit, num_t_factories=1, architecture="SSM"
            )
        else:
            test_layout = Column(
                input_circuit=circuit,
            )
        compiled = comp.ft_compile(test_layout, arc, with_barriers=with_barriers)
        for op in compiled.all_operations():
            is_primitive = False
            if arc.primitives.validate(op) or op in cirq.GateFamily(css.Barrier):
                is_primitive = True
            assert is_primitive


def test_end2end_distillery() -> None:
    q1, q2, q3 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
    circuit = cirq.Circuit(
        [cirq.CNOT.on(q1, q2), cirq.CCZ.on(q1, q2, q3), cirq.T.on_each(q1, q2, q3)]
    )
    layout = MovementDistillery(
        input_circuit=circuit, num_t_factories=1, num_ccz_factories=1, architecture="SSM"
    )
    arc = arch.DefaultMovement(post_op_correction=False, idling=False)
    compiled = comp.ft_compile(layout, arc, with_barriers=False)
    assert all(arc.primitives.validate(op) for op in compiled.all_operations())


def test_direct_substitution() -> None:
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
        lsp.Distil("T").on(*cirq.LineQubit.range(31)),
    ]:
        replacement = comp._decompose_to_primitives(
            circuit=cirq.Circuit(op_to_replace),
            layout=layout,
            arc=arch.DefaultMovement(),
        )
        assert replacement == cirq.Circuit(op_to_replace)

    # Test unrecognized gate
    with pytest.raises(ValueError, match="Invalid Op for non-transversal gate: Rx"):
        _ = comp.replace_cirq_op(
            op=cirq.Rx(rads=pi / 2).on(dummy_qubits[0]),
            layout=layout,
            transversal_cnot=False,
        )

    # Test TypeError is raised
    with pytest.raises(TypeError, match="Qubits must be instances"):
        _ = comp.replace_cirq_op(
            cirq.CNOT.on(*cirq.LineQubit.range(2)), layout=layout, transversal_cnot=False
        )


def test_replace_cirq_op_movement(bell_circuit: cirq.Circuit) -> None:
    movement_layout = MovementLayout(bell_circuit, num_t_factories=2, architecture="DSM")

    op_to_replace = cirq.T.on(cirq.GridQubit(0, 0))
    returned_ops = comp.replace_cirq_op(
        op=op_to_replace,
        layout=movement_layout,
        transversal_cnot=True,
    )
    ops_flattened = list(cirq.flatten_to_ops(returned_ops))
    expected_types: list[GateKey] = [
        lsp.Cultivate,
        lsp.Cultivate,
        cirq.CNOT,
        cirq.MeasurementGate,
        cirq.ResetChannel,
        cirq.S,
    ]
    assert len(expected_types) == len(ops_flattened)
    for op, expected_type in zip(ops_flattened, expected_types):
        assert op in cirq.GateFamily(expected_type)


@pytest.mark.parametrize("op_type", (cirq.S, cirq.T, cirq.CNOT))
def test_replace_cirq_op_lattice(op_type: cirq.Gate, bell_circuit: cirq.Circuit) -> None:
    layout = Column(bell_circuit)

    op_to_replace = op_type.on(*list(layout.mapped_circuit.all_qubits())[: op_type.num_qubits()])
    returned_ops = comp.replace_cirq_op(op=op_to_replace, layout=layout, transversal_cnot=False)
    ops_flattened = list(cirq.flatten_to_ops(returned_ops))
    expected_types: list[GateKey]
    if op_type == cirq.S:
        expected_types = [
            lsp.Cultivate,
            lsp.Cultivate,
            cirq.CNOT,
            cirq.MeasurementGate,
            cirq.ResetChannel,
            cirq.Z,
        ]
    elif op_type == cirq.T:
        expected_types = [
            lsp.Cultivate,
            lsp.Cultivate,
            cirq.CNOT,
            cirq.MeasurementGate,
            cirq.ResetChannel,
            cirq.S,
        ]
    elif op_type == cirq.CNOT:
        expected_types = [lsp.Merge, lsp.Split, lsp.Merge, lsp.Split]
    assert len(expected_types) == len(ops_flattened)
    for op, expected_type in zip(ops_flattened, expected_types):
        assert op in cirq.GateFamily(expected_type)


@pytest.mark.parametrize(
    "arc",
    [
        arch.DefaultLattice(idling=False, post_op_correction=False),
        arch.DefaultMovement(idling=False, post_op_correction=True),
    ],
)
def test_illegal_compile(arc: arch.Architecture) -> None:
    # Test illegal gates
    circuit = cirq.Circuit([cirq.Rx(rads=pi / 3).on(cirq.GridQubit(0, 0))])
    layout: MovementLayout | Column
    if arc.movement:
        layout = MovementLayout(circuit, num_t_factories=1, architecture="SSM")
    else:
        layout = Column(circuit)
    with pytest.raises(ValueError, match="This compiler only handles"):
        _ = comp.ft_compile(layout=layout, arc=arc)


def test_different_rounds() -> None:
    circuit = cirq.Circuit(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    layout = MovementLayout(input_circuit=circuit, architecture="SSM")
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
            op = arch._require_gate_operation(op)
            gate = op.gate
            if isinstance(gate, lsp.SyndromeExtract):
                assert gate.rounds == k


def test_deterministic_compilation(random_circ: cirq.Circuit) -> None:
    circuit = random_circ
    lay = Column(circuit)
    arc = arch.DefaultLattice()
    compiled1 = comp.ft_compile(lay, arc)
    compiled2 = comp.ft_compile(lay, arc)
    cirq.testing.assert_has_diagram(compiled1, str(compiled2))


def test_nondeterministic_compilation_T(random_circ2: cirq.Circuit) -> None:
    circuit = random_circ2
    lay = MovementDistillery(circuit, num_t_factories=1, architecture="SSM")
    arc = arch.DefaultMovement()
    compiled1 = comp.ft_compile(lay, arc, dynamic=False)
    compiled2 = comp.ft_compile(lay, arc, dynamic=True)
    # We expect basically the same operations except that there are ResourceCorrection gates instead
    # of S gates
    # Hoping that this is deterministic
    compiled1_ops = list(compiled1.all_operations())
    compiled2_ops = list(compiled2.all_operations())
    assert len(compiled1_ops) == len(compiled2_ops)
    for i, (op1, op2) in enumerate(zip(compiled1_ops, compiled2_ops)):
        op1 = _require_gate_operation(op1)
        op2 = _require_gate_operation(op2)
        if op1.gate != op2.gate:
            assert isinstance(op1.gate, cirq.ZPowGate)
            assert isinstance(op2.gate, lsp.ResourceCorrection)
            assert op2.gate.resource == "T"


def test_nondeterministic_compilation_CCZ() -> None:
    circuit = cirq.Circuit(
        cirq.CCZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
    )
    lay = MovementDistillery(circuit, num_t_factories=0, num_ccz_factories=1, architecture="SSM")
    arc = arch.DefaultMovement()
    compiled1 = comp.ft_compile(lay, arc, dynamic=False)
    compiled2 = comp.ft_compile(lay, arc, dynamic=True)
    # We expect the same operations on the CCZ qubits until there is a ResourceCorrection gate
    compiled1_ops = list(compiled1.all_operations())
    compiled2_ops = list(compiled2.all_operations())

    def relevant_op(op: cirq.Operation) -> bool:
        qubits = op.qubits
        if (
            cirq.GridQubit(0, 0) in qubits
            or cirq.GridQubit(0, 1) in qubits
            or cirq.GridQubit(0, 2) in qubits
        ):
            return True
        return False

    reached_correction = False
    moments1 = list(compiled1.moments)
    moments2 = list(compiled2.moments)
    correction_circuit = cirq.Circuit()
    for i in range(0, len(moments1)):
        # The circuits should be identical until we reach the correction part
        if not reached_correction:
            try:
                assert len(moments1[i].operations) == len(moments2[i].operations)
            except:
                reached_correction = True
                for op in moments1[i].operations:
                    if relevant_op(op):
                        assert isinstance(op.gate, cirq.HPowGate)
                for op in moments2[i].operations:
                    if relevant_op(op):
                        assert isinstance(op.gate, lsp.ResourceCorrection)
                        assert op.gate.resource == "CCZ"

        if not reached_correction:
            for j in range(0, len(moments2[i].operations)):
                assert moments1[i].operations[j] == moments2[i].operations[j]
        else:
            # Once we reach the correction part, we want to make sure compilation does exactly what
            # we expect on the logical qubits
            for op in moments1[i].operations:
                if relevant_op(op):
                    correction_circuit.append(op)
    # # If this errors, look at the correction circuit to make sure it's not just that the CNOT order
    # # changed or the qubit indices changes
    # cirq.testing.assert_has_diagram(
    #     correction_circuit,
    #     textwrap.dedent(
    #         """
    #     (0, 0): ───H───SE(1)───X───MOVE_IZ───@───MOVE_IZ───SE(1)───MOVE_IZ───@───MOVE_IZ───SE(1)───H─────────SE(1)─────────────────────────────────
    #                                          │                               │
    #     (0, 1): ───H───SE(1)───X───MOVE_IZ───X───MOVE_IZ───SE(1)───MOVE_IZ───┼───────────────────────────────@───────MOVE_IZ───SE(1)───H───SE(1)───
    #                                                                          │                               │
    #     (0, 2): ───H───SE(1)───X───MOVE_IZ───────────────────────────────────X───MOVE_IZ───SE(1)───MOVE_IZ───X───────MOVE_IZ───SE(1)───H───SE(1)───
    #     """
    #     ),
    # )


def test_other_passes(random_circ: cirq.Circuit) -> None:
    # If this test and test_deterministic_compilation both fail, that one likely causes the issue in this one
    circuit = random_circ
    lay = Column(circuit)
    arc = arch.DefaultLattice(idling=True, post_op_correction=True)
    compiled_circuit = comp.ft_compile(lay, arc)
    idling_corrected_resources = dict(
        collections.Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        ),
    )
    arc = arch.DefaultLattice(idling=False, post_op_correction=True)
    compiled_circuit = comp.ft_compile(lay, arc)
    corrected_resources = dict(
        collections.Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        ),
    )
    arc = arch.DefaultLattice(idling=False, post_op_correction=False)
    compiled_circuit = comp.ft_compile(lay, arc)
    uncorrected_resources = dict(
        collections.Counter(
            str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
            for op in compiled_circuit.all_operations()
        ),
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


def test_t_movement(t_circuit: cirq.Circuit) -> None:
    movement_layout = MovementLayout(t_circuit, num_t_factories=2, architecture="SSM")
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
                                                          ┌────────────┐                             ┌───────────┐
       (-1, 0): ─────────────────────────────────┤   0├────┤   0├──────────────────┤1   ├───┤1   ├───────────────────────────────────────────────────────────────────────────────────────────
                                                    
       (-1, 1): ─────────────────────────────────────────────────┤   0├────────────────────────────────────┤   0├────────────┤1   ├───┤1   ├─────────────────────────────────────────────────
                                                    
       (0, 0): ────SE(1)─────────H───────SE(1)───┤0   ├────────────────────@────────────────┤   1├────SE(1)──────────SE(1)───SE(1)────SE(1)────SE(1)───SE(1)─────────────────────────────────
                                                                           │
       (0, 1): ────SE(1)─────────SE(1)───SE(1)─────────────┤0   ├──────────X───────┤   1├───SE(1)──────────┤0   ├────X───────┤   1├───SE(1)────S───────SE(1)────SE(1)────────────────────────
                                                                                                                     │
       (1, 0): ────CULT(0.785)───SE(1)───SE(1)───SE(1)─────SE(1)───────────SE(1)───SE(1)────SE(1)─────SE(1)──────────┼───────────────────────────────────────────────────────────────────────
                                                                                                                     │
       (1, 1): ────CULT(0.785)───SE(1)───SE(1)───SE(1)───────────┤0   ├──────────────────────────────────────────────@────────────────┤   1├───SE(1)───┤0   ├───M('')───┤   1├───SE(1)───R───
                                                    
       (2, 0): ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤   0├───────────┤1   ├───────────────
                                                          └────────────┘                             └───────────┘   
            """,
        ),
    )


def test_t_lattice(t_circuit: cirq.Circuit):
    lattice_layout = Column(t_circuit)
    lattice_architecture = arch.DefaultLattice(
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=1,
        idling=True,
        post_op_correction=True,
    )
    compiled_t_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)
    # yes idling, yes post-op correction
    # I'm not sure if the splits are being handled properly. Should we idle on qubits being acted on by Split? Just the non-ancillas? There could be some discussion here
    compiled_t_circuit = cirq.align_left(compiled_t_circuit)
    cirq.testing.assert_has_diagram(
        compiled_t_circuit,
        textwrap.dedent(
            """
                                ┌──────────┐   ┌──────────┐                                   ┌──────────┐   ┌──────────┐
       (0, 0): ───CULT(1.571)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───────────────────────────────────
                                                                                                                                                                                                                    
       (0, 2): ───SE(1)──────────H──────────────MERGE─────────SPLIT───SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───SE(1)───────────────────
                                                │             │
       (0, 3): ─────────────────────────────────#2────────────#2──────MERGE───────────SPLIT─────────────────────────────────────────────────────────────────────────────────
                                                                      │               │
       (0, 4): ───SE(1)──────────SE(1)──────────SE(1)─────────────────#2──────SE(1)───#2───────#3─────────────#3────────────SE(1)───────────#2──────SE(1)───#2──────Z───────
                                                                                               │              │                             │               │
       (0, 5): ──────────────────#3─────────────#3─────────────────────────────────────────────#2─────────────#2────────────#2──────#2──────MERGE───────────SPLIT───────────
                                 │              │                                              │              │             │       │
       (0, 6): ───CULT(1.571)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────────────┼──────────────┼─────────────MERGE───SPLIT───M('')───SE(1)───R───────SE(1)───
                                 │              │                                              │              │
       (1, 0): ───CULT(0.785)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───────────────────────────────────
                                 │              │                                              │              │
       (1, 5): ──────────────────#2─────────────#2─────────────────────────────────────────────MERGE──────────SPLIT─────────────────────────────────────────────────────────
                                 │              │
       (1, 6): ───CULT(0.785)────MERGE──────────SPLIT─────────M('')───SE(1)───R───────SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────────────────
                                └──────────┘   └──────────┘                                   └──────────┘   └──────────┘        
                """
        ),
    )


def test_ssm_moves() -> None:
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    layout = MovementLayout(
        input_circuit=input_circuit, num_t_factories=1, num_ccz_factories=0, architecture="SSM"
    )
    interaction_qids = layout.zone_qubits(zone_type="interact")
    measurement_qids = layout.zone_qubits(zone_type="measure")
    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        css.MovementGate({0: 1}).on(c, interaction_qids[0]),
        css.MovementGate({0: 1}).on(b, interaction_qids[0]),
        cirq.CNOT.on(c, b),
        css.MovementGate({1: 0}).on(b, interaction_qids[0]),
        css.MovementGate({1: 0}).on(c, interaction_qids[0]),
        css.MovementGate({0: 1}).on(a, interaction_qids[1]),
        css.MovementGate({0: 1}).on(b, interaction_qids[1]),
        cirq.CNOT.on(a, b),
        css.MovementGate({1: 0}).on(b, interaction_qids[1]),
        css.MovementGate({1: 0}).on(a, interaction_qids[1]),
        css.MovementGate({0: 1}).on(c, measurement_qids[0]),
        cirq.MeasurementGate(1, key="").on(c),
        css.MovementGate({1: 0}).on(c, measurement_qids[0]),
    )
    # Aligning left avoids ambiguity
    output_circuit = cirq.align_left(comp.add_moves(circuit=layout.mapped_circuit, layout=layout))
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )


def test_mzo_moves() -> None:
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )

    layout = MovementLayout(
        input_circuit=input_circuit, num_t_factories=1, num_ccz_factories=0, architecture="MZO"
    )
    measurement_qids = layout.zone_qubits(zone_type="measure")

    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        css.MovementGate({0: 1}).on(c, b),
        cirq.CNOT.on(c, b),
        css.MovementGate({1: 0}).on(c, b),
        css.MovementGate({0: 1}).on(a, b),
        cirq.CNOT.on(a, b),
        css.MovementGate({1: 0}).on(a, b),
        css.MovementGate({0: 1}).on(c, measurement_qids[0]),
        cirq.MeasurementGate(1, key="").on(c),
        css.MovementGate({1: 0}).on(c, measurement_qids[0]),
    )
    output_circuit = cirq.align_left(comp.add_moves(circuit=layout.mapped_circuit, layout=layout))
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )


def test_hm_moves() -> None:
    a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
    input_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        cirq.CNOT.on(c, b),
        cirq.CNOT.on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    layout = MovementLayout(
        input_circuit=input_circuit,
        num_t_factories=1,
        num_ccz_factories=0,
        architecture="DSM",
    )
    expected_output_circuit = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on_each(a, b),
        lsp.Cultivate(pi / 4).on(c),
        css.MovementGate({0: 1}).on(c, b),
        cirq.CNOT.on(c, b),
        css.MovementGate({1: 0}).on(c, b),
        css.MovementGate({0: 1}).on(a, b),
        cirq.CNOT.on(a, b),
        css.MovementGate({1: 0}).on(a, b),
        cirq.MeasurementGate(1, key="").on(c),
    )
    output_circuit = comp.add_moves(circuit=layout.mapped_circuit, layout=layout)
    cirq.testing.assert_has_diagram(
        output_circuit,
        str(expected_output_circuit),
    )


def test_replace_cirq_op_distil_t(bell_circuit) -> None:
    distillery_layout = MovementDistillery(
        bell_circuit, num_t_factories=2, num_ccz_factories=0, architecture="DSM"
    )

    op_to_replace = cirq.T.on(cirq.GridQubit(0, 0))
    ops_flattened = list(
        cirq.flatten_to_ops(
            comp.replace_cirq_op(
                op=op_to_replace,
                layout=distillery_layout,
                transversal_cnot=True,
            )
        )
    )
    expected_types: list[GateKey] = [
        lsp.Distil("T"),
        lsp.Distil("T"),
        cirq.CNOT,
        cirq.MeasurementGate,
        cirq.ResetChannel,
        cirq.S,
    ]
    assert len(expected_types) == len(ops_flattened)
    for op, expected_type in zip(ops_flattened, expected_types):
        assert op in cirq.GateFamily(expected_type)


def test_replace_cirq_op_distil_ccz(random_circ: cirq.Circuit) -> None:
    distillery_layout = MovementDistillery(
        random_circ, num_ccz_factories=2, num_t_factories=0, architecture="DSM"
    )

    op_to_replace = cirq.CCZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
    returned_ops = list(
        cirq.flatten_to_ops(
            comp.replace_cirq_op(op=op_to_replace, layout=distillery_layout, transversal_cnot=True)
        )
    )
    # We flatten them here to be explicit about the order the operations should be in
    ops_flattened = list(cirq.flatten_to_ops(returned_ops))
    expected_types: list[GateKey] = [
        *([lsp.Distil("CCZ")] * 2),
        *([cirq.CNOT] * 3),
        *([cirq.MeasurementGate] * 3),
        *([cirq.ResetChannel] * 3),
        *([cirq.H] * 3),
        *([cirq.X] * 3),
        *([cirq.CNOT] * 3),
        *([cirq.H] * 3),
    ]
    assert len(expected_types) == len(ops_flattened)
    for op, expected_type in zip(ops_flattened, expected_types):
        assert op in cirq.GateFamily(expected_type)


def test_different_rounds_distil() -> None:
    circuit = cirq.Circuit(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    layout = MovementDistillery(input_circuit=circuit, architecture="SSM")
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
            if isinstance(op.gate, lsp.SyndromeExtract):
                assert op.gate.rounds == k


def test_teleport_resource_exceptions():
    invalid_resource = cirq.CCZ.on(*(cirq.GridQubit(0, i) for i in range(3)))
    layout = MovementLayout(cirq.Circuit(), architecture="DSM")
    with pytest.raises(ValueError, match="Invalid resource"):
        _ = comp.teleport_resource(invalid_resource, layout)
    
    invalid_qubit = cirq.LineQubit(0)
    with pytest.raises(TypeError, match="Qubits must be instances"):
        _ = comp.teleport_resource(cirq.T.on(invalid_qubit), layout)

def test_exceptions(bell_circuit: cirq.Circuit):
    # Test ft compile rejects incompatible layout-architecture combos
    inplace_layout = MovementLayout(input_circuit=bell_circuit, architecture="DSM")
    zoned_arc = arch.DefaultMovement()
    with pytest.raises(ValueError, match="zone operations"):
        _ = comp.ft_compile(layout=inplace_layout, arc=zoned_arc)
