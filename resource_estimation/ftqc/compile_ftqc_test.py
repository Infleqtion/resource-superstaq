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
import textwrap
from math import pi

import cirq
import cirq_superstaq as css
import pytest

import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.codepatch as cpa
import resource_estimation.ftqc.compile_ftqc as comp
import resource_estimation.ftqc.lattice_surgery_primitives as lsp
import resource_estimation.ftqc.layout as lyt
import resource_estimation.ftqc.mappers as mps


def make_context(roles: dict[int, int], **overrides: object) -> mps.Context:
    pid_to_qid = {pid: cirq.NamedQubit(f"p{pid}") for pid in roles}
    values = dict(
        roles=dict(roles),
        pid_to_qid=pid_to_qid,
        qid_to_pid={v: k for k, v in pid_to_qid.items()},
        factories={},
        blocks={},
        num_s_factories=0,
        num_t_factories=0,
        num_ccz_factories=0,
        transversal_cnot=False,
        distil_t=False,
        cult_t=True,
        dynamic=False,
        post_op_syndrome_extraction=False,
        idling=False,
    )
    values.update(**overrides)
    return mps.Context(**values)


@pytest.mark.parametrize("verbose", (True, False))
def test_validate_ops(verbose) -> None:
    qubits = cirq.LineQubit.range(3)
    valid_circuit = cirq.Circuit(
        cirq.T.on(qubits[0]),
        cirq.X.on(qubits[1]),
        cirq.Z.on(qubits[2]),
        cirq.S.on(qubits[0]),
        cirq.H.on(qubits[1]),
        cirq.I.on(qubits[2]),
        cirq.CNOT.on(*qubits[:2]),
        cirq.CCZ.on(*qubits),
        cirq.measure(qubits[0]),
        cirq.reset(qubits[1]),
    )
    invalid_circuit = valid_circuit + cirq.rx(7).on(qubits[2])
    # Check valid circuit produces no errors
    comp.validate_ops(valid_circuit, verbose=verbose)
    with pytest.raises(
        ValueError, match=r"This compiler only handles Clifford \+ T \+ CCZ circuits"
    ):
        comp.validate_ops(invalid_circuit, verbose=verbose)


@pytest.mark.parametrize(
    "op, transversal_cnot, expected",
    [
        (cirq.S.on(cirq.LineQubit(0)), True, False),
        (cirq.S.on(cirq.LineQubit(0)), False, True),
        (cirq.T.on(cirq.LineQubit(0)), True, True),
        (cirq.CCZ.on(*cirq.LineQubit.range(3)), True, True),
    ],
)
def test_requires_resource(op: cirq.Operation, transversal_cnot: bool, expected: bool) -> None:
    context = make_context(roles={}, transversal_cnot=transversal_cnot)
    print(context.transversal_cnot, transversal_cnot)
    assert comp._requires_resource(op, context) == expected


@pytest.mark.parametrize(
    "op, expected_prep, expected_correct, expected_rid",
    [
        (
            cirq.CCZ.on(*cirq.LineQubit.range(3)),
            lsp.Distil("CCZ"),
            lsp.ResourceCorrection("CCZ"),
            5,
        ),
        (
            cirq.S.on(cirq.LineQubit(0)),
            lsp.Cultivate(pi / 2),
            cirq.Z,
            3,
        ),
        (cirq.T.on(cirq.LineQubit(0)), lsp.Cultivate(pi / 4), lsp.ResourceCorrection("T"), 4),
        (cirq.T.on(cirq.LineQubit(0)), lsp.Distil("T"), lsp.ResourceCorrection("T"), 4),
    ],
)
def test_determine_resource_dynamic(
    op: cirq.Operation,
    expected_prep: cirq.Operation,
    expected_correct: cirq.Operation,
    expected_rid: int,
) -> None:
    distil_t = expected_prep == lsp.Distil("T")
    context = make_context(roles={}, distil_t=distil_t, dynamic=True)
    prep_gate, correction, resource_id = comp._determine_resource(op, context)
    assert prep_gate == expected_prep
    assert correction == expected_correct
    assert resource_id == expected_rid


def test_determine_resource_ccz_static() -> None:
    # Lots of gates in this one so doing it explicitly
    context = make_context(roles={}, distil_t=False, dynamic=False)
    qubits = cirq.LineQubit.range(3)
    base_op = cirq.CCZ.on(*qubits)
    expected_prep = lsp.Distil("CCZ")
    expected_correct = [
        *cirq.H.on_each(*qubits),
        *cirq.X.on_each(*qubits),
        cirq.CNOT.on(qubits[0], qubits[1]),
        cirq.CNOT.on(qubits[0], qubits[2]),
        cirq.CNOT.on(qubits[1], qubits[2]),
        *cirq.H.on_each(*qubits),
    ]
    expected_rid = 5
    prep, correct, rid = comp._determine_resource(base_op, context)
    print(correct)
    assert prep == expected_prep
    assert correct == expected_correct
    assert rid == expected_rid


@pytest.mark.parametrize("distil_t", (True, False))
def test_determine_resource_t_static(distil_t: bool) -> None:
    context = make_context(roles={}, distil_t=distil_t, dynamic=False)
    base_op = cirq.T.on(cirq.LineQubit(0))
    expected_prep = lsp.Distil("T") if distil_t else lsp.Cultivate(pi / 4)
    expected_correct = cirq.S
    expected_rid = 4
    prep, correct, rid = comp._determine_resource(base_op, context)
    assert prep == expected_prep
    assert correct == expected_correct
    assert rid == expected_rid


def test_determine_resource_raises_error() -> None:
    context = make_context(roles={})
    invalid_op = cirq.rz(1.23).on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match="Invalid resource encountered"):
        _ = comp._determine_resource(invalid_op, context)


@pytest.mark.parametrize(
    "resource_id, num_qubits, expected",
    [
        (3, 1, {(1,), (2,)}),
        (4, 1, {(3,), (4,)}),
        (5, 3, {(5, 6, 7), (8, 9, 10)}),
        (-1, 1, set()),  # To raise error
    ],
)
def test_find_factories_simple(
    resource_id: int,
    num_qubits: int,
    expected: set[tuple[int, ...]],
) -> None:
    roles = {
        0: 1,  # Logical qubit
        1: 3,
        2: 3,  # S factory patches
        3: 4,
        4: 4,  # T factory patches
        **dict.fromkeys(range(5, 11), 5),  # CCZ factory patches
    }
    context = make_context(
        roles,
        num_t_factories=2,
        num_ccz_factories=2,
        num_s_factories=2,
    )

    if resource_id == -1:
        with pytest.raises(ValueError, match="Cannot parse resource id"):
            _ = comp._find_factories_simple(context, resource_id, num_qubits)
    else:
        result = comp._find_factories_simple(context, resource_id, num_qubits)
        assert result == expected


@pytest.mark.parametrize(
    "resource_id, block_size, expected",
    [
        (3, 1, {(20,), (21,)}),
        (4, 2, {(5, 6), (7, 8)}),
        (5, 3, {(14, 15, 16), (17, 18, 19)}),
        (-1, 1, set()),
    ],
)
def test_find_blocks_simple(
    resource_id: int,
    block_size: int,
    expected: set[tuple[int, ...]],
) -> None:
    roles = {
        0: 1,
        1: 3,
        2: 3,
        **dict.fromkeys(range(3, 11), 4),
        **dict.fromkeys(range(11, 20), 5),
        20: 3,
        21: 3,  # Available S blocks
    }
    context = make_context(
        roles,
        factories={
            (1,): 3,
            (2,): 3,
            (4,): 4,
            (11, 12, 13): 5,
        },
        blocks={(4,): (3, 9)},
        num_s_factories=2,
        num_t_factories=2,
        num_ccz_factories=2,
    )

    if resource_id == -1:
        with pytest.raises(ValueError, match="Cannot parse resource id"):
            _ = comp._find_blocks_simple(context, resource_id, block_size)
    else:
        result = comp._find_blocks_simple(context, resource_id, block_size)
        assert result == expected


@pytest.mark.parametrize(
    "distil_t, num_t_factories, num_ccz_factories",
    [
        (False, 2, 0),  # T cultivation
        (True, 2, 0),  # T distillation
        (False, 1, 1),  # T cultivation + CCZ
        (True, 1, 1),  # T distillation + CCZ
    ],
)
def test_allocate_qubits_simple(
    distil_t: bool,
    num_t_factories: int,
    num_ccz_factories: int,
) -> None:
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.I.on_each(*qubits))

    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=2,
        surface_code_distance=3,
        t_cultivation_factories=0 if distil_t else num_t_factories,
        t_distillation_factories=num_t_factories if distil_t else 0,
        ccz_distillation_factories=num_ccz_factories,
        interaction_zone=False,
        measure_zone=False,
    )

    qid_to_pid, roles = comp.allocate_qubits_simple(
        circuit, layout, distil_t, num_t_factories, num_ccz_factories
    )

    num_t_patches = num_t_factories * (31 if distil_t else 1)
    ccz_start = 2 + num_t_patches

    expected_roles = {
        0: 1,
        1: 1,
        **dict.fromkeys(range(2, ccz_start), 4),
        **dict.fromkeys(range(ccz_start, ccz_start + 23 * num_ccz_factories), 5),
    }

    assert roles == expected_roles
    assert set(qid_to_pid[q] for q in qubits) == {0, 1}
    assert set(qid_to_pid.values()) == set(expected_roles)

    # Remove one patch to induce error
    layout.remove(1)
    with pytest.raises(ValueError, match="Layout does not have enough patches"):
        _ = comp.allocate_qubits_simple(
            circuit, layout, distil_t, num_t_factories, num_ccz_factories
        )


def test_nearest_factory() -> None:
    layout = lyt.Layout(
        rows=100,
        cols=100,
        placements={
            1: lyt.Placement(cpa.RotatedSurfaceCodePatch(0, 3), row=0, col=0),
            2: lyt.Placement(cpa.RotatedSurfaceCodePatch(1, 3), row=10, col=0),
            3: lyt.Placement(cpa.RotatedSurfaceCodePatch(2, 3), row=10, col=10),
        },
    )
    source_factory = (cirq.NamedQubit("p0"),)
    # Most basic example
    context = make_context(roles={0: 1, 1: 4, 2: 4}, factories={(1,): 4, (2,): 4})
    expected_nearest_factory = (1,)
    assert comp._nearest_factory(source_factory, 4, layout, context) == expected_nearest_factory
    # Check that removing closest factory changes answer
    context = make_context(roles={0: 1, 1: 4, 2: 4}, factories={(2,): 4})
    expected_nearest_factory = (2,)
    assert comp._nearest_factory(source_factory, 4, layout, context) == expected_nearest_factory
    # Raises error when factory is unavailable
    with pytest.raises(ValueError, match="No available factories"):
        _ = comp._nearest_factory(source_factory, 5, layout, context)
    # Make sure closest does not change in context of other factories
    context = make_context(roles={0: 1, 1: 5, 2: 4}, factories={(1,): 5, (2,): 4})
    assert comp._nearest_factory(source_factory, 4, layout, context) == (2,)


def test_teleport_resource() -> None:
    # Use the generator in layout to easily generate layouts that can be interrogated
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=5,
        surface_code_distance=5,
        t_cultivation_factories=3,
        t_distillation_factories=0,
        ccz_distillation_factories=2,
        interaction_zone=False,
        measure_zone=False,
    )
    # Start with factories initially unassigned
    context = make_context(
        roles={
            **dict.fromkeys(range(5), 1),
            **dict.fromkeys(range(5, 8), 4),
            **dict.fromkeys(range(8, 54), 5),
        },
        num_t_factories=3,
        num_ccz_factories=2,
        dynamic=True,
    )
    qubits = tuple(context.pid_to_qid[i] for i in range(3))
    op = cirq.CCZ.on(*qubits)

    result = comp.teleport_resource(op, layout, context)

    # Check the returned operations
    assert len(result) == 5
    assert result[0].gate == lsp.Distil("CCZ")
    assert len(result[0].qubits) == 23

    factory_qubits = tuple(context.pid_to_qid[i] for i in (8, 9, 10))

    assert result[1] == cirq.Moment(cirq.CNOT.on(f, q) for f, q in zip(factory_qubits, qubits))
    assert result[2] == cirq.Moment(cirq.measure(q, key="") for q in factory_qubits)
    assert result[3] == cirq.Moment(cirq.reset(q) for q in factory_qubits)
    assert result[4] == lsp.ResourceCorrection("CCZ").on(*qubits)

    # Check that the consumed resources are no longer available
    assert (8, 9, 10) not in context.factories
    assert len(context.factories) == 1
    assert len(context.blocks) == 1

    second_result = comp.teleport_resource(op, layout, context)

    # The second teleportation uses a different factory
    assert second_result[0].qubits[:3] != result[0].qubits[:3]

    # Both CCZ factories and their blocks have been consumed
    assert not context.factories
    assert not context.blocks

    # Remove all but three qubits to induce error for underresourced blocks
    for pid in range(11, 54):
        context.roles[pid] = 0
    with pytest.raises(ValueError, match="Not enough resource blocks available"):
        _ = comp.teleport_resource(op, layout, context)


@pytest.mark.parametrize(
    "interaction_zone, measure_zone", [(False, False), (False, True), (True, False), (True, True)]
)
def test_add_zones_to_context(interaction_zone: bool, measure_zone: bool) -> None:
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=5,
        surface_code_distance=5,
        t_cultivation_factories=3,
        t_distillation_factories=0,
        ccz_distillation_factories=2,
        interaction_zone=interaction_zone,
        measure_zone=measure_zone,
    )
    context = make_context(
        roles={
            **dict.fromkeys(range(5), 1),
            **dict.fromkeys(range(5, 8), 4),
            **dict.fromkeys(range(8, 54), 5),
        },
        num_t_factories=3,
        num_ccz_factories=2,
        dynamic=True,
    )
    original_roles = context.roles.copy()

    new_context = comp.add_zones_to_context(
        context=context,
        arc=arch.DefaultMovement(d=5),
        layout=layout,
    )

    expected_roles = original_roles.copy()
    next_pid = 54

    if interaction_zone:
        expected_roles.update(dict.fromkeys(range(next_pid, next_pid + 8), 6))
        next_pid += 8

    if measure_zone:
        expected_roles.update(dict.fromkeys(range(next_pid, next_pid + 8), 7))

    assert new_context.roles == expected_roles
    assert context.roles == original_roles

    # Verify the new qubit mappings
    assert set(new_context.pid_to_qid) == set(expected_roles)
    for pid, qid in new_context.pid_to_qid.items():
        assert new_context.qid_to_pid[qid] == pid


@pytest.mark.parametrize("verbose", (False, True))
def test_post_op_syndrome_extraction(verbose: bool) -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT.on(qubits[0], qubits[1]),
        cirq.X.on(qubits[2]),
        cirq.S.on(qubits[0]),
        cirq.H.on(qubits[1]),  # Only receives SE for movement
        cirq.CNOT.on(qubits[1], qubits[2]),
    )
    movement_architecture = arch.DefaultMovement()
    lattice_architecture = arch.DefaultLattice()

    movement_replacement = comp.post_op_syndrome_extraction(
        circuit, movement_architecture, verbose=verbose
    )
    cirq.testing.assert_has_diagram(
        movement_replacement,
        textwrap.dedent(
            """
            0: ───@───SE(1)───S───SE(1)───────────────
                  │
            1: ───X───SE(1)───H───SE(1)───@───SE(1)───
                                          │
            2: ───X───────────────────────X───SE(1)───
            """
        ),
    )

    # This is actually an unfeasible circuit for lattice surgery, but it shows the purpose of the pass
    lattice_replacement = comp.post_op_syndrome_extraction(
        circuit, lattice_architecture, verbose=verbose
    )
    cirq.testing.assert_has_diagram(
        lattice_replacement,
        textwrap.dedent(
            """
            0: ───@───SE(7)───S───SE(7)───────────────
                  │
            1: ───X───SE(7)───H───────────@───SE(7)───
                                          │
            2: ───X───────────────────────X───SE(7)───
            """
        ),
    )


@pytest.mark.parametrize("verbose", (True, False))
def test_handle_idling(verbose: bool) -> None:
    p0, p1, anc = cirq.NamedQubit("p0"), cirq.NamedQubit("p1"), cirq.NamedQubit("p2")

    context = make_context(roles={0: 1, 1: 1, 2: 2})
    arc = arch.DefaultLattice(d=5, syndrome_rounds=2)

    se = lsp.SyndromeExtract(1, 2)
    split = lsp.Split(partitions=[1, 1], smooth=True).on(p0, anc)

    circuit = cirq.Circuit(
        cirq.Moment([se.on(p0)]),  # Initial state prep
        cirq.Moment([cirq.X(p0)]),  # p1 is idle
        cirq.Moment([split]),  # Split-only moment
        cirq.Moment(se.on_each(p0, p1)),  # All logical qubits receive SE
        cirq.Moment([cirq.X(p1)]),  # p0 is idle
        cirq.Moment([cirq.X(p0), cirq.X(p1)]),  # No logical qubits idle
        cirq.Moment([cirq.X(anc)]),  # Ancilla active, both logical qubits idle
    )

    expected = cirq.Circuit(
        cirq.Moment([se.on(p0)]),
        cirq.Moment([cirq.X(p0), se.on(p1)]),
        cirq.Moment([split]),
        cirq.Moment(se.on_each(p0, p1)),
        cirq.Moment([cirq.X(p1), se.on(p0)]),
        cirq.Moment([cirq.X(p0), cirq.X(p1)]),
        cirq.Moment([cirq.X(anc), se.on(p0), se.on(p1)]),
    )

    result = comp.handle_idling(circuit, arc, context, verbose=verbose)
    assert result == expected


def test_route_and_decompose_lattice_cnot() -> None:
    # Patch positions in grid coordinates: (row, col)
    positions = {
        0: (0, 2),  # Control
        1: (0, 1),  # Routing ancilla
        2: (0, 0),  # Target
        3: (1, 0),  # Farther option
        4: (1, 1),
    }

    distance = 3
    patch_size = 2 * distance - 1

    layout = lyt.Layout(
        rows=2 * patch_size,
        cols=3 * patch_size,
        placements={
            pid: lyt.Placement(
                cpa.RotatedSurfaceCodePatch(pid, distance),
                row=r * patch_size,
                col=c * patch_size,
            )
            for pid, (r, c) in positions.items()
        },
    )
    context = make_context(roles={0: 1, 1: 2, 2: 1, 3: 1, 4: 2})

    best_route = comp.route_lattice_cnot(
        context.pid_to_qid[0], context.pid_to_qid[2], layout, context
    )
    assert best_route == [2, 1, 0]  # Does not take longer route
    # Change role and check that route no longer exists
    context.roles[1] = 1
    with pytest.raises(ValueError, match="Could not find valid path"):
        _ = comp.route_lattice_cnot(context.pid_to_qid[0], context.pid_to_qid[2], layout, context)

    # Cover the decomposition while we are here
    context.roles[1] = 2  # Fix the overwritten role
    q0, q1, q2 = (context.pid_to_qid[i] for i in range(3))

    result = comp._decompose_lattice_cnot(q0, q2, layout, context)

    expected = [
        lsp.Merge(num_qubits=2, smooth=True).on(q2, q1),
        lsp.Split(partitions=[1, 1], smooth=True).on(q2, q1),
        lsp.Merge(num_qubits=2, smooth=False).on(q1, q0),
        lsp.Split(partitions=[1, 1], smooth=False).on(q1, q0),
    ]

    assert result == expected

    # Also test _decompose_to_primitives while here
    decomposed = comp._decompose_to_primitives(
        circuit=cirq.Circuit(cirq.CNOT(q0, q2)),
        layout=layout,
        arc=arch.DefaultLattice(d=distance),
        context=context,
    )
    assert list(decomposed.all_operations()) == expected


@pytest.mark.parametrize("verbose", (True, False))
@pytest.mark.parametrize(
    "interaction_zone, measure_zone", [(False, False), (False, True), (True, False), (True, True)]
)
def test_add_moves(interaction_zone: bool, measure_zone: bool, verbose: bool) -> None:
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=3,
        surface_code_distance=5,
        t_cultivation_factories=0,
        t_distillation_factories=0,
        ccz_distillation_factories=0,
        interaction_zone=interaction_zone,
        measure_zone=measure_zone,
    )
    context = make_context(roles=dict.fromkeys(range(3), 1))

    context = comp.add_zones_to_context(
        context=context,
        arc=arch.DefaultMovement(d=5),
        layout=layout,
    )
    qubits = tuple(context.pid_to_qid[i] for i in range(3))
    circuit = cirq.Circuit(
        cirq.X.on(qubits[0]),
        cirq.CNOT.on(qubits[0], qubits[1]),
        cirq.CNOT.on(qubits[1], qubits[2]),
        cirq.measure(qubits[2], key=""),
    )
    result = comp.add_moves(circuit, context, layout, verbose=verbose)

    interaction_qids = [context.pid_to_qid[pid] for pid, role in context.roles.items() if role == 6]
    measurement_qids = [context.pid_to_qid[pid] for pid, role in context.roles.items() if role == 7]
    move = css.MovementGate({0: 1})
    move_dg = css.MovementGate({1: 0})

    expected = cirq.Circuit(cirq.X(qubits[0]))

    for ctrl, trgt in [(qubits[0], qubits[1]), (qubits[1], qubits[2])]:
        if interaction_zone:
            zone = interaction_qids.pop(0)
            expected.append(
                [
                    move.on(ctrl, zone),
                    move.on(trgt, zone),
                    cirq.CNOT(ctrl, trgt),
                    move_dg.on(trgt, zone),
                    move_dg.on(ctrl, zone),
                ]
            )
        else:
            expected.append(
                [
                    move.on(ctrl, trgt),
                    cirq.CNOT(ctrl, trgt),
                    move_dg.on(ctrl, trgt),
                ]
            )

    if measure_zone:
        zone = measurement_qids[0]
        expected.append(move.on(qubits[2], zone))

    expected.append(cirq.measure(qubits[2], key=""))

    if measure_zone:
        expected.append(move_dg.on(qubits[2], zone))

    assert list(result.all_operations()) == list(expected.all_operations())


def test_replace_cirq_op() -> None:
    # TODO: consider more thorough injections if not covered elsewhere
    base_circuit = cirq.Circuit(cirq.I.on_each(*cirq.LineQubit.range(2)))
    # replace a CNOT in a column layout
    lattice_layout = lyt.generate_column_layout(
        surface_code_program_qubits=2, surface_code_distance=3
    )
    lattice_context = mps.build_column_context(
        base_circuit, lattice_layout, surface_code_distance=3
    )
    ctrl = cirq.NamedQubit("1-0-4")
    trgt = cirq.LineQubit(1)
    op_to_replace = cirq.CNOT.on(ctrl, trgt)
    decomposition = comp.replace_cirq_op(op_to_replace, lattice_layout, lattice_context)
    assert len(decomposition) == 4
    assert [type(op.gate) for op in decomposition] == [
        lsp.Merge,
        lsp.Split,
        lsp.Merge,
        lsp.Split,
    ]

    # Replace a T gate with an injected T state in a movement layout
    movement_layout = lyt.generate_movement_layout(
        surface_code_program_qubits=2,
        surface_code_distance=3,
        t_cultivation_factories=1,
        t_distillation_factories=0,
        ccz_distillation_factories=0,
        interaction_zone=False,
        measure_zone=False,
    )
    movement_context = mps.build_movement_context(
        base_circuit,
        movement_layout,
        t_cultivation_factories=1,
        t_distillation_factories=0,
        ccz_distillation_factories=0,
        dynamic=False,
    )
    logical_pid = next(pid for pid, role in movement_context.roles.items() if role == 1)
    t_qubit = movement_context.pid_to_qid[logical_pid]
    op_to_replace = cirq.T.on(t_qubit)

    decomposition = comp.replace_cirq_op(op_to_replace, movement_layout, movement_context)
    assert len(decomposition) == 5
    assert decomposition[0].gate == lsp.Cultivate(pi / 4)
    assert decomposition[-1] == cirq.S.on(t_qubit)

    # Test that some operations cannot be replaced
    with pytest.raises(ValueError, match="Invalid Op for the given layout and context"):
        comp.replace_cirq_op(
            cirq.TOFFOLI.on(*cirq.LineQubit.range(3)),
            movement_layout,
            movement_context,
        )


@pytest.mark.parametrize("movement", (False, True))
def test_ft_compile(movement: bool) -> None:
    # Simple integration test
    # Core mechanisms are covered elsewhere
    distance = 3
    if movement:
        arc = arch.DefaultMovement(d=distance, syndrome_rounds=1)
        layout = lyt.generate_movement_layout(
            surface_code_program_qubits=1,
            surface_code_distance=distance,
            t_cultivation_factories=0,
            t_distillation_factories=0,
            ccz_distillation_factories=0,
            interaction_zone=False,
            measure_zone=False,
        )
    else:
        arc = arch.DefaultLattice(d=distance, syndrome_rounds=1)
        layout = lyt.generate_column_layout(
            surface_code_program_qubits=1,
            surface_code_distance=distance,
        )

    context = make_context(
        roles={0: 1},
        transversal_cnot=movement,
        post_op_syndrome_extraction=movement,
        idling=movement,
    )

    q0 = context.pid_to_qid[0]
    circuit = cirq.Circuit(cirq.X(q0))

    compiled, new_context = comp.ft_compile(
        circuit=circuit,
        arc=arc,
        layout=layout,
        context=context,
    )

    expected = cirq.Circuit(
        lsp.SyndromeExtract(1, 1).on(q0),
        cirq.X(q0),
    )

    assert compiled == expected
    assert new_context.roles == context.roles


# @pytest.fixture
# def bell_circuit() -> cirq.Circuit:
#     qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
#     circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b)])
#     return circuit


# @pytest.fixture
# def t_circuit() -> cirq.Circuit:
#     qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
#     circuit = cirq.Circuit([cirq.H.on(qubit_a), cirq.CNOT.on(qubit_a, qubit_b), cirq.T.on(qubit_b)])
#     return circuit


# @pytest.fixture
# def random_circ() -> cirq.Circuit:
#     return cirq.testing.random_circuit(
#         qubits=5,
#         n_moments=8,
#         op_density=1,
#         gate_domain={cirq.H: 1, cirq.CNOT: 2, cirq.T: 1, cirq.S: 1},
#         random_state=73,
#     )


# @pytest.fixture
# def random_circ2() -> cirq.Circuit:
#     return cirq.testing.random_circuit(
#         qubits=4,
#         n_moments=7,
#         op_density=0.8,
#         gate_domain={cirq.H: 1, cirq.CNOT: 2, cirq.T: 1, cirq.S: 1},
#         random_state=73,
#     )

# @pytest.fixture
# def lattice_context() -> comp.Context:
#     return comp.Context(
#         roles={},
#         qid_to_pid={},
#         pid_to_qid={},
#         factories={},
#         transversal_cnot=False,
#         distil_t=False,
#         cult_t=True,
#         dynamic=False,
#     )
# @pytest.fixture
# def movement_context() -> comp.Context:
#     return comp.Context(
#         roles={},
#         qid_to_pid={},
#         pid_to_qid={},
#         factories={},
#         transversal_cnot=True,
#         distil_t=False,
#         cult_t=True,
#         dynamic=False,
#     )

# @pytest.mark.parametrize(
#     "with_barriers",
#     (True, False),
# )
# def test_end2end(with_barriers: bool) -> None:
#     # Circuit that tests all uses all possible gates
#     q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(2, 2)
#     circuit = cirq.Circuit(
#         [
#             cirq.H.on(q0),
#             cirq.CNOT.on(q0, q1),
#             cirq.T.on(q1),
#             cirq.H.on(q0),
#             cirq.CNOT.on(q0, q1),
#             cirq.T.on(q1),
#             cirq.X.on(q0),
#             cirq.Z.on(q1),
#             cirq.S.on(q1),
#             cirq.I.on_each(q0, q1),
#             cirq.MeasurementGate(2, key="end").on(q0, q1),
#         ],
#     )
#     test_layout: MovementLayout | Column
#     for arc in [
#         arch.DefaultLattice(idling=False, post_op_correction=True),
#         arch.DefaultLattice(idling=True, post_op_correction=True),
#         arch.DefaultMovement(idling=False, post_op_correction=True),
#         arch.DefaultMovement(idling=True, post_op_correction=True),
#         arch.DefaultLattice(idling=False, post_op_correction=False),
#         arch.DefaultMovement(idling=False, post_op_correction=False),
#     ]:
#         if arc.movement:
#             test_layout = MovementLayout(
#                 input_circuit=circuit, num_t_factories=1, architecture="SSM"
#             )
#         else:
#             test_layout = Column(
#                 input_circuit=circuit,
#             )
#         compiled = comp.ft_compile(test_layout, arc, with_barriers=with_barriers)
#         for op in compiled.all_operations():
#             is_primitive = False
#             if arc.primitives.validate(op) or op in cirq.GateFamily(css.Barrier):
#                 is_primitive = True
#             assert is_primitive


# def test_end2end_distillery() -> None:
#     q1, q2, q3 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
#     circuit = cirq.Circuit(
#         [cirq.CNOT.on(q1, q2), cirq.CCZ.on(q1, q2, q3), cirq.T.on_each(q1, q2, q3)]
#     )
#     layout = MovementDistillery(
#         input_circuit=circuit, num_t_factories=1, num_ccz_factories=1, architecture="SSM"
#     )
#     arc = arch.DefaultMovement(post_op_correction=False, idling=False)
#     compiled = comp.ft_compile(layout, arc, with_barriers=False)
#     assert all(arc.primitives.validate(op) for op in compiled.all_operations())


# def test_direct_substitution(lattice_context: comp.Context, movement_context: comp.Context) -> None:
#     dummy_qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
#     nothing_circuit = cirq.Circuit(cirq.I.on_each(dummy_qubits))
#     lattice_layout = Embedded(input_circuit=nothing_circuit)
#     movement_layout = MovementLayout(input_circuit=nothing_circuit)

#     # Test primitives that are the same between movement and no movement
#     for arc, layout in [(arch.DefaultMovement(), movement_layout), (arch.DefaultLattice(), lattice_layout)]:
#         for op_to_replace in [
#             cirq.I.on(dummy_qubits[0]),
#             cirq.H.on(dummy_qubits[0]),
#             cirq.X.on(dummy_qubits[0]),
#             cirq.Z.on(dummy_qubits[0]),
#             cirq.MeasurementGate(1).on(dummy_qubits[0]),
#             cirq.ResetChannel().on(dummy_qubits[0]),
#             lsp.Cultivate(pi / 4).on(dummy_qubits[0]),
#             lsp.SyndromeExtract(1, 1).on(dummy_qubits[0]),
#             lsp.ErrorCorrect(1).on(dummy_qubits[0]),
#         ]:
#             replacement = comp._decompose_to_primitives(
#                 circuit=cirq.Circuit(op_to_replace),
#                 layout=layout,
#                 arc=arc,
#                 context=comp.contextualize(layout, arc)
#             )
#             assert replacement == cirq.Circuit(op_to_replace)

#     # Test primitives that are reserved for no movement
#     for op_to_replace in [
#         lsp.Merge(2).on(*dummy_qubits[:2]),
#         lsp.Split([1, 1]).on(*dummy_qubits[:2]),
#     ]:
#         replacement = comp._decompose_to_primitives(
#             circuit=cirq.Circuit(op_to_replace),
#             layout=lattice_layout,
#             arc=arch.DefaultLattice(),
#             context=lattice_context,
#         )
#         assert replacement == cirq.Circuit(op_to_replace)

#     # Test primitives that are reserved for movement
#     for op_to_replace in [
#         cirq.CNOT.on(*dummy_qubits[:2]),
#         cirq.S.on(dummy_qubits[0]),
#         lsp.Distil("T").on(*cirq.LineQubit.range(31)),
#     ]:
#         replacement = comp._decompose_to_primitives(
#             circuit=cirq.Circuit(op_to_replace),
#             layout=layout,
#             arc=arch.DefaultMovement(),
#             context=movement_context,
#         )
#         assert replacement == cirq.Circuit(op_to_replace)

#     # Test unrecognized gate
#     with pytest.raises(ValueError, match="Invalid Op for Layout and Context: Rx"):
#         _ = comp.replace_cirq_op(
#             op=cirq.Rx(rads=pi / 2).on(dummy_qubits[0]),
#             layout=layout,
#             context=lattice_context,
#         )

#     # Test TypeError is raised
#     with pytest.raises(TypeError, match="Qubits must be instances"):
#         _ = comp.replace_cirq_op(
#             cirq.CNOT.on(*cirq.LineQubit.range(2)), layout=layout, context=lattice_context
#         )


# def test_replace_cirq_op_movement(bell_circuit: cirq.Circuit, movement_context: comp.Context) -> None:
#     movement_layout = MovementLayout(bell_circuit, num_t_factories=2, architecture="DSM")

#     op_to_replace = cirq.T.on(cirq.GridQubit(0, 0))
#     returned_ops = comp.replace_cirq_op(
#         op=op_to_replace,
#         layout=movement_layout,
#         context=movement_context,
#     )
#     ops_flattened = list(cirq.flatten_to_ops(returned_ops))
#     expected_types: list[GateKey] = [
#         lsp.Cultivate,
#         lsp.Cultivate,
#         cirq.CNOT,
#         cirq.MeasurementGate,
#         cirq.ResetChannel,
#         cirq.S,
#     ]
#     assert len(expected_types) == len(ops_flattened)
#     for op, expected_type in zip(ops_flattened, expected_types):
#         assert op in cirq.GateFamily(expected_type)


# @pytest.mark.parametrize("op_type", (cirq.S, cirq.T, cirq.CNOT))
# def test_replace_cirq_op_lattice(op_type: cirq.Gate, bell_circuit: cirq.Circuit, lattice_context: comp.Context) -> None:
#     layout = Column(bell_circuit)

#     op_to_replace = op_type.on(*list(layout.mapped_circuit.all_qubits())[: op_type.num_qubits()])
#     returned_ops = comp.replace_cirq_op(op=op_to_replace, layout=layout, context=lattice_context)
#     ops_flattened = list(cirq.flatten_to_ops(returned_ops))
#     expected_types: list[GateKey]
#     if op_type == cirq.S:
#         expected_types = [
#             lsp.Cultivate,
#             lsp.Cultivate,
#             cirq.CNOT,
#             cirq.MeasurementGate,
#             cirq.ResetChannel,
#             cirq.Z,
#         ]
#     elif op_type == cirq.T:
#         expected_types = [
#             lsp.Cultivate,
#             lsp.Cultivate,
#             cirq.CNOT,
#             cirq.MeasurementGate,
#             cirq.ResetChannel,
#             cirq.S,
#         ]
#     elif op_type == cirq.CNOT:
#         expected_types = [lsp.Merge, lsp.Split, lsp.Merge, lsp.Split]
#     assert len(expected_types) == len(ops_flattened)
#     for op, expected_type in zip(ops_flattened, expected_types):
#         assert op in cirq.GateFamily(expected_type)


# @pytest.mark.parametrize(
#     "arc",
#     [
#         arch.DefaultLattice(idling=False, post_op_correction=False),
#         arch.DefaultMovement(idling=False, post_op_correction=True),
#     ],
# )
# def test_illegal_compile(arc: arch.Architecture) -> None:
#     # Test illegal gates
#     circuit = cirq.Circuit([cirq.Rx(rads=pi / 3).on(cirq.GridQubit(0, 0))])
#     layout: MovementLayout | Column
#     if arc.movement:
#         layout = MovementLayout(circuit, num_t_factories=1, architecture="SSM")
#     else:
#         layout = Column(circuit)
#     with pytest.raises(ValueError, match="This compiler only handles"):
#         _ = comp.ft_compile(layout=layout, arc=arc)


# def test_different_rounds() -> None:
#     circuit = cirq.Circuit(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
#     layout = MovementLayout(input_circuit=circuit, architecture="SSM")
#     for k in [1, 5, 7]:
#         architecture = arch.DefaultMovement(
#             idling=False,
#             post_op_correction=True,
#             d=7,
#             cultivation_repetition=1,
#             syndrome_rounds=k,
#         )
#         compiled_circuit = comp.ft_compile(layout=layout, arc=architecture)
#         for op in compiled_circuit.all_operations():
#             op = arch._require_gate_operation(op)
#             gate = op.gate
#             if isinstance(gate, lsp.SyndromeExtract):
#                 assert gate.rounds == k


# def test_deterministic_compilation(random_circ: cirq.Circuit) -> None:
#     circuit = random_circ
#     lay = Column(circuit)
#     arc = arch.DefaultLattice()
#     compiled1 = comp.ft_compile(lay, arc)
#     compiled2 = comp.ft_compile(lay, arc)
#     cirq.testing.assert_has_diagram(compiled1, str(compiled2))


# def test_nondeterministic_compilation_T(random_circ2: cirq.Circuit) -> None:
#     # Overried context
#     circuit = random_circ2
#     lay = MovementDistillery(circuit, num_t_factories=1, architecture="SSM")
#     arc = arch.DefaultMovement()
#     compiled1 = comp.ft_compile(lay, arc, dynamic=False)
#     compiled2 = comp.ft_compile(lay, arc, dynamic=True)
#     # We expect basically the same operations except that there are ResourceCorrection gates instead
#     # of S gates
#     # Hoping that this is deterministic
#     compiled1_ops = list(compiled1.all_operations())
#     compiled2_ops = list(compiled2.all_operations())
#     assert len(compiled1_ops) == len(compiled2_ops)
#     for i, (op1, op2) in enumerate(zip(compiled1_ops, compiled2_ops)):
#         op1 = _require_gate_operation(op1)
#         op2 = _require_gate_operation(op2)
#         if op1.gate != op2.gate:
#             assert isinstance(op1.gate, cirq.ZPowGate)
#             assert isinstance(op2.gate, lsp.ResourceCorrection)
#             assert op2.gate.resource == "T"


# def test_nondeterministic_compilation_CCZ() -> None:
#     circuit = cirq.Circuit(
#         cirq.CCZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
#     )
#     lay = MovementDistillery(circuit, num_t_factories=0, num_ccz_factories=1, architecture="SSM")
#     arc = arch.DefaultMovement()
#     compiled1 = comp.ft_compile(lay, arc, dynamic=False)
#     compiled2 = comp.ft_compile(lay, arc, dynamic=True)
#     # We expect the same operations on the CCZ qubits until there is a ResourceCorrection gate
#     compiled1_ops = list(compiled1.all_operations())
#     compiled2_ops = list(compiled2.all_operations())

#     def relevant_op(op: cirq.Operation) -> bool:
#         qubits = op.qubits
#         if (
#             cirq.GridQubit(0, 0) in qubits
#             or cirq.GridQubit(0, 1) in qubits
#             or cirq.GridQubit(0, 2) in qubits
#         ):
#             return True
#         return False

#     reached_correction = False
#     moments1 = list(compiled1.moments)
#     moments2 = list(compiled2.moments)
#     correction_circuit = cirq.Circuit()
#     for i in range(0, len(moments1)):
#         # The circuits should be identical until we reach the correction part
#         if not reached_correction:
#             try:
#                 assert len(moments1[i].operations) == len(moments2[i].operations)
#             except:
#                 reached_correction = True
#                 for op in moments1[i].operations:
#                     if relevant_op(op):
#                         assert isinstance(op.gate, cirq.HPowGate)
#                 for op in moments2[i].operations:
#                     if relevant_op(op):
#                         assert isinstance(op.gate, lsp.ResourceCorrection)
#                         assert op.gate.resource == "CCZ"

#         if not reached_correction:
#             for j in range(0, len(moments2[i].operations)):
#                 assert moments1[i].operations[j] == moments2[i].operations[j]
#         else:
#             # Once we reach the correction part, we want to make sure compilation does exactly what
#             # we expect on the logical qubits
#             for op in moments1[i].operations:
#                 if relevant_op(op):
#                     correction_circuit.append(op)
#     # If this errors, look at the correction circuit to make sure it's not just that the CNOT order
#     # changed or the qubit indices changes
#     cirq.testing.assert_has_diagram(
#         correction_circuit,
#         textwrap.dedent(
#             """
#                                                                             ┌───────────┐                                           ┌───────────┐
#         (-1, 3): ───────────────────┤   0├───┤   0├───────┤1   ├───┤1   ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

#         (-1, 4): ───────────────────────────────────────────────────────────────────────────┤   0├───┤   0├───────┤1   ├───┤1   ├─────────────────────────────────────────────────────────────────

#         (-1, 5): ─────────────────────────────────────────────────────────────────┤   0├──────────────────────────────────────────────────┤   0├────────┤1   ├───┤1   ├───────────────────────────

#         (0, 0): ────H───SE(1)───X───┤0   ├────────────@────────────┤   1├────SE(1)──────────┤0   ├────────────@────────────┤   1├────SE(1)──────────H───SE(1)─────────────────────────────────────
#                                                       │                                                       │
#         (0, 1): ────H───SE(1)───X────────────┤0   ├───X───┤   1├───SE(1)──────────┤0   ├──────────────────────┼─────────────────────────────────────@────────────┤   1├───SE(1)───H───────SE(1)───
#                                                                                                               │                                     │
#         (0, 2): ────H───SE(1)───X────────────────────────────────────────────────────────────────────┤0   ├───X───┤   1├───SE(1)──────────┤0   ├────X───┤   1├───SE(1)────H───────SE(1)───────────
#                                                                             └───────────┘                                           └───────────┘
#         """
#         ),
#     )


# def test_other_passes(random_circ: cirq.Circuit) -> None:
#     # If this test and test_deterministic_compilation both fail, that one likely causes the issue in this one
#     circuit = random_circ
#     lay = Column(circuit)
#     arc = arch.DefaultLattice(idling=True, post_op_correction=True)
#     compiled_circuit = comp.ft_compile(lay, arc)
#     idling_corrected_resources = dict(
#         collections.Counter(
#             str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
#             for op in compiled_circuit.all_operations()
#         ),
#     )
#     arc = arch.DefaultLattice(idling=False, post_op_correction=True)
#     compiled_circuit = comp.ft_compile(lay, arc)
#     corrected_resources = dict(
#         collections.Counter(
#             str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
#             for op in compiled_circuit.all_operations()
#         ),
#     )
#     arc = arch.DefaultLattice(idling=False, post_op_correction=False)
#     compiled_circuit = comp.ft_compile(lay, arc)
#     uncorrected_resources = dict(
#         collections.Counter(
#             str(op.gate) if op not in cirq.GateFamily(cirq.MeasurementGate) else "Measure"
#             for op in compiled_circuit.all_operations()
#         ),
#     )
#     assert (
#         idling_corrected_resources["MERGE"]
#         == corrected_resources["MERGE"]
#         == uncorrected_resources["MERGE"]
#     )
#     assert (
#         idling_corrected_resources["SPLIT"]
#         == corrected_resources["SPLIT"]
#         == uncorrected_resources["SPLIT"]
#     )
#     assert (
#         idling_corrected_resources["CULT(0.785)"]
#         == corrected_resources["CULT(0.785)"]
#         == uncorrected_resources["CULT(0.785)"]
#     )
#     assert (
#         idling_corrected_resources["CULT(1.571)"]
#         == corrected_resources["CULT(1.571)"]
#         == uncorrected_resources["CULT(1.571)"]
#     )
#     assert idling_corrected_resources["H"] == corrected_resources["H"] == uncorrected_resources["H"]
#     assert (
#         idling_corrected_resources["Measure"]
#         == corrected_resources["Measure"]
#         == uncorrected_resources["Measure"]
#     )
#     assert (
#         idling_corrected_resources["reset"]
#         == corrected_resources["reset"]
#         == uncorrected_resources["reset"]
#     )
#     assert idling_corrected_resources["Z"] == corrected_resources["Z"] == uncorrected_resources["Z"]
#     assert (
#         idling_corrected_resources["SE(7)"]
#         >= corrected_resources["SE(7)"]
#         >= uncorrected_resources["SE(7)"]
#     )


# def test_t_movement(t_circuit: cirq.Circuit) -> None:
#     movement_layout = MovementLayout(t_circuit, num_t_factories=2, architecture="SSM")
#     movement_architecture = arch.MeasureZonesOnly(
#         d=7,
#         cultivation_repetition=1,
#         syndrome_rounds=1,
#         idling=True,
#         post_op_correction=True,
#     )
#     compiled_t_circuit = comp.ft_compile(layout=movement_layout, arc=movement_architecture)
#     # yes idling, yes post-op correction
#     compiled_t_circuit = cirq.align_left(compiled_t_circuit)
#     # This test was updated both by aligning left and to reflect the change to make cultivation happen later in the circuit,
#     # The old version is left commented out below
#     cirq.testing.assert_has_diagram(
#         compiled_t_circuit,
#         textwrap.dedent(
#             """
#                                                               ┌────────────┐                             ┌───────────┐
#            (-1, 0): ─────────────────────────────────┤   0├────┤   0├──────────────────┤1   ├───┤1   ├───────────────────────────────────────────────────────────────────────────────────────────

#            (-1, 1): ─────────────────────────────────────────────────┤   0├────────────────────────────────────┤   0├────────────┤1   ├───┤1   ├─────────────────────────────────────────────────

#            (0, 0): ────SE(1)─────────H───────SE(1)───┤0   ├────────────────────@────────────────┤   1├────SE(1)──────────SE(1)───SE(1)────SE(1)────SE(1)───SE(1)─────────────────────────────────
#                                                                                │
#            (0, 1): ────SE(1)─────────SE(1)───SE(1)─────────────┤0   ├──────────X───────┤   1├───SE(1)──────────┤0   ├────X───────┤   1├───SE(1)────S───────SE(1)────SE(1)────────────────────────
#                                                                                                                          │
#            (1, 0): ────CULT(0.785)───SE(1)───SE(1)───SE(1)─────SE(1)───────────SE(1)───SE(1)────SE(1)─────SE(1)──────────┼───────────────────────────────────────────────────────────────────────
#                                                                                                                          │
#            (1, 1): ────CULT(0.785)───SE(1)───SE(1)───SE(1)───────────┤0   ├──────────────────────────────────────────────@────────────────┤   1├───SE(1)───┤0   ├───M('')───┤   1├───SE(1)───R───

#            (2, 0): ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤   0├───────────┤1   ├───────────────
#                                                               └────────────┘                             └───────────┘
#             """,
#         ),
#     )


# def test_t_lattice(t_circuit: cirq.Circuit) -> None:
#     lattice_layout = Column(t_circuit)
#     lattice_architecture = arch.DefaultLattice(
#         d=7,
#         cultivation_repetition=1,
#         syndrome_rounds=1,
#         idling=True,
#         post_op_correction=True,
#     )
#     compiled_t_circuit = comp.ft_compile(layout=lattice_layout, arc=lattice_architecture)
#     # yes idling, yes post-op correction
#     # I'm not sure if the splits are being handled properly. Should we idle on qubits being acted on by Split? Just the non-ancillas? There could be some discussion here
#     compiled_t_circuit = cirq.align_left(compiled_t_circuit)
#     cirq.testing.assert_has_diagram(
#         compiled_t_circuit,
#         textwrap.dedent(
#             """
#                                     ┌──────────┐   ┌──────────┐                                   ┌──────────┐   ┌──────────┐
#            (0, 0): ───CULT(1.571)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───────────────────────────────────

#            (0, 2): ───SE(1)──────────H──────────────MERGE─────────SPLIT───SE(1)───SE(1)───SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───SE(1)───────────────────
#                                                     │             │
#            (0, 3): ─────────────────────────────────#2────────────#2──────MERGE───────────SPLIT─────────────────────────────────────────────────────────────────────────────────
#                                                                           │               │
#            (0, 4): ───SE(1)──────────SE(1)──────────SE(1)─────────────────#2──────SE(1)───#2───────#3─────────────#3────────────SE(1)───────────#2──────SE(1)───#2──────Z───────
#                                                                                                    │              │                             │               │
#            (0, 5): ──────────────────#3─────────────#3─────────────────────────────────────────────#2─────────────#2────────────#2──────#2──────MERGE───────────SPLIT───────────
#                                      │              │                                              │              │             │       │
#            (0, 6): ───CULT(1.571)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)────────────┼──────────────┼─────────────MERGE───SPLIT───M('')───SE(1)───R───────SE(1)───
#                                      │              │                                              │              │
#            (1, 0): ───CULT(0.785)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───SE(1)───SE(1)────┼────SE(1)─────┼────SE(1)────SE(1)───SE(1)───────────────────────────────────
#                                      │              │                                              │              │
#            (1, 5): ──────────────────#2─────────────#2─────────────────────────────────────────────MERGE──────────SPLIT─────────────────────────────────────────────────────────
#                                      │              │
#            (1, 6): ───CULT(0.785)────MERGE──────────SPLIT─────────M('')───SE(1)───R───────SE(1)────SE(1)──────────SE(1)─────────SE(1)───SE(1)───SE(1)───────────────────────────
#                                     └──────────┘   └──────────┘                                   └──────────┘   └──────────┘
#             """
#         ),
#     )


# def test_ssm_moves() -> None:
#     a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
#     input_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         cirq.CNOT.on(c, b),
#         cirq.CNOT.on(a, b),
#         cirq.MeasurementGate(1, key="").on(c),
#     )
#     layout = MovementLayout(
#         input_circuit=input_circuit, num_t_factories=1, num_ccz_factories=0, architecture="SSM"
#     )
#     interaction_qids = layout.zone_qubits(zone_type="interact")
#     measurement_qids = layout.zone_qubits(zone_type="measure")
#     expected_output_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         css.MovementGate({0: 1}).on(c, interaction_qids[0]),
#         css.MovementGate({0: 1}).on(b, interaction_qids[0]),
#         cirq.CNOT.on(c, b),
#         css.MovementGate({1: 0}).on(b, interaction_qids[0]),
#         css.MovementGate({1: 0}).on(c, interaction_qids[0]),
#         css.MovementGate({0: 1}).on(a, interaction_qids[1]),
#         css.MovementGate({0: 1}).on(b, interaction_qids[1]),
#         cirq.CNOT.on(a, b),
#         css.MovementGate({1: 0}).on(b, interaction_qids[1]),
#         css.MovementGate({1: 0}).on(a, interaction_qids[1]),
#         css.MovementGate({0: 1}).on(c, measurement_qids[0]),
#         cirq.MeasurementGate(1, key="").on(c),
#         css.MovementGate({1: 0}).on(c, measurement_qids[0]),
#     )
#     # Aligning left avoids ambiguity
#     output_circuit = cirq.align_left(comp.add_moves(circuit=layout.mapped_circuit, layout=layout))
#     cirq.testing.assert_has_diagram(
#         output_circuit,
#         str(expected_output_circuit),
#     )


# def test_mzo_moves() -> None:
#     a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
#     input_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         cirq.CNOT.on(c, b),
#         cirq.CNOT.on(a, b),
#         cirq.MeasurementGate(1, key="").on(c),
#     )

#     layout = MovementLayout(
#         input_circuit=input_circuit, num_t_factories=1, num_ccz_factories=0, architecture="MZO"
#     )
#     measurement_qids = layout.zone_qubits(zone_type="measure")

#     expected_output_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         css.MovementGate({0: 1}).on(c, b),
#         cirq.CNOT.on(c, b),
#         css.MovementGate({1: 0}).on(c, b),
#         css.MovementGate({0: 1}).on(a, b),
#         cirq.CNOT.on(a, b),
#         css.MovementGate({1: 0}).on(a, b),
#         css.MovementGate({0: 1}).on(c, measurement_qids[0]),
#         cirq.MeasurementGate(1, key="").on(c),
#         css.MovementGate({1: 0}).on(c, measurement_qids[0]),
#     )
#     output_circuit = cirq.align_left(comp.add_moves(circuit=layout.mapped_circuit, layout=layout))
#     cirq.testing.assert_has_diagram(
#         output_circuit,
#         str(expected_output_circuit),
#     )


# def test_hm_moves() -> None:
#     a, b, c = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)
#     input_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         cirq.CNOT.on(c, b),
#         cirq.CNOT.on(a, b),
#         cirq.MeasurementGate(1, key="").on(c),
#     )
#     layout = MovementLayout(
#         input_circuit=input_circuit,
#         num_t_factories=1,
#         num_ccz_factories=0,
#         architecture="DSM",
#     )
#     expected_output_circuit = cirq.Circuit(
#         lsp.SyndromeExtract(1, 1).on_each(a, b),
#         lsp.Cultivate(pi / 4).on(c),
#         css.MovementGate({0: 1}).on(c, b),
#         cirq.CNOT.on(c, b),
#         css.MovementGate({1: 0}).on(c, b),
#         css.MovementGate({0: 1}).on(a, b),
#         cirq.CNOT.on(a, b),
#         css.MovementGate({1: 0}).on(a, b),
#         cirq.MeasurementGate(1, key="").on(c),
#     )
#     output_circuit = comp.add_moves(circuit=layout.mapped_circuit, layout=layout)
#     cirq.testing.assert_has_diagram(
#         output_circuit,
#         str(expected_output_circuit),
#     )


# def test_replace_cirq_op_distil_t(bell_circuit: cirq.Circuit, movement_context: comp.Context) -> None:
#     distillery_layout = MovementDistillery(
#         bell_circuit, num_t_factories=2, num_ccz_factories=0, architecture="DSM"
#     )
#     # Override context
#     movement_context.distil_t = True

#     op_to_replace = cirq.T.on(cirq.GridQubit(0, 0))
#     ops_flattened = list(
#         cirq.flatten_to_ops(
#             comp.replace_cirq_op(
#                 op=op_to_replace,
#                 layout=distillery_layout,
#                 context=movement_context,
#             )
#         )
#     )
#     expected_types: list[GateKey] = [
#         lsp.Distil("T"),
#         lsp.Distil("T"),
#         cirq.CNOT,
#         cirq.MeasurementGate,
#         cirq.ResetChannel,
#         cirq.S,
#     ]
#     assert len(expected_types) == len(ops_flattened)
#     for op, expected_type in zip(ops_flattened, expected_types):
#         assert op in cirq.GateFamily(expected_type)


# def test_replace_cirq_op_distil_ccz(random_circ: cirq.Circuit, movement_context: comp.Context) -> None:
#     distillery_layout = MovementDistillery(
#         random_circ, num_ccz_factories=2, num_t_factories=0, architecture="DSM"
#     )

#     op_to_replace = cirq.CCZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
#     returned_ops = list(
#         cirq.flatten_to_ops(
#             comp.replace_cirq_op(op=op_to_replace, layout=distillery_layout, context=movement_context)
#         )
#     )
#     # We flatten them here to be explicit about the order the operations should be in
#     ops_flattened = list(cirq.flatten_to_ops(returned_ops))
#     expected_types: list[GateKey] = [
#         *([lsp.Distil("CCZ")] * 2),
#         *([cirq.CNOT] * 3),
#         *([cirq.MeasurementGate] * 3),
#         *([cirq.ResetChannel] * 3),
#         *([cirq.H] * 3),
#         *([cirq.X] * 3),
#         *([cirq.CNOT] * 3),
#         *([cirq.H] * 3),
#     ]
#     assert len(expected_types) == len(ops_flattened)
#     for op, expected_type in zip(ops_flattened, expected_types):
#         assert op in cirq.GateFamily(expected_type)


# def test_different_rounds_distil() -> None:
#     circuit = cirq.Circuit(cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
#     layout = MovementDistillery(input_circuit=circuit, architecture="SSM")
#     for k in [1, 5, 7]:
#         architecture = arch.DefaultMovement(
#             idling=False,
#             post_op_correction=True,
#             d=7,
#             cultivation_repetition=1,
#             syndrome_rounds=k,
#         )
#         compiled_circuit = comp.ft_compile(layout=layout, arc=architecture)
#         for op in compiled_circuit.all_operations():
#             if isinstance(op.gate, lsp.SyndromeExtract):
#                 assert op.gate.rounds == k


# def test_teleport_resource_exceptions(movement_context: comp.Context) -> None:
#     invalid_resource = cirq.CCZ.on(*(cirq.GridQubit(0, i) for i in range(3)))
#     layout = MovementLayout(cirq.Circuit(), architecture="DSM")
#     with pytest.raises(ValueError, match="Invalid resource"):
#         _ = comp.teleport_resource(invalid_resource, layout, movement_context)

#     invalid_qubit = cirq.LineQubit(0)
#     with pytest.raises(TypeError, match="Qubits must be instances"):
#         _ = comp.teleport_resource(cirq.T.on(invalid_qubit), layout, movement_context)


# def test_add_moves() -> None:
#     line_qubits = cirq.LineQubit.range(8)
#     cnot_circuit = cirq.Circuit([cirq.CNOT.on_each(*zip(line_qubits[::2], line_qubits[1::2]))])
#     measure_circuit = cirq.Circuit([cirq.measure_each(*line_qubits)])

#     cnot_zoned_layout = MovementLayout(input_circuit=cnot_circuit, architecture="SSM")
#     meas_zoned_layout = MovementLayout(input_circuit=measure_circuit, architecture="MZO")
#     cnot_inplace_layout = MovementLayout(input_circuit=cnot_circuit, architecture="DSM")
#     expected_moment_types: list[GateKey]

#     # Test that interaction zones makes sense for a zoned layout
#     # There should be three interaction zones, so we expect to see an extra moment for the final CNOT
#     circuit_with_moves = comp.add_moves(cnot_zoned_layout.mapped_circuit, layout=cnot_zoned_layout)
#     expected_moment_types = [
#         css.MovementGate,  # Move controls to the three available interaction zones
#         css.MovementGate,  # Move targets to the three available interaction zones
#         cirq.CNOT,  # Perform the CNOT operation
#         css.MovementGate,  # Move the controls from the three available interaction zones
#         css.MovementGate,  # Move the targets from the three available interaction zones
#         css.MovementGate,  # Move the controls to the three available interaction zones
#         css.MovementGate,  # Move the targets to the three available interaction zones
#         cirq.CNOT,  # Perform the CNOT operation
#         css.MovementGate,  # Move the controls from the three available interaction zones
#         css.MovementGate,  # Move the targeets from the three available interaction zones
#     ]
#     moments_match_expected = [
#         all(op in cirq.GateFamily(expected_op) for op in moment)
#         for moment, expected_op in zip(circuit_with_moves, expected_moment_types)
#     ]
#     assert all(moments_match_expected)

#     # Test that the measurement zone makes sense for a zoned layout
#     circuit_with_moves = comp.add_moves(meas_zoned_layout.mapped_circuit, layout=meas_zoned_layout)
#     expected_moment_types = [
#         css.MovementGate,  # Populate the three available measurement zones
#         cirq.MeasurementGate,  # Measure the logical patches
#         css.MovementGate,  # Exit the three filled measurement zones
#         css.MovementGate,  # Populate the three available measurement zones
#         cirq.MeasurementGate,  # Measure the logical patches
#         css.MovementGate,  # Exit the three filled measurement zones
#         css.MovementGate,  # Populate the three available measurement zones
#         cirq.MeasurementGate,  # Measure the logical patches
#         css.MovementGate,  # Exit the three filled measurement zones
#     ]
#     moments_match_expected = [
#         all(op in cirq.GateFamily(expected_op) for op in moment)
#         for moment, expected_op in zip(circuit_with_moves, expected_moment_types)
#     ]
#     assert all(moments_match_expected)

#     # Test that inplace moves make sense for an inplace layout
#     circuit_with_moves = comp.add_moves(
#         cnot_inplace_layout.mapped_circuit, layout=cnot_inplace_layout
#     )
#     expected_moment_types = [
#         css.MovementGate,  # Move all controls to their respective targets
#         cirq.CNOT,  # Perform the CNOT operation
#         css.MovementGate,  # Move all controls from their respective targets
#     ]
#     moments_match_expected = [
#         all(op in cirq.GateFamily(expected_op) for op in moment)
#         for moment, expected_op in zip(circuit_with_moves, expected_moment_types)
#     ]
#     assert all(moments_match_expected)


# def test_exceptions(bell_circuit: cirq.Circuit) -> None:
#     # Test ft compile rejects incompatible layout-architecture combos
#     inplace_layout = MovementLayout(input_circuit=bell_circuit, architecture="DSM")
#     zoned_arc = arch.DefaultMovement()
#     with pytest.raises(ValueError, match="Mismatch between"):
#         _ = comp.ft_compile(layout=inplace_layout, arc=zoned_arc)
