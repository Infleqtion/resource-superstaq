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

import textwrap
from math import pi

import cirq
import cirq_superstaq as css
import pytest

import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.codepatch as cpa
import resource_estimation.ftqc.compile_ftqc as comp
import resource_estimation.ftqc.context as ctx
import resource_estimation.ftqc.lattice_surgery_primitives as lsp
import resource_estimation.ftqc.layout as lyt


def make_context(
    roles: dict[int, int],
    *,
    transversal_cnot: bool = False,
    distil_t: bool = False,
    cult_t: bool = True,
    dynamic: bool = False,
    num_s_factories: int = 0,
    num_t_factories: int = 0,
    num_ccz_factories: int = 0,
    factories: dict[tuple[int, ...], int] | None = None,
    blocks: dict[tuple[int, ...], tuple[int, ...]] | None = None,
    post_op_syndrome_extraction: bool = False,
    idling: bool = False,
) -> ctx.Context:
    pid_to_qid: dict[int, cirq.Qid] = {pid: cirq.NamedQubit(f"p{pid}") for pid in roles}

    return ctx.Context(
        roles=dict(roles),
        pid_to_qid=pid_to_qid,
        qid_to_pid={qid: pid for pid, qid in pid_to_qid.items()},
        factories={} if factories is None else dict(factories),
        blocks={} if blocks is None else dict(blocks),
        num_s_factories=num_s_factories,
        num_t_factories=num_t_factories,
        num_ccz_factories=num_ccz_factories,
        transversal_cnot=transversal_cnot,
        distil_t=distil_t,
        cult_t=cult_t,
        dynamic=dynamic,
        post_op_syndrome_extraction=post_op_syndrome_extraction,
        idling=idling,
    )


@pytest.mark.parametrize("verbose", (True, False))
def test_validate_ops(verbose: bool) -> None:
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
    first_op = result[0]
    assert isinstance(first_op, cirq.GateOperation)
    assert first_op.gate == lsp.Distil("CCZ")
    assert len(first_op.qubits) == 23

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
    second_op = second_result[0]
    assert isinstance(second_op, cirq.GateOperation)
    # The second teleportation uses a different factory
    assert second_op.qubits[:3] != first_op.qubits[:3]

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
    lattice_context = ctx.build_column_context(
        base_circuit, lattice_layout, surface_code_distance=3
    )
    ctrl = cirq.NamedQubit("1-0-4")
    trgt = cirq.LineQubit(1)
    op_to_replace = cirq.CNOT.on(ctrl, trgt)
    decomposition = comp.replace_cirq_op(op_to_replace, lattice_layout, lattice_context)
    assert len(decomposition) == 4
    gate_types = []
    for op in decomposition:
        assert isinstance(op, cirq.GateOperation)
        gate_types.append(type(op.gate))
    assert gate_types == [
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
    movement_context = ctx.build_movement_context(
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
    first_op = decomposition[0]
    last_op = decomposition[-1]
    assert isinstance(first_op, cirq.GateOperation)
    assert isinstance(last_op, cirq.GateOperation)
    assert len(decomposition) == 5
    assert first_op.gate == lsp.Cultivate(pi / 4)
    assert last_op == cirq.S.on(t_qubit)

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
    arc: arch.DefaultMovement | arch.DefaultLattice
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
