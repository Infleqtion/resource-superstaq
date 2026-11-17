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

import copy
import itertools
import os
import sys
import time
from itertools import combinations, islice
from math import pi
from typing import TYPE_CHECKING, cast
import math

import cirq
import cirq_superstaq as css
import tqdm
from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Iterator, Literal, Sequence

    from resource_estimation.ftqc.architecture import Architecture, DefaultMovement
    from resource_estimation.typing import GateKey

from . import lattice_surgery_primitives as lsp
from . import layout as lyt
from . import codepatch as cpa
import resource_estimation.ftqc.mappers as mps
import resource_estimation.ftqc.architecture as arch

## Some questions I still have:
##  Which functions should be private and which should not?
##  Can I simplify context to not need things like idling, post-op correction etc?

# This function is only visual and is extremely finicky, so it is not tested
def knock_off_tqdm(
    moment_idx: int,
    total: int,
    tstart: float,
    message: str,
) -> None:  # pragma: no cover
    """Implements tqdm-like behavior for the compiler"""
    if not sys.stdout.isatty():
        # This is to ensure that testing can progress as normal
        return
    WIDTH = os.get_terminal_size().columns
    moment_idx += 1
    time_passed = time.time() - tstart
    guessed_time = time_passed * (total / moment_idx)
    offset = len(
        f"{message} || {moment_idx} / {total} ["
        f"{int(time_passed // 3600)}:{int(time_passed // 60)}:{int(time_passed % 60)}.{int(10 * time_passed) % 10}{int(100 * time_passed) % 10}<"
        f"{int(guessed_time // 3600)}:{int(guessed_time // 60)}:{int(guessed_time % 60)}.{int(10 * guessed_time) % 10}{int(100 * guessed_time) % 10}, "
        f"{round(moment_idx / time_passed, 2)}it/s]",
    )
    bars = int((WIDTH - offset) * moment_idx / total)
    spaces = int(WIDTH - offset) - bars
    full_bar = (
        f"{message} |\033[36m{'█' * bars + ' ' * spaces}\033[0m| {moment_idx} / {total} ["
        f"{int(time_passed // 3600)}:{int(time_passed // 60)}:{int(time_passed % 10)}.{int(10 * time_passed) % 10}{int(100 * time_passed) % 10}<"
        f"{int(guessed_time // 3600)}:{int(guessed_time // 60)}:{int(guessed_time % 10)}.{int(10 * guessed_time) % 10}{int(100 * guessed_time) % 10}, "
        f"{round(moment_idx / time_passed, 2)}it/s]"
    )
    print(
        full_bar,
        end="\r" if moment_idx < total else "\n",
    )

def route_lattice_cnot(
        ctrl: cirq.Qid, 
        trgt: cirq.Qid, 
        layout: lyt.Layout, 
        context: mps.Context,
    ) -> list[int]:
    """
    Extremely dumb routing algorithm for routing a merge operation.
    Explores all possible paths according to set constraints.
    Path must go through logical nearest neighbors.
    QIDs in path must be ancilla patches (or ctrl/trgt).
    No loops allowed.
    """
    pid_to_placement = {
        p.element.patch_id: p for p in layout.placements.values() if isinstance(p.element, cpa.RotatedSurfaceCodePatch)
    }
    ancilla_role = 2
    # Just do a dumb routing algorithm for now
    last = context.qid_to_pid[ctrl]
    first = context.qid_to_pid[trgt]
    paths = [[first]]
    while not any(last in path for path in paths):
        new_paths = []
        for path in paths:
            last_placement = pid_to_placement[path[-1]]
            for pid, placement in pid_to_placement.items():
                is_adjacent = placement.borders(last_placement)
                is_valid_step = context.roles[pid] == ancilla_role or pid == last
                is_not_in_path = pid not in path
                if is_adjacent and is_valid_step and is_not_in_path:
                    new_paths.append(path + [pid])
        if not new_paths:
            raise ValueError(f"Could not find valid path from {ctrl} to {trgt}")
        paths = new_paths
    best_path = min(
        [path for path in paths if last in path], 
        key=len
    )
    return best_path

def _decompose_lattice_cnot(
        ctrl: cirq.Qid, 
        trgt: cirq.Qid, 
        layout: lyt.Layout,
        context: mps.Context,
    ) -> list[cirq.Operation | cirq.Moment]:
        path_patch_ids = route_lattice_cnot(ctrl, trgt, layout, context=context)
        path_qubits = [context.pid_to_qid[patch_id] for patch_id in path_patch_ids]
        # path_qubits.reverse()
        num_qubits = len(path_qubits)
        decomposition = [
            lsp.Merge(num_qubits=num_qubits-1, smooth=True).on(*path_qubits[:-1]),
            lsp.Split(partitions=[1, len(path_qubits[:-1]) - 1], smooth=True).on(
                *path_qubits[:-1],
            ),
            lsp.Merge(num_qubits=num_qubits - 1, smooth=False).on(*path_qubits[1:]),
            lsp.Split(partitions=[1] * (len(path_qubits[1:])), smooth=False).on(*path_qubits[1:]),
        ]
        return decomposition


def _requires_resource(op: cirq.Operation, context: mps.Context) -> bool:
    """Checks if performing an operation requires a resource state. S is assumed to need a resource state when transversal CNOT is unavailable."""
    if op in cirq.GateFamily(cirq.S) and not context.transversal_cnot:
        return True
    if op in cirq.GateFamily(cirq.T):
        return True
    if op in cirq.GateFamily(cirq.CCZ):
        return True
    return False


def replace_cirq_op(
    op: cirq.Operation,
    layout: lyt.Layout,
    context: mps.Context
) -> list[cirq.Operation | cirq.Moment]:
    """Replacement logic similar to decomposition for cirq operations to be converted to primitives.

    op: cirq operation to be unrolled
    layout: Layout of the logical qubits
    primitives: cirq gates that are allowed in the underlying architecture
    verbose: flag to print more information
    """
    if op.gate == cirq.CNOT and not context.transversal_cnot:
        ctrl, trgt = op.qubits
        return _decompose_lattice_cnot(ctrl=ctrl, trgt=trgt, layout=layout, context=context)
    if _requires_resource(op, context=context):
        return teleport_resource(op, layout, context)
    raise ValueError(
        f"Invalid Op for the given layout and context: {op.gate}",
    )

def _determine_resource(op: cirq.Operation, context: mps.Context) -> tuple[cirq.Operation, cirq.Operation, int]:
    if op in cirq.GateFamily(cirq.CCZ):
        prep_gate = lsp.Distil("CCZ")
        if context.dynamic:
            correction = lsp.ResourceCorrection("CCZ")
            resource_id = 5
        else:
            correction = [
                *cirq.H.on_each(*op.qubits),
                *cirq.X.on_each(*op.qubits),
                *cirq.CNOT.on_each(*combinations(op.qubits, 2)),
                *cirq.H.on_each(*op.qubits),
            ]
            resource_id = 5
    elif op in cirq.GateFamily(cirq.S):
        prep_gate = lsp.Cultivate(pi / 2)
        correction = cirq.Z
        resource_id = 3
    elif op in cirq.GateFamily(cirq.T) and context.distil_t:
        prep_gate = lsp.Distil("T")
        correction = lsp.ResourceCorrection("T") if context.dynamic else cirq.S
        resource_id = 4
    elif op in cirq.GateFamily(cirq.T) and context.cult_t:
        prep_gate = lsp.Cultivate(pi / 4)
        correction = lsp.ResourceCorrection("T") if context.dynamic else cirq.S
        resource_id = 4
    else:
        raise ValueError(f"Invalid resource encountered: {op.gate}")
    return (prep_gate, correction, resource_id)

def _nearest_factory(qids: tuple[cirq.Qid, ...], resource_id: int, layout: lyt.Layout, context: mps.Context) -> tuple[int, ...]:
    pids = set(context.qid_to_pid[qid] for qid in qids)
    patch_placements = {}
    for placement in layout.placements.values():
        if isinstance(placement.element, cpa.RotatedSurfaceCodePatch):
            patch_placements[placement.element.patch_id] = placement
    resource_states = set(factory for factory, rid in context.factories.items() if rid == resource_id)
    if not resource_states:
        raise ValueError(
            f"No available factories for resource ID {resource_id}"
        )
    def dist(resource_state: tuple[int, ...]) -> int:
        total = 0
        for resource_pid in resource_state:
            for target_pid in pids:
                r_patch = patch_placements[resource_pid]
                t_patch = patch_placements[target_pid]
                total += abs(r_patch.row - t_patch.row) + abs(r_patch.col - t_patch.col)
        return total
    best_candidate = min(resource_states, key=dist)
    return best_candidate

def _find_factories_simple(context: mps.Context, resource_id: int, num_qubits: int) -> set[tuple[int, ...]]:
    if resource_id == 3:
        max_facts = context.num_s_factories
    elif resource_id == 4:
        max_facts = context.num_t_factories
    elif resource_id == 5:
        max_facts = context.num_ccz_factories
    else:
        raise ValueError(f"Cannot parse resource id {resource_id}")
    found_factories = set()
    current_factory = []
    for pid, role in context.roles.items():
        if len(found_factories) == max_facts:
            break
        if role == resource_id:
            current_factory.append(pid)
        if len(current_factory) == num_qubits:
            found_factories.add(tuple(current_factory))
            current_factory = []
    return found_factories

def _find_blocks_simple(context: mps.Context, resource_id: int, block_size: int) -> set[tuple[int, ...]]:
    assigned_blocks = {
        pid
        for block in context.blocks.values()
        for pid in block
    }
    assigned_factories = {pid for factory, rid in context.factories.items() for pid in factory if rid == resource_id}
    if resource_id == 3:
        max_blocks = context.num_s_factories
    elif resource_id == 4:
        max_blocks = context.num_t_factories
    elif resource_id == 5:
        max_blocks = context.num_ccz_factories
    else:
        raise ValueError(f"Cannot parse resource id {resource_id}")
    found_blocks = set()
    current_block = []
    for pid, role in context.roles.items():
        if pid in assigned_factories or pid in assigned_blocks:
            continue
        if len(found_blocks) == max_blocks:
            continue
        if role == resource_id:
            current_block.append(pid)
        if len(current_block) == block_size:
            found_blocks.add(tuple(current_block))
            current_block = []
    return found_blocks


def teleport_resource(
    op: cirq.Operation, layout: lyt.Layout, context: mps.Context,
) -> list[cirq.Moment | cirq.Operation]:
    prep_gate, correction, resource_id = _determine_resource(op, context)
    block_size = cirq.num_qubits(prep_gate) - cirq.num_qubits(op)

    # Get available Factories and Blocks
    available_factories = set(pid for pid, rid in context.factories.items() if rid == resource_id)
    # Reload Factories if needed
    if not available_factories:
        all_factories = _find_factories_simple(context=context, resource_id=resource_id, num_qubits=cirq.num_qubits(op))
        for factory_qubits in all_factories:
            context.factories.update({factory_qubits: resource_id})
    # Reload blocks if needed
    if block_size:
        missing_factories = [
            factory
            for factory, rid in context.factories.items()
            if rid == resource_id and factory not in context.blocks
        ]

        if missing_factories:
            all_blocks = _find_blocks_simple(
                context, resource_id, block_size
            )

            if len(all_blocks) < len(missing_factories):
                raise ValueError("Not enough resource blocks available")

            for factory in missing_factories:
                context.blocks[factory] = all_blocks.pop()
    # Route qubits in operation to the nearest factory
    routed_factory = _nearest_factory(op.qubits, resource_id, layout, context)

    # Map patch ids for factory and block to qids
    factory_qids = tuple(map(context.pid_to_qid.get, routed_factory))
    block_qids = tuple(map(context.pid_to_qid.get, context.blocks[routed_factory])) if block_size else tuple()

    # Write replacement injection operation
    resource_state_prep = prep_gate.on(*(factory_qids + block_qids))
    cnots, measurements, resets = [], [], []  # resets should be state preps...
    corrections: list[cirq.Operation] = (
        correction if isinstance(correction, list) else [correction.on(*op.qubits)]
    )
    for factory_qubit, program_qubit in zip(factory_qids, op.qubits):
        cnots.append(cirq.CNOT.on(factory_qubit, program_qubit))
        measurements.append(cirq.MeasurementGate(1, key="").on(factory_qubit))
        resets.append(cirq.ResetChannel().on(factory_qubit))
    operations = [
        resource_state_prep,
        cirq.Moment(cnots),
        cirq.Moment(measurements),
        cirq.Moment(resets),
        *corrections,
    ]
    # Mark spent factories and blocks
    context.factories.pop(routed_factory)
    if block_size:
        context.blocks.pop(routed_factory)
    return operations

def handle_idling(
    circuit: cirq.Circuit,
    arc: arch.Architecture,
    context: mps.Context,
    verbose: int = 0,
) -> cirq.Circuit:
    """Helper function for the compiler that handles idling. This way we can experiment with different kinds of idling or even turn it off entirely.
    This function is still a work in progress, but it is likely to take the form of various compiler passes.
    """
    non_ancillas = {context.pid_to_qid[pid] for pid, rid in context.roles.items() if rid != 2}

    # Build circuit where Syndrome Extract is performed on Idling qubits that are not being acted upon
    # Split moments are treated separately because they can always get absorbed into the previous moment
    total = len(circuit)
    tstart = time.time()

    rounds = arc.d if arc.syndrome_rounds is None else arc.syndrome_rounds
    se = lsp.SyndromeExtract(1, rounds)

    def _map_fn(moment: cirq.Moment, moment_idx: int) -> cirq.Moment | Sequence[cirq.Moment]:
        if verbose > 0:
            knock_off_tqdm(moment_idx=moment_idx, total=total, tstart=tstart, message="Idling:")
        if all(isinstance(gate.gate, (css.Barrier, lsp.Split)) for gate in moment):
            return moment
        if moment_idx == 0 or sum(
            isinstance(gate.gate, lsp.SyndromeExtract) for gate in moment
        ) == len(non_ancillas):
            # This is stateprep or a moment equivalent to idling, so we do not need to add idling
            return moment

        idling_qubits = non_ancillas - moment.qubits

        moment = cirq.Moment(*moment, *se.on_each(*idling_qubits), _flatten_contents=False)

        return moment

    circuit_with_idling = cirq.map_moments(circuit, _map_fn)

    return circuit_with_idling


def add_zones_to_context(context: mps.Context, arc: arch.Architecture, layout: lyt.Layout) -> mps.Context:
    """
    Add qids to the context to represent locations in the layout's zone.
    I want to make this closer to a compiler pass than having it statically defined at the Context initialization.
    These QIDs are only in service of the current, relatively rough, handling of inter-patch movement as a css.MovementGate.
    """
    # Copy context
    new_context = copy.deepcopy(context)
    patch_side_length = 2*arc.d - 1  # The length of a surface code patch
    # Break up zone into qids
    if interaction_zone := layout.regions.get(0, None):
        num_zone_qubits = interaction_zone.element.footprint.cols // patch_side_length
        qids = cirq.NamedQubit.range(num_zone_qubits, prefix="IZ")
        next_pid = max(new_context.pid_to_qid) + 1
        pids = range(next_pid, next_pid+len(qids))
        update_qid_to_pid = {qid: pid for qid, pid in zip(qids, pids)}
        update_pid_to_qid = {pid: qid for pid, qid in zip(pids, qids)}
        update_roles = {pid: 6 for pid in pids}
        new_context.qid_to_pid.update(update_qid_to_pid)
        new_context.pid_to_qid.update(update_pid_to_qid)
        new_context.roles.update(update_roles)
    if measure_zone := layout.regions.get(1, None):
        num_zone_qubits = measure_zone.element.footprint.cols // patch_side_length
        qids = cirq.NamedQubit.range(num_zone_qubits, prefix="MZ")
        next_pid = max(new_context.pid_to_qid) + 1
        pids = range(next_pid, next_pid+len(qids))
        update_qid_to_pid = {qid: pid for qid, pid in zip(qids, pids)}
        update_pid_to_qid = {pid: qid for pid, qid in zip(pids, qids)}
        update_roles = {pid: 7 for pid in pids}
        new_context.qid_to_pid.update(update_qid_to_pid)
        new_context.pid_to_qid.update(update_pid_to_qid)
        new_context.roles.update(update_roles)
    return new_context

def post_op_syndrome_extraction(
    circuit: cirq.Circuit,
    # context: mps.Context,
    arc: Architecture,
    verbose: int = 0,
) -> cirq.Circuit:
    """For movement, it has been suggested that we just do syndrome extraction (for a single round) right after a logical operations."""
    # Allowing a little bit of flexibility on what we want to correct
    # Might even want to add Lattice Primitives, but there aren't many (any?) that are not implicitly corrected
    ops_to_correct = [
        cirq.CNOT,
        cirq.S,
        lsp.ResourceCorrection("T"),
        lsp.ResourceCorrection("CCZ"),
    ]
    if isinstance(arc, arch.DefaultMovement):
        ops_to_correct.append(cirq.H)

    rounds = arc.d if arc.syndrome_rounds is None else arc.syndrome_rounds
    syndrome_extract = lsp.SyndromeExtract(1, rounds)
    # barrier = css.barrier(*sorted(circuit.all_qubits()))

    total = len(circuit)
    tstart = time.time()

    def _map_fn(op: cirq.Operation, moment_idx: int) -> Iterator[cirq.Operation]:
        if verbose:
            knock_off_tqdm(
                moment_idx=moment_idx,
                total=total,
                tstart=tstart,
                message="Post-Op Correction:",
            )

        yield op

        qubits_to_correct = [
            q
            for q in op.qubits
            if op.gate in ops_to_correct or isinstance(op.gate, cirq.MeasurementGate)
        ]
        if qubits_to_correct:
            yield from syndrome_extract.on_each(*qubits_to_correct)

    return cirq.map_operations_and_unroll(circuit, _map_fn, raise_if_add_qubits=False)


def validate_ops(circuit: cirq.Circuit, verbose: int = 1) -> None:
    """Checks that the given circuit is in the Clifford+T gateset. CCZs are also allowed"""
    valid_gates = (
        cirq.T,
        cirq.X,
        cirq.Z,
        cirq.S,
        cirq.H,
        cirq.I,
        cirq.CNOT,
        cirq.CCZ,
    )
    valid_types = (
        cirq.MeasurementGate,
        cirq.ResetChannel,
    )
    total_ops = len(list(circuit.all_operations()))
    if not all(
        op.gate in valid_gates or isinstance(op.gate, valid_types)
        for op in tqdm.tqdm(circuit.all_operations(), total=total_ops, disable=not verbose)
    ):
        raise ValueError("This compiler only handles Clifford + T + CCZ circuits")


def _decompose_to_primitives(
    circuit: cirq.Circuit,
    layout: lyt.Layout,
    arc: Architecture,
    context: mps.Context,
) -> cirq.Circuit:
    # TODO: Is this function still necessary if it is just a wrapper for replace_cirq_op?
    primitives = cirq.Gateset(
        *(cirq.GateFamily(g._gate, ignore_global_phase=False) for g in arc.primitives.gates),
    )
    def _map_fn(op: cirq.Operation) -> cirq.OP_TREE:
        return replace_cirq_op(
            op=op, layout=layout, context=context
        )

    # TODO: can we turn layout into a decomposition_context?
    ops = cirq.decompose(
        circuit,
        intercepting_decomposer=_map_fn,
        keep=primitives.__contains__,
    )
    return cirq.Circuit(ops)


def add_moves(
    circuit: cirq.Circuit,
    # arc: arch.Architecture,
    context: mps.Context,
    layout: lyt.Layout,
    verbose: int = 0,
) -> cirq.Circuit:
    """
    Add logical css.MovementGate operations according to the architecture/layout type
    For layouts with zones, target the next zone in a cycle
    For inplace CNOTs, move the control to the target logical patch
    The movement sequence is designed to be simple and repeatable to make resourcing easier at lower levels of the stack, but a more dedicated compiler pass could be swapped in with a more efficient policy for logical movement
    """
    total = len(circuit)
    tstart = time.time()

    interaction_cycle = itertools.cycle([
        context.pid_to_qid[pid] for pid, rid in context.roles.items() if rid == 6
    ])
    measurement_cycle = itertools.cycle([
        context.pid_to_qid[pid] for pid, rid in context.roles.items() if rid == 7
    ])
    ops_to_replace: list[GateKey] = [cirq.CNOT]
    if any(region.element.role == "Measure" for region in layout.regions.values()):
        ops_to_replace.append(cirq.MeasurementGate)
        measure_zones = True
    else:
        measure_zones = False
    if any(region.element.role == "Interaction" for region in layout.regions.values()):
        inplace_cnot, interaction_zones = False, True
    else:
        inplace_cnot, interaction_zones = True, False

    replacement_gateset = cirq.Gateset(*ops_to_replace)

    def map_func(op: cirq.Operation, moment_idx: int) -> Iterator[cirq.OP_TREE]:
        if verbose:
            knock_off_tqdm(
                moment_idx=moment_idx,
                total=total,
                tstart=tstart,
                message="Adding Qubit Movement:",
            )
        if op not in replacement_gateset:
            yield op
        else:
            move = css.MovementGate({0: 1})
            move_dg = css.MovementGate({1: 0})
            if inplace_cnot and op.gate == cirq.CNOT:
                ctrl, trgt = op.qubits
                op_sequence = [move.on(ctrl, trgt), op, move_dg.on(ctrl, trgt)]
            elif interaction_zones and op.gate == cirq.CNOT:
                ctrl, trgt = op.qubits
                zone_qubit = next(interaction_cycle)
                op_sequence = [
                    move.on(ctrl, zone_qubit),
                    move.on(trgt, zone_qubit),
                    op,
                    move_dg.on(trgt, zone_qubit),
                    move_dg.on(ctrl, zone_qubit),
                ]
            elif measure_zones and cirq.is_measurement(op):
                op_sequence = []
                for q in op.qubits:
                    zone_qubit = next(measurement_cycle)
                    # TODO: Upgrade to contain more information like key
                    single_measure = cirq.MeasurementGate(1).on(q)
                    op_sequence.extend([move.on(q, zone_qubit), single_measure, move_dg.on(q, zone_qubit)])
            for op in op_sequence:
                yield op

    return cirq.map_operations_and_unroll(circuit, map_func, raise_if_add_qubits=False)

def allocate_qubits_simple(
        circuit: cirq.Circuit, 
        layout: lyt.Layout, 
        distil_t: bool,
        num_t_factories: int,
        num_ccz_factories: int,
    ) -> tuple[dict[cirq.Qid, int], dict[int, int]]:
    num_logical_qubits = cirq.num_qubits(circuit)
    patches_per_t_factory = 31 if distil_t else 1
    num_required_patches = num_logical_qubits + patches_per_t_factory*num_t_factories+23*num_ccz_factories
    num_code_patches = sum(isinstance(p.element, cpa.RotatedSurfaceCodePatch) for p in layout.placements.values())
    if num_required_patches > num_code_patches:
        raise ValueError("Layout does not have enough patches")
    
    # Map qids in original circuit to patch ids in Layout
    qid_to_pid = {qid: idx for idx, qid in enumerate(circuit.all_qubits())}
    # Assign roles to patches
    roles = {pid: 1 for pid in qid_to_pid.values()}
    t_qids = cirq.NamedQubit.range(num_t_factories*patches_per_t_factory, prefix="t")
    for idx, qid in enumerate(t_qids):
        pid = num_logical_qubits + idx
        qid_to_pid[qid] = pid
        roles[pid] = 4
    ccz_qids = cirq.NamedQubit.range(23*num_ccz_factories, prefix="ccz")
    for idx, qid in enumerate(ccz_qids):
        pid = num_logical_qubits + num_t_factories * patches_per_t_factory + idx
        qid_to_pid[qid] = pid
        roles[pid] = 5
    return qid_to_pid, roles

def ft_compile(
        circuit: cirq.Circuit,
        arc: Architecture,
        layout: lyt.Layout,
        context: mps.Context,
    ):
    # Decompose CNOTs and add resource state injections
    circuit = _decompose_to_primitives(circuit=circuit, layout=layout, arc=arc, context=context)

    # Prepare initial logical states
    # I would like to replace this with a dedicated gate, rather than SyndromeExtraction
    logical_qubits = [context.pid_to_qid[pid] for pid, rid in context.roles.items() if rid == 1]
    rounds = arc.d if arc.syndrome_rounds is None else arc.syndrome_rounds
    state_prep = cirq.Circuit(lsp.SyndromeExtract(1, rounds).on_each(*logical_qubits))
    circuit = state_prep + circuit

    # Add SyndromeExtraction after logical operations
    if context.post_op_syndrome_extraction:
        circuit = post_op_syndrome_extraction(circuit=circuit, arc=arc)
    # Add SyndromeExtraction to idling logical qubits
    if context.idling:
        circuit = handle_idling(circuit=circuit, arc=arc, context=context)
    # This is where movement was originally...
    # So I'll keep it here for now...
    if isinstance(arc, arch.DefaultMovement):
        # context = add_zones_to_context(context=context, arc=arc, layout=layout)
        context = add_zones_to_context(context=context, arc=arc, layout=layout)
        circuit = add_moves(circuit=circuit, context=context, layout=layout)
    
    return circuit, context



# def ft_compile(
#     layout: Layout,
#     arc: Architecture,
#     verbose: int = 1,
#     with_barriers: bool = False,
#     skip_validation: bool = False,
#     dynamic: bool = False,
# ) -> cirq.Circuit:
#     """Basic read/replace compiler that converts a cirq Circuit over the Clifford + T + CCZ gateset to a cirq circuit of primitives.
#     The layout input contains the input circuit and information about any routing that might be necessary during the compilation process.
#     The architecture input contains information about what primtives are accessible to the compiler and which extra passes should be added to the primitive circuit.
#     The passes available are post op correction and idling.
#     The architecture is also the source of information for how many rounds of syndrome extraction should be performed when syndrome extraction is called for.
#     """
#     zone_ops_but_no_zones = arc.zone_ops.gates and not (
#         layout.measure_zones or layout.interaction_zones
#     )
#     zones_but_no_zone_ops = (
#         layout.measure_zones or layout.interaction_zones
#     ) and not arc.zone_ops.gates
#     if zone_ops_but_no_zones or zones_but_no_zone_ops:
#         raise ValueError("Mismatch between Architecture zone ops and Layout zones")
#     layout = copy.deepcopy(layout)
#     layout.reset_graph()
#     G = layout.layout_graph

#     circuit = layout.mapped_circuit
#     if skip_validation:  # pragma: no cover
#         print("Validation Turned Off")
#     else:
#         validate_ops(circuit, verbose=verbose)
#     context = contextualize(layout, arc, dynamic=dynamic)
#     circuit = _decompose_to_primitives(circuit, layout=layout, arc=arc, context=context)

#     # Handling State Prep
#     # In a more optimized world this could happen the moment before the first logical operation
#     logical_qubits = [node for node in G.nodes if G.nodes[node]["patch_type"] == "data"]
#     state_prep = cirq.Circuit(lsp.SyndromeExtract(1, rounds=arc.rounds).on_each(*logical_qubits))
#     if with_barriers:
#         state_prep += css.barrier(*sorted(circuit.all_qubits()))
#     circuit = state_prep + circuit

#     if arc.post_op_correction:
#         circuit = post_op_syndrome_extraction(
#             circuit=circuit,
#             movement=arc.movement,
#             with_barriers=with_barriers,
#             rounds=arc.rounds,
#             verbose=verbose,
#         )

#     if arc.idling:
#         circuit = handle_idling(
#             circuit=circuit,
#             layout=layout,
#             with_barriers=with_barriers,
#             rounds=arc.rounds,
#             verbose=verbose,
#         )

#     if isinstance(layout, MovementLayout):
#         circuit = add_moves(
#             circuit=circuit,
#             layout=layout,
#             verbose=verbose,
#         )

#     return circuit
