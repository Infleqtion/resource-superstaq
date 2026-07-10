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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resource_estimation.ftqc.architecture import Architecture
import copy
import os
import sys
from collections.abc import Iterator
from itertools import combinations
from functools import partial
from math import pi
from time import time
from warnings import warn

import cirq
import cirq_superstaq as css
from cirq_superstaq import Barrier, barrier
from tqdm import tqdm

from . import lattice_surgery_primitives as lsp
from .layout import Layout

# IMPORTANT NOTES
# Classical control has not been implemented yet
#   If you requested S, I assume you measure 1 and have to do Z
#   If you requested T, I assume you measure 1 and have to do S
# ABC -- Always be Cultivating


# This function is only visual and is extremely finicky, so it is not tested
def knock_off_tqdm(
    moment_idx: int, total: int, tstart: float, message: str
) -> None:  # pragma: no cover
    """Implements tqdm-like behavior for the compiler"""
    if not sys.stdout.isatty():
        # This is to ensure that testing can progress as normal
        return
    WIDTH = os.get_terminal_size().columns
    moment_idx += 1
    time_passed = time() - tstart
    guessed_time = time_passed * (total / moment_idx)
    offset = len(
        f"{message} || {moment_idx} / {total} ["
        f"{int(time_passed // 3600)}:{int(time_passed // 60)}:{int(time_passed % 60)}.{int(10 * time_passed) % 10}{int(100 * time_passed) % 10}<"
        f"{int(guessed_time // 3600)}:{int(guessed_time // 60)}:{int(guessed_time % 60)}.{int(10 * guessed_time) % 10}{int(100 * guessed_time) % 10}, "
        f"{round(moment_idx / time_passed, 2)}it/s]"
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


def _requires_resource(op: cirq.Operation, transversal_cnot: bool) -> bool:
    """Checks if performing an operation requires a resource state. S is assumed to need a resource state when transversal CNOT is unavailable."""
    if op in cirq.GateFamily(cirq.S) and not transversal_cnot:
        return True
    if op in cirq.GateFamily(cirq.T):
        return True
    if op in cirq.GateFamily(cirq.CCZ):
        return True
    return False


def replace_cirq_op(
    op: cirq.Operation,
    layout: Layout,
    transversal_cnot: bool,
) -> list[cirq.Operation]:
    """Replacement logic similar to decomposition for cirq operations to be converted to primitives.

    op: cirq operation to be unrolled
    layout: Layout of the logical qubits
    primitives: cirq gates that are allowed in the underlying architecture
    verbose: flag to print more information
    """
    if op.gate == cirq.CNOT and not transversal_cnot:
        path_patches = layout.route_cnot(*op.qubits)
        num_qubits = len(path_patches)
        return [
            lsp.Merge(num_qubits=num_qubits - 1, smooth=True).on(*path_patches[:-1]),
            lsp.Split(partitions=[1, len(path_patches[:-1]) - 1], smooth=True).on(
                *path_patches[:-1]
            ),
            lsp.Merge(num_qubits=num_qubits - 1, smooth=False).on(*path_patches[1:]),
            lsp.Split(partitions=[1] * (len(path_patches[1:])), smooth=False).on(*path_patches[1:]),
        ]
    if _requires_resource(op, transversal_cnot):
        return teleport_resource(op, layout)
    raise ValueError(
        f"Invalid Op for {'transversal' if transversal_cnot else 'non-transversal'} gate: {op.gate}"
    )


def teleport_resource(op: cirq.Operation, layout: Layout) -> list[cirq.Operation]:
    distil_t = layout.distil and op in cirq.GateFamily(cirq.T)
    distil_ccz = layout.distil and op in cirq.GateFamily(cirq.CCZ)
    cultivate_t = (not layout.distil) and op in cirq.GateFamily(cirq.T)
    cultivate_s = op in cirq.GateFamily(cirq.S)
    if distil_t:
        ftype = "t"
        prep_gate = lsp.Distil("T")
        correction = cirq.S
    elif cultivate_t:
        ftype = "t"
        prep_gate = lsp.Cultivate(pi / 4)
        correction = cirq.S
    elif cultivate_s:
        ftype = "s"
        prep_gate = lsp.Cultivate(pi / 2)
        correction = cirq.Z
    elif distil_ccz:
        ftype = "ccz"
        prep_gate = lsp.Distil("CCZ")
        # Since CNOT is the logical primitve, we use conjugation here
        correction = [
            *cirq.H.on_each(*op.qubits),
            *cirq.X.on_each(*op.qubits),
            *cirq.CNOT.on_each(*combinations(op.qubits, 2)),
            *cirq.H.on_each(*op.qubits),
        ]
    else:
        raise ValueError(f"Invalid resource encountered: {op.gate}")
    available_factories = layout.available_factories(ftype)
    all_factories = layout.all_factories(ftype)
    operations = []
    if not available_factories:
        if distil_t or distil_ccz:
            operations += [
                prep_gate.on(*layout.distillation_block(factory)) for factory in all_factories
            ]
        else:
            operations += [prep_gate.on(*factory) for factory in all_factories]
        layout.reload_factories(ftype=ftype)
    # These should be tuples of qubits
    routed_factory = layout.nearest_factory(op.qubits, ftype=ftype)
    cnots, measurements, resets = [], [], []
    corrections = correction if isinstance(correction, list) else [correction.on(*op.qubits)]
    for factory_qubit, program_qubit in zip(routed_factory, op.qubits):
        cnots.append(cirq.CNOT.on(factory_qubit, program_qubit))
        measurements.append(cirq.MeasurementGate(1, key="").on(factory_qubit))
        resets.append(cirq.ResetChannel().on(factory_qubit))
    operations += [
        cirq.Moment(cnots),
        cirq.Moment(measurements),
        cirq.Moment(resets),
        *corrections,
    ]
    return operations


def handle_idling(
    circuit: cirq.Circuit,
    layout: Layout,
    with_barriers: bool,
    rounds: int,
    verbose=0,
) -> cirq.Circuit:
    """Helper function for the compiler that handles idling. This way we can experiment with different kinds of idling or even turn it off entirely.
    This function is still a work in progress, but it is likely to take the form of various compiler passes.
    """
    # TODO: This pass is a main bottleneck for larger experiments, so make it faster
    # Assemble Qubits that will be subject to Idling
    G = layout.layout_graph
    logical_qubits = list(node for node in G.nodes if G.nodes[node]["patch_type"] == "data")
    t_factories = list(
        node
        for node in G.nodes
        if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
    )
    s_factories = list(
        node
        for node in G.nodes
        if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
    )
    non_ancillas = logical_qubits + s_factories + t_factories
    # Ensures no idling happens on qubits that are not used in the circuit
    # This is a bit faster
    non_ancillas = {q for op in circuit.all_operations() for q in op.qubits if q in non_ancillas}

    # Build circuit where Syndrome Extract is performed on Idling qubits that are not being acted upon
    # Split moments are treated separately because they can always get absorbed into the previous moment
    total = len(circuit)
    tstart = time()

    se = lsp.SyndromeExtract(1, rounds)

    def _map_func(moment, moment_idx):
        if verbose > 0:
            knock_off_tqdm(moment_idx=moment_idx, total=total, tstart=tstart, message="Idling:")
        if all(isinstance(gate.gate, (Barrier, lsp.Split)) for gate in moment):
            return moment
        if moment_idx == 0 or sum(
            isinstance(gate.gate, lsp.SyndromeExtract) for gate in moment
        ) == len(non_ancillas):
            # This is stateprep or a moment equivalent to idling, so we do not need to add idling
            return moment

        idling_qubits = non_ancillas - moment.qubits

        moment = cirq.Moment(*moment, *se.on_each(*idling_qubits), _flatten_contents=False)

        if with_barriers:
            return [moment, cirq.Moment(barrier(*circuit.all_qubits()))]

        return moment

    circuit_with_idling = cirq.map_moments(circuit, _map_func)

    return circuit_with_idling


def post_op_syndrome_extraction(
    circuit: cirq.Circuit,
    with_barriers: bool,
    movement: bool,
    rounds: int,
    verbose: int = 0,
) -> cirq.Circuit:
    """For movement, it has been suggested that we just do syndrome extraction (for a single round) right after a logical operations."""
    # Allowing a little bit of flexibility on what we want to correct
    # Might even want to add Lattice Primitives, but there aren't many (any?) that are not implicitly corrected
    ops_to_correct = [
        cirq.CNOT,
        cirq.S,
        # cirq.X,
        # cirq.Z,
    ]
    if movement:
        ops_to_correct.append(cirq.H)

    syndrom_extract = lsp.SyndromeExtract(1, rounds)
    barrier = css.barrier(*sorted(circuit.all_qubits()))

    total = len(circuit)
    tstart = time()

    def _map_func(op: cirq.Operation, moment_idx: int) -> Iterator[cirq.Operation]:
        if verbose:
            knock_off_tqdm(
                moment_idx=moment_idx,
                total=total,
                tstart=tstart,
                message="Post-Op Correction:",
            )

        yield op

        if with_barriers and not isinstance(op.gate, Barrier):
            yield barrier

        qubits_to_correct = [
            q
            for q in op.qubits
            if op.gate in ops_to_correct or isinstance(op.gate, cirq.MeasurementGate)
        ]
        if qubits_to_correct:
            yield from syndrom_extract.on_each(*qubits_to_correct)

            if with_barriers:
                yield barrier

    return cirq.map_operations_and_unroll(circuit, _map_func, raise_if_add_qubits=False)


def validate_ops(circuit: cirq.Circuit, verbose: int = 1):
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
        for op in tqdm(circuit.all_operations(), total=total_ops, disable=not verbose)
    ):
        raise ValueError("This compiler only handles Clifford + T + CCZ circuits")


def _decompose_to_primitives(
    circuit: cirq.Circuit,
    layout: Layout,
    arc: Architecture,
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    primitives = cirq.Gateset(
        *(cirq.GateFamily(g._gate, ignore_global_phase=False) for g in arc.primitives.gates)
    )
    transversal_cnot = cirq.CX in primitives

    def _map_fn(op: cirq.Operation) -> list[cirq.Operation]:
        return replace_cirq_op(op=op, layout=layout, transversal_cnot=transversal_cnot)

    # TODO: can we turn layout into a decomposition_context?
    ops = cirq.decompose(
        circuit,
        intercepting_decomposer=_map_fn,
        keep=primitives.__contains__,
    )
    return cirq.Circuit(ops)


def add_moves(
    circuit: cirq.Circuit,
    zone_ops: cirq.Gateset,
    alley_ops: cirq.Gateset,
    verbose: int = 0,
) -> cirq.Circuit:
    """Handles replacement moves for both alley movement and interaction zone movement"""
    total = len(circuit)
    tstart = time()

    def map_func(op, moment_idx):
        if verbose:
            knock_off_tqdm(
                moment_idx=moment_idx,
                total=total,
                tstart=tstart,
                message="Adding Qubit Movement:",
            )
        if op not in zone_ops and op not in alley_ops:
            yield op
        else:
            op_qubits = list(op.qubits)
            zone_type = None
            if op.gate in zone_ops:
                zone_type = "interact" if op.gate == cirq.CNOT else "measure"
            move_op = (
                partial(lsp.Move(zone=zone_type).on)
                if zone_type is None
                else partial(lsp.Move(zone=zone_type).on_each)
            )
            yield move_op(*op_qubits)
            yield op
            yield move_op(*op_qubits[::-1])

    return cirq.map_operations_and_unroll(circuit, map_func)


def ft_compile(
    layout: Layout,
    arc: Architecture,
    verbose: int = 1,
    with_barriers: bool = False,
    num_threads: int = 1,
    skip_validation: bool = False,
) -> cirq.Circuit:
    """Basic read/replace compiler that converts a cirq Circuit over the Clifford + T + CCZ gateset to a cirq circuit of primitives.
    The layout input contains the input circuit and information about any routing that might be necessary during the compilation process.
    The architecture input contains information about what primtives are accessible to the compiler and which extra passes should be added to the primitive circuit.
    The passes available are post op correction and idling.
    The architecture is also the source of information for how many rounds of syndrome extraction should be performed when syndrome extraction is called for.
    """
    # TODO: Aligning left results in circuits that have are more expensive in terms of circuit time than not aligning left. This is probably the result of requesting a layer of parallel cultivations but realigning so the expensive cultivation operations become spread out over multiple moments. It is currently unclear if aligning left is correct or not in general, but the specific tests for ft_compile very much rely on it...
    layout = copy.deepcopy(layout)
    layout.reset_graph()
    G = layout.layout_graph

    circuit = layout.mapped_circuit
    if verbose > 1:
        print("Validating Circuit Operations")
    if skip_validation:  # pragma: no cover
        print("Validation Turned Off")
    else:
        validate_ops(circuit, verbose=verbose)

    circuit = _decompose_to_primitives(circuit, layout=layout, arc=arc)
    if verbose > 1:
        verbose_list = [list(moment.operations) for moment in circuit.moments]

    # Handling State Prep
    # In a more optimized world this could happen the moment before the first logical operation
    logical_qubits = [node for node in G.nodes if G.nodes[node]["patch_type"] == "data"]
    state_prep = cirq.Circuit(lsp.SyndromeExtract(1, rounds=arc.rounds).on_each(*logical_qubits))
    if with_barriers:
        state_prep += css.barrier(*sorted(circuit.all_qubits()))
    circuit = state_prep + circuit

    if arc.post_op_correction:
        circuit = post_op_syndrome_extraction(
            circuit=circuit,
            movement=arc.movement,
            with_barriers=with_barriers,
            rounds=arc.rounds,
            verbose=verbose,
        )

    if arc.idling:
        if num_threads == 1:
            circuit = handle_idling(
                circuit=circuit,
                layout=layout,
                with_barriers=with_barriers,
                rounds=arc.rounds,
                verbose=verbose,
            )
        else:  # pragma: no cover
            warn("Parallelization is untested. Use at your own peril")
            from resource_estimation.ftqc.compile_ftqc_parallel import handle_idling_parallel

            circuit = handle_idling_parallel(
                circuit=circuit,
                layout=layout,
                rounds=arc.rounds,
                num_threads=num_threads,
            )

    if arc.zone_ops is not None or arc.alley_ops is not None:
        zone_ops = arc.zone_ops if arc.zone_ops is not None else cirq.Gateset()
        alley_ops = arc.alley_ops if arc.alley_ops is not None else cirq.Gateset()
        circuit = add_moves(
            circuit=circuit, verbose=verbose, zone_ops=zone_ops, alley_ops=alley_ops
        )

    if verbose > 1:
        return (verbose_list, circuit)

    return circuit
