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
import itertools
from dataclasses import dataclass

import cirq

import resource_estimation.ftqc.codepatch as cpa
import resource_estimation.ftqc.layout as lyt


@dataclass
class Context:
    """Initial concept data structure designed to help take responsibilities away from Layout and into the compiler"""

    roles: dict[int, int]  # Patch id to integer role. {2: ancilla, 3: S, 4: T, 5: CCZ}
    qid_to_pid: dict[cirq.Qid, int]  # qid to patch id
    pid_to_qid: dict[int, cirq.Qid]  # patch id to qid
    factories: dict[
        tuple[int, ...], int
    ]  # The current mapping of patch ids to factory types that are available
    blocks: dict[tuple[int, ...], tuple[int, ...]]
    transversal_cnot: bool  # Can we avoid doing lattice surgery
    distil_t: bool  # Could be toggled
    cult_t: bool  # Probably conflicts with the above
    dynamic: bool  # I plan to remove this...
    num_s_factories: int = 0
    num_t_factories: int = 0
    num_ccz_factories: int = 0
    post_op_syndrome_extraction: bool = True
    idling: bool = False


def build_movement_context(
    circuit: cirq.Circuit,
    layout: lyt.Layout,
    t_cultivation_factories: int = 0,
    t_distillation_factories: int = 0,
    ccz_distillation_factories: int = 0,
    dynamic: bool = False,
    **kwargs,
) -> Context:
    """
    A replacement for much of the logic that was in the various flavors of `MovementLayout` previously.
    Instead of holding information about allocated qubits and roles within the Layout class, we want to let the compiler manage this context.
    This function should represent the outcome of several choices no that context.
    It takes in a circuit and operates under the assumption that all program qubits will require surface code patches.
    It ensures that the Layout has been built with enough space to hold all of the expected patches.
    `t_cultivation_factories` is the number of factories cultivating T states (1 patch per factory)
    `t_distillation_factories` is the number of factories distilling T states (31 patches per factory)
        It includes extra space for cultivating T states within its block, like the old version
        An error is raised if both types of factories is requested
    `ccz_distillation_factories` is the number of factories distilling CCZ states (23 patches per factory)
        It also includes extra space for cultivating T states within its block, like the old version
    # `arc` is the Architecture--currently only used to supply the availability of transversal CNOT
    `dynamic` is a compiler keyword argument that should eventually become obselete
    The main role of this function is to create an initial mapping between patch ids and qids and to assign patch ids to roles before compilation begins.
    In a better world, this mapping and allocation would happen dynamically during compilation, but the first step is getting the context out of the Layout class.
    """
    if t_cultivation_factories and t_distillation_factories:
        raise ValueError("Current compilation does not support two types of T generation")
    surface_code_program_qubits = cirq.num_qubits(circuit)
    total_surface_code_patches = (
        surface_code_program_qubits  # Logical qubits required by the circuit
        + t_cultivation_factories  # Patches dedicated to T state cultivation
        + 31 * t_distillation_factories  # 15 ancilla + 15 Cult + 1 Out
        + 23 * ccz_distillation_factories  # 8 Cult + 12 Ancilla + 3 Out
    )
    patches_in_layout = len(
        [
            p
            for p in layout.placements.values()
            if isinstance(p.element, cpa.RotatedSurfaceCodePatch)
        ]
    )
    if patches_in_layout != total_surface_code_patches:
        raise ValueError(
            f"Patches in Layout ({patches_in_layout}) does not equal patches expected ({total_surface_code_patches})"
        )
    qid_to_pid = {}
    roles = {}
    counter = 0
    for qid in circuit.all_qubits():
        qid_to_pid[qid] = counter
        roles[counter] = 1
        counter += 1
    for idx in range(t_cultivation_factories):
        qid = cirq.NamedQubit(f"ct{idx}")
        qid_to_pid[qid] = counter
        roles[counter] = 4
        counter += 1
    for idx in range(t_distillation_factories * 31):
        qid = cirq.NamedQubit(f"dt{idx}")
        qid_to_pid[qid] = counter
        roles[counter] = 4
        counter += 1
    for idx in range(ccz_distillation_factories * 23):
        qid = cirq.NamedQubit(f"ccz{idx}")
        qid_to_pid[qid] = counter
        roles[counter] = 5
        counter += 1
    pid_to_qid = {pid: qid for qid, pid in qid_to_pid.items()}
    return Context(
        roles=roles,
        qid_to_pid=qid_to_pid,
        pid_to_qid=pid_to_qid,
        factories={},
        blocks={},
        distil_t=t_distillation_factories > 0,
        cult_t=t_cultivation_factories > 0,
        dynamic=dynamic,
        transversal_cnot=True,
        num_ccz_factories=ccz_distillation_factories,
        num_t_factories=t_cultivation_factories or t_distillation_factories,
    )


def build_column_context(
    circuit: cirq.Circuit,
    layout: lyt.Layout,
    surface_code_distance: int,
    # arc: arch.Architecture,
    dynamic: bool = False,
    **kwargs,
) -> Context:
    """
    This function largely replaces the `ColumnLayout`.
    It assigns qubit roles according to a layered procedure, where each logical qubit gets an S factory and a T factory.
    It lazily maps QIDs in `circuit` to patches in the layout.
    """
    expected_rows = cirq.num_qubits(circuit) + cirq.num_qubits(circuit) % 2
    expected_cols = 7
    patches_in_layout = len(
        [
            p
            for p in layout.placements.values()
            if isinstance(p.element, cpa.RotatedSurfaceCodePatch)
        ]
    )
    if patches_in_layout != expected_rows * expected_cols:
        raise ValueError(
            f"Patches in Layout ({patches_in_layout}) does not equal patches expected ({expected_rows * expected_cols})"
        )
    expected_length = 2 * surface_code_distance - 1

    def get_patch_id_at(logical_row, logical_col):
        anchor_row, anchor_col = logical_row * expected_length, logical_col * expected_length
        match_placements = {
            obj
            for obj in layout.placements.values()
            if obj.row == anchor_row and obj.col == anchor_col
        }
        assert len(match_placements) == 1
        return match_placements.pop().element.patch_id

    qid_to_pid = {}
    roles = {}
    role_cycle = itertools.cycle(
        [
            3,
            2,
            1,
            2,
            1,
            2,
            3,  # SAQAQAS
            4,
            2,
            2,
            2,
            2,
            2,
            4,  # TAAAAAT
        ]
    )
    qid_iter = iter(circuit.all_qubits())

    for logical_row in range(expected_rows):
        for logical_col in range(expected_cols):
            pid = get_patch_id_at(logical_row, logical_col)
            role_index = next(role_cycle)

            roles[pid] = role_index
            if role_index == 1:
                try:
                    qid = next(qid_iter)
                except StopIteration:  # Overwrite non-existent logical qubit
                    qid = cirq.NamedQubit(f"{logical_row}-{logical_col}-{2}")
                    roles[pid] = 2
            else:
                qid = cirq.NamedQubit(f"{logical_row}-{logical_col}-{role_index}")
            qid_to_pid[qid] = pid
    pid_to_qid = {pid: qid for qid, pid in qid_to_pid.items()}
    return Context(
        roles=roles,
        qid_to_pid=qid_to_pid,
        pid_to_qid=pid_to_qid,
        factories={},
        blocks={},
        distil_t=False,
        cult_t=True,
        dynamic=dynamic,
        transversal_cnot=False,
        num_ccz_factories=0,
        num_t_factories=2 * expected_rows,
        num_s_factories=2 * expected_rows,
    )
