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
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal

import cirq
import cultiv
import stim

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

STR2GATE = {
    "PhasedXZGate": cirq.PhasedXZGate,
    "QubitPermutationGate": cirq.QubitPermutationGate,
    "MeasurementGate": cirq.MeasurementGate,
    "CZ": cirq.CZ,
    "ResetChannel": cirq.ResetChannel,
    "CCZ": cirq.CCZ,
}


_STIM_OP_MAP = {
    "CX": ("CZ",),
    "CY": ("CZ",),
    "CZ": ("CZ",),
    "SWAP": ("QubitPermutationGate",),
    "S": ("PhasedXZGate",),
    "S_DAG": ("PhasedXZGate",),
    "SQRT_X": ("PhasedXZGate",),
    "SQRT_X_DAG": ("PhasedXZGate",),
    "SQRT_Y": ("PhasedXZGate",),
    "SQRT_Y_DAG": ("PhasedXZGate",),
    "SQRT_Z": ("PhasedXZGate",),
    "SQRT_Z_DAG": ("PhasedXZGate",),
    "H": ("PhasedXZGate",),
    "H_XY": ("PhasedXZGate",),
    "H_XZ": ("PhasedXZGate",),
    "H_YZ": ("PhasedXZGate",),
    "C_XYZ": ("PhasedXZGate",),
    "C_ZYX": ("PhasedXZGate",),
    "X": ("PhasedXZGate",),
    "Y": ("PhasedXZGate",),
    "Z": ("PhasedXZGate",),
    "MX": ("PhasedXZGate", "MeasurementGate"),
    "MY": ("PhasedXZGate", "MeasurementGate"),
    "M": ("MeasurementGate",),
    "R": ("ResetChannel",),
    "RX": ("ResetChannel", "PhasedXZGate"),
    "RY": ("ResetChannel", "PhasedXZGate"),
    "I": (),
}

_STIM_OPS_TO_IGNORE = {
    "DETECTOR",
    "MPP",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
}


def count_stim_resources(
    stim_circuit: stim.Circuit, *, scheduling: Literal["ticks", "asap"] = "ticks"
) -> dict[str, Counter[cirq.Gate, int]]:
    """
    Parse a Stim circuit and return parallel and serial physical-operation costs.

    ``ticks`` preserves the historical behavior where TICK instructions define moments. ``asap``
    greedily schedules operations by their qubit dependencies and is intended for qLDPC's
    tick-free transversal circuits.
    """
    if scheduling == "asap":
        return _count_stim_resources_asap(stim_circuit)
    if scheduling != "ticks":
        raise ValueError(f"Unknown Stim scheduling mode: {scheduling!r}")

    total_serial = Counter(dict())
    total_parallel = Counter(dict())
    tick_total = Counter(
        dict()
    )  # Keeps partial total for different operations that can be done in parallel
    for instr in stim_circuit:
        if instr.name in _STIM_OPS_TO_IGNORE:
            continue
        elif instr.name == "TICK":
            total_parallel += tick_total
            tick_total = Counter({})  # Reset moment counting
            continue
        elif instr.name == "REPEAT":
            repeats = instr.repeat_count
            one_round = count_stim_resources(instr.body_copy(), scheduling="ticks")
            total_serial += {k: v * repeats for k, v in one_round["serial"].items()}
            total_parallel += {k: v * repeats for k, v in one_round["parallel"].items()}
        elif instr.name not in _STIM_OP_MAP:
            raise ValueError(f"Unknown Instruction: {instr.name}")
        else:
            replacement = _STIM_OP_MAP[instr.name]
            # Add up all the gates serially
            total_serial += {
                STR2GATE[gate_type]: len(instr.target_groups()) for gate_type in replacement
            }
            # Add new moments to current tick
            tick_total += {
                STR2GATE[gate_type]: 1
                for gate_type in replacement
                if STR2GATE[gate_type] not in tick_total
            }
    return {"serial": total_serial, "parallel": total_parallel}


def _count_stim_resources_asap(
    stim_circuit: stim.Circuit,
) -> dict[str, Counter[cirq.Gate, int]]:
    total_serial: Counter[cirq.Gate, int] = Counter()
    occupied_layers: dict[cirq.Gate, set[int]] = {}
    next_layer: dict[int, int] = {}
    barrier = 0

    for instr in stim_circuit.flattened():
        if instr.name in _STIM_OPS_TO_IGNORE:
            continue
        if instr.name == "TICK":
            barrier = max([barrier, *next_layer.values()])
            continue
        if instr.name not in _STIM_OP_MAP:
            raise ValueError(f"Unknown Instruction: {instr.name}")

        replacement = _STIM_OP_MAP[instr.name]
        for target_group in instr.target_groups():
            qubits = [target.value for target in target_group if target.is_qubit_target]
            layer = max([barrier, *(next_layer.get(qubit, barrier) for qubit in qubits)])
            for gate_name in replacement:
                gate = STR2GATE[gate_name]
                total_serial[gate] += 1
                occupied_layers.setdefault(gate, set()).add(layer)
                layer += 1
            for qubit in qubits:
                next_layer[qubit] = layer

    total_parallel = Counter(
        {gate: len(layers) for gate, layers in occupied_layers.items() if layers}
    )
    return {"serial": total_serial, "parallel": total_parallel}


def load_saved_cost(
    dsurface: int,
    op_key: Literal["cultivate", "cnot", "memory_d_rounds", "memory_1_round"],
    style: Literal[None, "gidney", "yale"] = None,
    fault_distance: Literal[None, 3, 5] = None,
) -> dict[Literal["serial", "parallel"], Counter[cirq.Gate, int]]:
    """
    Gets saved serial and parallel costs from the `cultivate_costs.json` file
    Converts saved strings to proper cirq gate objects
    """
    if op_key == "cultivate" and style is None:
        raise ValueError("Style cannot be None for cultivation")
    if op_key == "cultivate" and fault_distance is None:
        raise ValueError("Fault distance cannot be None for cultivation")
    with open(DATA_DIR / "cultivate_costs.json") as f:
        saved_costs = json.load(f)
    loaded_costs = (
        saved_costs[str(dsurface)][op_key][style][str(fault_distance)]
        if op_key == "cultivate"
        else saved_costs[str(dsurface)][op_key]
    )
    # Check to make sure there are no out of bounds gates saved
    assert all(k in STR2GATE for k in loaded_costs.get("serial"))
    assert all(k in STR2GATE for k in loaded_costs.get("parallel"))
    serial_cost = {STR2GATE[k]: v for k, v in loaded_costs["serial"].items()}
    parallel_cost = {STR2GATE[k]: v for k, v in loaded_costs["parallel"].items()}
    return {"serial": serial_cost, "parallel": parallel_cost}


def cultivate(
    dsurface: int,
    fault_distance: int,
    fold: bool = False,
    for_test: bool = False,
) -> dict[Literal["serial", "parallel"], Counter[cirq.Gate, int]]:
    """
    Generates the physical qubit resources required for folded (Yale) or unfolded (Gidney)
    If the final patch size is less than 25 it reads from saved resources instead of calling the functions directly
    The `for_test` argument is to turn off the loading behvior for the purpose of testing
    """
    if dsurface < 7 and fault_distance == 3:
        warnings.warn(
            "Code distance must be an odd value of at least 2 * fault_distance + 1. Returning result for d=7"
        )
        dsurface = 7
    if dsurface < 11 and fault_distance == 5:
        warnings.warn(
            "Code distance must be an odd value of at least 2 * fault_distance + 1. Returning result for d=11"
        )
        dsurface = 11
    style = "yale" if fold else "gidney"
    if dsurface <= 25 and not for_test:
        if fault_distance not in (3, 5):
            raise ValueError(
                "Saved cultivation costs are only available for fault_distance values 3 and 5."
            )
        return load_saved_cost(
            dsurface=dsurface, op_key="cultivate", style=style, fault_distance=fault_distance
        )
    if fold:
        resources = cultiv.make_cirq_circuits.dirty_count(
            cultiv.make_cirq_circuits.make_cirq_circuit(
                code_distance=dsurface, fault_distance=fault_distance
            )
        )
    else:
        stim_circuit = cultiv.make_end2end_cultivation_circuit(
            dcolor=fault_distance,
            dsurface=dsurface,
            basis="Y",
            r_growing=1,
            r_end=dsurface,  # This parameter controls the number of times we a block of Reset -> 8 CX Moments -> Measure (Repeat)
            inject_style="unitary",
        )
        resources = count_stim_resources(stim_circuit=stim_circuit)
    return resources
