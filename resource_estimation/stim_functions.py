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
import os
import warnings
from collections import Counter

import cirq
import cultiv
import stim
from typing import Literal

STR2GATE = {
    "PhasedXZGate": cirq.PhasedXZGate,
    "QubitPermutationGate": cirq.QubitPermutationGate,
    "MeasurementGate": cirq.MeasurementGate,
    "CZ": cirq.CZ,
    "ResetChannel": cirq.ResetChannel,
    "CCZ": cirq.CCZ,
}


def count_stim_resources(stim_circuit: stim.Circuit) -> dict[str, Counter[cirq.Gate, int]]:
    """
    Parses stim circuit to count relevant operations and returns both parallel and serial costs
    """
    # A map from Stim operations to replacement physical operations
    op_map = {
        "CX": ("CZ",),
        "CY": ("CZ",),
        "CZ": ("CZ",),
        "S": ("PhasedXZGate",),
        "S_DAG": ("PhasedXZGate",),
        "MX": ("PhasedXZGate", "MeasurementGate"),
        "MY": ("PhasedXZGate", "MeasurementGate"),
        "M": ("MeasurementGate",),
        "R": ("ResetChannel",),
        "RX": ("ResetChannel", "PhasedXZGate"),
        "RY": ("ResetChannel", "PhasedXZGate"),
        "H": ("PhasedXZGate",),
        "I": (),
    }
    # Operations that are not tracked for the purpose of resource estimation
    ops_to_ignore = [
        "DETECTOR",
        "MPP",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
    ]
    total_serial = Counter(dict())
    total_parallel = Counter(dict())
    tick_total = Counter(
        dict()
    )  # Keeps partial total for different operations that can be done in parallel
    for instr in stim_circuit:
        if instr.name in ops_to_ignore:
            continue
        elif instr.name == "TICK":
            total_parallel += tick_total
            tick_total = Counter({})  # Reset moment counting
            continue
        elif instr.name == "REPEAT":
            repeats = instr.repeat_count
            one_round = count_stim_resources(instr.body_copy())
            total_serial += {k: v * repeats for k, v in one_round["serial"].items()}
            total_parallel += {k: v * repeats for k, v in one_round["parallel"].items()}
        elif instr.name not in op_map:
            raise ValueError(f"Unknown Instruction: {instr.name}")
        else:
            replacement = op_map[instr.name]
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
    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/../data/cultivate_costs.json", "r"
    ) as f:
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
    fold=False,
    for_test=False,
) -> Counter[cirq.Gate, int]:
    """
    Generates the physical qubit resources required for folded (Yale) or unfolded (Gidney)
    If the final patch size is less than 25 it reads from saved resources instead of calling the functions directly
    The `for_test` argument is to turn off the loading behvior for the purpose of testing
    """
    if dsurface < 7:
        warnings.warn("Cultivation code does not work with d<7. Returning result for d=7")
        dsurface = 7
    style = "yale" if fold else "gidney"
    if dsurface <= 25 and not for_test:
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
            dcolor=3,  # It might be possible to make this 5 now
            dsurface=dsurface,
            basis="Y",
            r_growing=1,
            r_end=dsurface,  # This parameter controls the number of times we a block of Reset -> 8 CX Moments -> Measure (Repeat)
            inject_style="unitary",
        )
        resources = count_stim_resources(stim_circuit=stim_circuit)
    return resources
