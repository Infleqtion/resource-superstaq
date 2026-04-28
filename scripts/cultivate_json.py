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
import sys
from pathlib import Path
from typing import Literal
import cirq

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import os

import cultiv
from resource_estimation.stim_functions import count_stim_resources, STR2GATE
from tqdm import tqdm


GATE2STR = {v: k for k, v in STR2GATE.items()}


def format_cost_dict(
    cost_dict: dict[Literal["serial", "parallel"], dict[cirq.Gate], int],
) -> dict[Literal["serial", "parallel"], dict[str, int]]:
    """
    Converts cost dictionaries from `count_stim_resources` from cirq gate to string format
    """
    refomatted = {
        "serial": {GATE2STR[k]: v for k, v in cost_dict["serial"].items()},
        "parallel": {GATE2STR[k]: v for k, v in cost_dict["parallel"].items()},
    }
    return refomatted


if __name__ == "__main__":
    resources_dict = {}
    for d in tqdm(range(3, 26, 2)):
        # Establish official resources as basis
        cnot = cultiv.make_surface_code_cnot(distance=d, basis="Z")
        memory_d_rounds = cultiv.make_surface_code_memory_circuit(dsurface=d, rounds=d, basis="Z")
        memory_1_round = cultiv.make_surface_code_memory_circuit(dsurface=d, rounds=1, basis="Z")
        gidney_cultiv3 = cultiv.make_end2end_cultivation_circuit(
            dcolor=3,
            dsurface=7 if d <= 7 else d,
            basis="Y",
            r_growing=1,
            r_end=7 if d <= 7 else d,
            inject_style="unitary",
        )
        gidney_cultiv5 = cultiv.make_end2end_cultivation_circuit(
            dcolor=3,
            dsurface=7 if d <= 7 else d,
            basis="Y",
            r_growing=1,
            r_end=7 if d <= 7 else d,
            inject_style="unitary",
        )
        yale_cultiv3 = cultiv.make_cirq_circuits.make_cirq_circuit(
            code_distance=max(7, d), fault_distance=3
        )
        yale_cultiv5 = cultiv.make_cirq_circuits.make_cirq_circuit(
            code_distance=max(7, d), fault_distance=5
        )

        # Count up the resources and format the results
        cnot_costs = format_cost_dict(count_stim_resources(stim_circuit=cnot))
        memory_d_rounds_costs = format_cost_dict(count_stim_resources(stim_circuit=memory_d_rounds))
        memory_1_round_costs = format_cost_dict(count_stim_resources(stim_circuit=memory_1_round))
        gidney_cultiv3_costs = format_cost_dict(count_stim_resources(stim_circuit=gidney_cultiv3))
        gidney_cultiv5_costs = format_cost_dict(count_stim_resources(stim_circuit=gidney_cultiv5))
        yale_cultiv3_costs = format_cost_dict(
            cultiv.make_cirq_circuits.dirty_count(circuit=yale_cultiv3)
        )
        yale_cultiv5_costs = format_cost_dict(
            cultiv.make_cirq_circuits.dirty_count(circuit=yale_cultiv5)
        )

        # Add the costs to the dictionary
        resources_dict[d] = {
            "cnot": cnot_costs,
            "memory_d_rounds": memory_d_rounds_costs,
            "memory_1_round": memory_1_round_costs,
            "cultivate": {
                "gidney": {3: gidney_cultiv3_costs, 5: gidney_cultiv5_costs},
                "yale": {3: yale_cultiv3_costs, 5: yale_cultiv5_costs},
            },
        }
        # Save at each iteration
        with open(
            os.path.dirname(os.path.abspath(__file__)) + "/../data/cultivate_costs.json", "w"
        ) as f:
            json.dump(resources_dict, f, indent=4)
