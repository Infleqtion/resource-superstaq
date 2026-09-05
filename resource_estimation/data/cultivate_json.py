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
import sys
from pathlib import Path

import cultiv
import tqdm

from resource_estimation.ftqc.stim_functions import count_stim_resources
from resource_estimation.typing import (
    CountsDict,
    StrCounts,
    GATE2STR
)

def _format_counts_dict(
    counts_dict: CountsDict,
) -> dict[str, StrCounts]:
    """
    Converts cost dictionaries from `count_stim_resources` from cirq gate to string format
    """
    reformatted = {
        "serial": {GATE2STR[k]: v for k, v in counts_dict.serial.items()},
        "parallel": {GATE2STR[k]: v for k, v in counts_dict.parallel.items()},
    }
    return reformatted


def cultivate_json(savefile: Path | str | None = None, max_dist: int = 25) -> None:
    """Saves cultivation costs in a json for later use"""
    # I don't know how to test this properly
    if savefile is None:  # pragma: no cover
        savefile = str(Path(__file__).resolve().parents[2] / "resource_estimation" / "data"/ "_cultivate_costs.json" )

    resources_dict = {}
    for d in tqdm.tqdm(range(3, max_dist+1, 2)):
        # Establish official resources as basis
        gidney_cultiv3 = cultiv.make_end2end_cultivation_circuit(
            dcolor=3,
            dsurface=max(7, d),
            basis="Y",
            r_growing=1,
            r_end=max(7, d),
            inject_style="unitary",
        )
        gidney_cultiv5 = cultiv.make_end2end_cultivation_circuit(
            dcolor=5,
            dsurface=max(11, d),
            basis="Y",
            r_growing=1,
            r_end=max(11, d),
            inject_style="unitary",
        )
        yale_cultiv3 = cultiv.make_cirq_circuits.make_cirq_circuit(
            code_distance=max(7, d),
            fault_distance=3,
        )
        yale_cultiv5 = cultiv.make_cirq_circuits.make_cirq_circuit(
            code_distance=max(11, d),
            fault_distance=5,
        )

        # Count up the resources and format the results
        gidney_cultiv3_costs = _format_counts_dict(count_stim_resources(stim_circuit=gidney_cultiv3))
        gidney_cultiv5_costs = _format_counts_dict(count_stim_resources(stim_circuit=gidney_cultiv5))
        yale_cultiv3_costs = _format_counts_dict(
            CountsDict(**cultiv.make_cirq_circuits.dirty_count(circuit=yale_cultiv3)),
        )
        yale_cultiv5_costs = _format_counts_dict(
            CountsDict(**cultiv.make_cirq_circuits.dirty_count(circuit=yale_cultiv5)),
        )

        # Add the costs to the dictionary
        resources_dict[d] = {
            "gidney": {3: gidney_cultiv3_costs, 5: gidney_cultiv5_costs},
            "yale": {3: yale_cultiv3_costs, 5: yale_cultiv5_costs},
        }
        # Save at each iteration
        with open(savefile, "w") as f:
            json.dump(resources_dict, f, indent=4)