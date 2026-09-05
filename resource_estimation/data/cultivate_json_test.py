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
from pathlib import Path

import cirq

from resource_estimation.data.cultivate_json import _format_counts_dict, cultivate_json
from resource_estimation.typing import CountsDict


def test_cultivate_json(tmp_path: Path) -> None:
    savefile = tmp_path / "_cultivate_costs.json"
    # Don't run all the way to 25
    cultivate_json(savefile=savefile, max_dist=3)
    with open(savefile, "r") as f:
        saved_costs = json.load(f)
    assert len(saved_costs) == 1
    assert "gidney" in saved_costs["3"]
    assert "yale" in saved_costs["3"]


def test_format_counts_dict() -> None:
    example_counts_dict = CountsDict(
        serial={cirq.ResetChannel: 10, cirq.CZ: 100}, parallel={cirq.ResetChannel: 1, cirq.CZ: 5}
    )
    expected_reformatting = {
        "serial": {"ResetChannel": 10, "CZ": 100},
        "parallel": {"ResetChannel": 1, "CZ": 5},
    }
    returned_reformatting = _format_counts_dict(example_counts_dict)
    assert expected_reformatting == returned_reformatting
