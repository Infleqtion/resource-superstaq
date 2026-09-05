import pytest
import os
from pathlib import Path
from resource_estimation.data.cultivate_json import cultivate_json, _format_counts_dict
from resource_estimation.typing import CountsDict
import json
import cirq


def test_cultivate_json(tmp_path: Path) -> None:
    savefile = tmp_path / '_cultivate_costs.json'
    # Don't run all the way to 25
    cultivate_json(savefile=savefile, max_dist=3)
    with open(savefile, 'r') as f:
        saved_costs = json.load(f)
    assert len(saved_costs) == 1
    assert 'gidney' in saved_costs['3']
    assert 'yale' in saved_costs['3']


def test_format_counts_dict() -> None:
    example_counts_dict = CountsDict(
        serial={cirq.ResetChannel: 10, cirq.CZ: 100},
        parallel={cirq.ResetChannel: 1, cirq.CZ: 5}
    )
    expected_reformatting = {
        'serial': {'ResetChannel': 10, 'CZ': 100},
        'parallel': {'ResetChannel': 1, 'CZ': 5}
    }
    returned_reformatting = _format_counts_dict(example_counts_dict)
    assert expected_reformatting == returned_reformatting
