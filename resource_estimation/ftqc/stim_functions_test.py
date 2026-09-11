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
from typing import Literal

import cirq
import cultiv
import pytest
import stim

from resource_estimation.ftqc.stim_functions import (
    count_stim_resources,
    cultivate,
    load_saved_cost,
)


@pytest.fixture
def gidney3() -> stim.Circuit:
    return cultiv.make_end2end_cultivation_circuit(
        dsurface=7, dcolor=3, basis="Y", r_growing=1, r_end=7, inject_style="unitary"
    )


@pytest.fixture
def gidney5() -> stim.Circuit:
    return cultiv.make_end2end_cultivation_circuit(
        dsurface=11, dcolor=5, basis="Y", r_growing=1, r_end=11, inject_style="unitary"
    )


def test_known_gidney(gidney3: stim.Circuit) -> None:
    costs = count_stim_resources(gidney3)
    expected_parallel_costs = {
        cirq.ResetChannel: 13,
        cirq.CZ: 101,
        cirq.MeasurementGate: 12,
        cirq.PhasedXZGate: 28,
    }
    expected_serial_costs = {
        cirq.ResetChannel: 528,
        cirq.CZ: 1749,
        cirq.MeasurementGate: 472,
        cirq.PhasedXZGate: 535,
    }
    assert costs.parallel == expected_parallel_costs
    assert costs.serial == expected_serial_costs


@pytest.mark.parametrize("fault_distance", (3, 5))
def test_saved_gidney(
    gidney3: stim.Circuit, gidney5: stim.Circuit, fault_distance: Literal[3, 5]
) -> None:
    example_gidney = gidney3 if fault_distance == 3 else gidney5
    dsurface = 2 * fault_distance + 1
    cost_from_load = load_saved_cost(
        dsurface=dsurface,
        style="gidney",
        fault_distance=fault_distance,
    )
    cost_from_file = cultivate(
        dsurface=dsurface,
        fold=False,
        fault_distance=fault_distance,
        load_from_file=True,
    )

    cost_from_counts = count_stim_resources(stim_circuit=example_gidney)

    with pytest.warns(UserWarning, match="save cultivation costs"):
        cost_from_generate = cultivate(
            dsurface=dsurface,
            fold=False,
            fault_distance=fault_distance,
            load_from_file=False,
        )

    assert cost_from_load == cost_from_file == cost_from_counts == cost_from_generate


@pytest.mark.parametrize("fault_distance", (3, 5))
def test_saved_yale(fault_distance: Literal[3, 5]) -> None:
    # There is no stim circuit for this cultivation circuit, so there are only saved and generated costs
    dsurface = 2 * fault_distance + 1
    cost_from_load = load_saved_cost(
        dsurface=dsurface,
        style="yale",
        fault_distance=fault_distance,
    )

    cost_from_file = cultivate(
        dsurface=dsurface,
        fault_distance=fault_distance,
        fold=True,
        load_from_file=True,
    )

    with pytest.warns(UserWarning, match="save cultivation costs"):
        cost_from_generate = cultivate(
            dsurface=dsurface,
            fault_distance=fault_distance,
            fold=True,
            load_from_file=False,
        )
    assert cost_from_load == cost_from_file == cost_from_generate


def test_error_handling() -> None:
    bad_circuit = stim.Circuit("CZSWAP 5 6")
    with pytest.raises(ValueError, match="Unknown Instruction"):
        _ = count_stim_resources(bad_circuit)
    with pytest.raises(ValueError, match="fault_distance values 3 and 5"):
        _ = cultivate(dsurface=15, fault_distance=7, fold=False)  # type: ignore[arg-type]


def test_cultivation_low_distance_warning() -> None:
    # Just trigger the impossible branch once
    with pytest.warns(UserWarning, match="Returning result for d=7"):
        cultivate(
            dsurface=5,
            fold=False,
            fault_distance=3,
        )
    with pytest.warns(UserWarning, match="Returning result for d=11"):
        cultivate(
            dsurface=7,
            fold=False,
            fault_distance=5,
        )
