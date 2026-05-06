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
import cirq
import pytest
from cultiv import make_end2end_cultivation_circuit, make_cirq_circuits
from resource_estimation.stim_functions import (
    count_stim_resources,
    cultivate,
    load_saved_cost,
)
import stim


@pytest.fixture
def gidney3():
    return make_end2end_cultivation_circuit(
        dsurface=7, dcolor=3, basis="Y", r_growing=1, r_end=7, inject_style="unitary"
    )


@pytest.fixture
def gidney5():
    return make_end2end_cultivation_circuit(
        dsurface=11, dcolor=5, basis="Y", r_growing=1, r_end=11, inject_style="unitary"
    )


@pytest.fixture
def yale3():
    return make_cirq_circuits.make_cirq_circuit(code_distance=7, fault_distance=3)


@pytest.fixture
def yale5():
    return make_cirq_circuits.make_cirq_circuit(code_distance=11, fault_distance=5)


def test_known_gidney(gidney3):
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
    assert costs["parallel"] == expected_parallel_costs
    assert costs["serial"] == expected_serial_costs


@pytest.mark.parametrize("fault_distance", (3, 5))
def test_saved_gidney(gidney3, gidney5, fault_distance):
    example_gidney = gidney3 if fault_distance == 3 else gidney5
    dsurface = 2 * fault_distance + 1
    saved_cost = load_saved_cost(
        dsurface=dsurface, op_key="cultivate", style="gidney", fault_distance=fault_distance
    )
    cultivate_cost = cultivate(
        dsurface=dsurface, fold=False, for_test=True, fault_distance=fault_distance
    )
    counted_cost = count_stim_resources(stim_circuit=example_gidney)
    assert saved_cost == counted_cost
    assert cultivate_cost == counted_cost


# def test_known_yale(example_yale):
#    costs = count_stim_resources(example_yale)
#    expected_parallel_costs = {
#        cirq.ResetChannel: 12,
#        cirq.CZ: 47,
#        cirq.MeasurementGate: 10,
#        cirq.PhasedXZGate: 23,
#    }
#    expected_serial_costs = {
#        cirq.ResetChannel: 275,
#        cirq.CZ: 819,
#        cirq.MeasurementGate: 239,
#        cirq.PhasedXZGate: 258,
#    }
#    assert costs["parallel"] == expected_parallel_costs
#    assert costs["serial"] == expected_serial_costs


@pytest.mark.parametrize("fault_distance", (3, 5))
def test_saved_yale(yale3, yale5, fault_distance):
    # There is no stim circuit for this cultivation circuit, so there are only saved and generated costs
    saved_cost = load_saved_cost(
        dsurface=2 * fault_distance + 1,
        op_key="cultivate",
        style="yale",
        fault_distance=fault_distance,
    )
    cultivate_cost = cultivate(
        dsurface=2 * fault_distance + 1, fold=True, for_test=True, fault_distance=fault_distance
    )
    assert saved_cost == cultivate_cost


def test_error_handling():
    bad_circuit = stim.Circuit("CZSWAP 5 6")
    with pytest.raises(ValueError, match="Unknown Instruction"):
        _ = count_stim_resources(bad_circuit)
    with pytest.raises(ValueError, match="Style cannot be None for cultivation"):
        _ = load_saved_cost(dsurface=7, op_key="cultivate")
    with pytest.raises(ValueError, match="cannot be None"):
        _ = load_saved_cost(dsurface=7, op_key="cultivate", style="yale")
    with pytest.raises(ValueError, match="fault_distance values 3 and 5"):
        _ = cultivate(dsurface=15, fault_distance=7, fold=False, for_test=False)


def test_cultivation_low_distance_warning():
    # Just trigger the impossible branch once
    with pytest.warns(UserWarning, match="Returning result for d=7"):
        cultivate(
            dsurface=5,
            for_test=True,
            fold=False,
            fault_distance=3,
        )
    with pytest.warns(UserWarning, match="Returning result for d=11"):
        cultivate(
            dsurface=7,
            for_test=True,
            fold=False,
            fault_distance=5,
        )
