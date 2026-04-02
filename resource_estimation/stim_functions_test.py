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
from cultiv import make_end2end_cultivation_circuit, make_folded_transversal_circuit
from resource_estimation.stim_functions import (
    count_stim_resources,
    cultivate,
    load_saved_cost,
)
import stim


@pytest.fixture
def example_gidney():
    return make_end2end_cultivation_circuit(
        dsurface=7, dcolor=3, basis="Y", r_growing=1, r_end=7, inject_style="unitary"
    )


@pytest.fixture
def example_yale():
    return make_folded_transversal_circuit(
        noise_strength=0.0001,  # Required argument that adds noise moments
        dfinal=7,
        ghz_size=3,  # Should this parameter be more configurable?
        latter_rounds=3,
        prep="hookinj",
        ps_on_d3=1,
    ).without_noise()


def test_known_gidney(example_gidney):
    costs = count_stim_resources(example_gidney)
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


def test_saved_gidney(example_gidney):
    # These should all be the same
    saved_cost = load_saved_cost(dsurface=7, op_key="cultivate", style="gidney")
    cultivate_cost = cultivate(dsurface=7, fold=False, for_test=True)
    counted_cost = count_stim_resources(stim_circuit=example_gidney)
    assert saved_cost == counted_cost
    assert cultivate_cost == counted_cost


def test_known_yale(example_yale):
    costs = count_stim_resources(example_yale)
    expected_parallel_costs = {
        cirq.ResetChannel: 12,
        cirq.CZ: 47,
        cirq.MeasurementGate: 10,
        cirq.PhasedXZGate: 23,
    }
    expected_serial_costs = {
        cirq.ResetChannel: 275,
        cirq.CZ: 819,
        cirq.MeasurementGate: 239,
        cirq.PhasedXZGate: 258,
    }
    assert costs["parallel"] == expected_parallel_costs
    assert costs["serial"] == expected_serial_costs


def test_saved_yale(example_yale):
    # These should all be the same
    saved_cost = load_saved_cost(dsurface=7, op_key="cultivate", style="yale")
    cultivate_cost = cultivate(dsurface=7, fold=True, for_test=True)
    counted_cost = count_stim_resources(stim_circuit=example_yale)
    assert saved_cost == counted_cost
    assert cultivate_cost == counted_cost


def test_error_handling():
    bad_circuit = stim.Circuit("CZSWAP 5 6")
    with pytest.raises(ValueError, match="Unknown Instruction"):
        _ = count_stim_resources(bad_circuit)
    with pytest.raises(ValueError, match="Style cannot be None for cultivation"):
        _ = load_saved_cost(dsurface=7, op_key="cultivate")


def test_cultivation_low_distance_warning():
    # Just trigger the impossible branch once
    with pytest.warns(UserWarning, match="d<7"):
        cultivate(
            dsurface=5,
            for_test=True,
            fold=False,
        )
