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
from math import pi

import cirq
import pytest

import resource_estimation.ftqc.compile_ftqc as comp

# from resource_estimation.ftqc.layout import MovementDistillery
import resource_estimation.ftqc.layout as lyt
import resource_estimation.ftqc.mappers as mps
from resource_estimation.ftqc.architecture import DefaultMovement
from resource_estimation.ftqc.distil import distil_15_to_1, precompute_distil_cost
from resource_estimation.ftqc.lattice_surgery_primitives import Cultivate


def test_15_to_one() -> None:
    """Check to confirm that the compression technique agrees with the standard circuit"""
    circuit = distil_15_to_1()

    # There should be exactly 15 Cultivations
    assert sum(op in cirq.GateFamily(Cultivate(pi / 4)) for op in circuit.all_operations()) == 15

    # There should be 30 Measurements
    assert sum(op in cirq.GateFamily(cirq.MeasurementGate) for op in circuit.all_operations()) == 30

    # There should be 7*5 + 15 = 50 CNOT gates
    assert sum(op in cirq.GateFamily(cirq.CNOT) for op in circuit.all_operations()) == 50


@pytest.mark.parametrize(
    "interaction_zone, measure_zone", [(False, False), (False, True), (True, False), (True, True)]
)
def test_precompute_distil_cost(interaction_zone: bool, measure_zone: bool) -> None:
    empty_circuit = cirq.Circuit(cirq.I.on_each(*cirq.LineQubit.range(4)))
    arc = DefaultMovement()
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=cirq.num_qubits(empty_circuit),
        surface_code_distance=arc.d,
        t_cultivation_factories=0,
        t_distillation_factories=1,
        ccz_distillation_factories=1,
        interaction_zone=interaction_zone,
        measure_zone=measure_zone,
    )
    context = mps.build_movement_context(
        circuit=empty_circuit,
        layout=layout,
        t_cultivation_factories=0,
        t_distillation_factories=1,
        ccz_distillation_factories=1,
    )
    context = comp.add_zones_to_context(context, arc, layout)

    with pytest.raises(ValueError, match="Unknown distillation resource"):
        _ = precompute_distil_cost("Toffoli", layout=layout, arc=arc, context=context)  # type: ignore[arg-type]

    # Distil T and CCZ have the same critical path, but T cultivation moves further, so it should be more expensive
    # Cultivation is a subcomponent, so it should be faster than the Distillation implementations
    cult = arc._cultivate_t_cost.op_time
    t_distil = precompute_distil_cost("T", layout=layout, arc=arc, context=context).op_time
    ccz_distil = precompute_distil_cost("CCZ", layout=layout, arc=arc, context=context).op_time
    assert cult < ccz_distil < t_distil
