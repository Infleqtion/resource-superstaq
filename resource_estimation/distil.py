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
from collections import Counter
from typing import Literal
import cirq
# from compile_ftqc import ft_compile
# from architecture import Architecture
# from layout import MovementLayout
# from estimate import ResourceEstimator
import resource_estimation as res
from resource_estimation import architecture
from . import magic_state_distillation as msd
# import magic_state_distillation as msd

def distil(
        arch,
) -> dict[Literal['op_time', 'moment_cost', 'gate_cost'],
          Counter[cirq.Gate, int]]:
    """
    Generates the logical qubit resources required for 15-to-1 magic
    state distilation.
    """
    circuit = msd.msd_15_to_1()
    movement = arch.movement
    if movement:
        layout = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=15)
    else:
        layout = res.layout.FactorySandwich(input_circuit=circuit, num_t_factories=15, num_s_factories=1)
    primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=False)
    estimator = res.estimate.ResourceEstimator(arc=arch)
    parallel_gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=False)
    serial_gate_cost = estimator.serial_circuit_cost(primitive_circuit, pretty=False)
    circuit_time = estimator.parallel_circuit_time(primitive_circuit)
    resources = {'op_time': circuit_time, 'moment_cost': Counter(parallel_gate_cost), 'gate_cost': Counter(serial_gate_cost)}
    return resources

