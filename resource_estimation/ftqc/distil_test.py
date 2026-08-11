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

from resource_estimation.ftqc.distil import distil_15_to_1
from resource_estimation.ftqc.lattice_surgery_primitives import Cultivate

# Need to add tests here.

def test_15_to_one() -> None:
    """Check to confirm that the compression technique agrees with the standard circuit"""
    circuit = distil_15_to_1()

    # There should be exactly 15 Cultivations
    assert sum(op in cirq.GateFamily(Cultivate(pi / 4)) for op in circuit.all_operations()) == 15

    # There should be 30 Measurements
    assert sum(op in cirq.GateFamily(cirq.MeasurementGate) for op in circuit.all_operations()) == 30

    # There should be 7*5 + 15 = 50 CNOT gates
    assert sum(op in cirq.GateFamily(cirq.CNOT) for op in circuit.all_operations()) == 50
