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

from resource_estimation.ftqc.architecture import DefaultMovement
from resource_estimation.ftqc.distil import distil_15_to_1, precompute_distil_cost
from resource_estimation.ftqc.lattice_surgery_primitives import Cultivate
from resource_estimation.ftqc.layout import MovementDistillery


@pytest.fixture
def base_15_to_one() -> cirq.Circuit:
    qubits = cirq.LineQubit.range(15) + [cirq.NamedQubit("F")]

    # cir = cirq.Circuit()
    circuit = cirq.Circuit(
        [
            cirq.ResetChannel().on_each(*qubits),
            # css.Barrier(16).on(*qubits),
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[3]),
            cirq.H(qubits[7]),
            cirq.H(qubits[15]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[15], qubits[14]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[7], qubits[8]),
            cirq.CNOT(qubits[7], qubits[9]),
            cirq.CNOT(qubits[7], qubits[10]),
            cirq.CNOT(qubits[7], qubits[11]),
            cirq.CNOT(qubits[7], qubits[12]),
            cirq.CNOT(qubits[7], qubits[13]),
            cirq.CNOT(qubits[7], qubits[14]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[3], qubits[4]),
            cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6]),
            cirq.CNOT(qubits[3], qubits[11]),
            cirq.CNOT(qubits[3], qubits[12]),
            cirq.CNOT(qubits[3], qubits[13]),
            cirq.CNOT(qubits[3], qubits[14]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[6]),
            cirq.CNOT(qubits[1], qubits[9]),
            cirq.CNOT(qubits[1], qubits[10]),
            cirq.CNOT(qubits[1], qubits[13]),
            cirq.CNOT(qubits[1], qubits[14]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.CNOT(qubits[0], qubits[4]),
            cirq.CNOT(qubits[0], qubits[6]),
            cirq.CNOT(qubits[0], qubits[8]),
            cirq.CNOT(qubits[0], qubits[10]),
            cirq.CNOT(qubits[0], qubits[12]),
            cirq.CNOT(qubits[0], qubits[14]),
            # css.Barrier(16).on(*qubits),
            cirq.CNOT(qubits[14], qubits[2]),
            cirq.CNOT(qubits[14], qubits[4]),
            cirq.CNOT(qubits[14], qubits[5]),
            cirq.CNOT(qubits[14], qubits[8]),
            cirq.CNOT(qubits[14], qubits[9]),
            cirq.CNOT(qubits[14], qubits[11]),
            # css.Barrier(16).on(*qubits),
            cirq.T.on_each(*qubits[:-1]),
            # css.Barrier(16).on(*qubits),
            cirq.H.on_each(*qubits[:-1]),
        ],
    )
    circuit.append(cirq.Moment(cirq.measure_each(*qubits[:-1])))
    return circuit[1:-1]  # Just the non-classical part


def test_15_to_one(base_15_to_one) -> None:
    """Check to confirm that the compression technique agrees with the standard circuit"""
    circuit = distil_15_to_1()

    # There should be exactly 15 Cultivations
    assert sum(op.gate == Cultivate(pi / 4) for op in circuit.all_operations()) == 15

    # There should be 30 Measurements
    assert sum(type(op.gate) == cirq.MeasurementGate for op in circuit.all_operations()) == 30

    # There should be 7*5 + 15 = 50 CNOT gates
    assert sum(op.gate in cirq.GateFamily(cirq.CNOT) for op in circuit.all_operations()) == 50


def test_precompute_distil_cost():
    empty_circuit = cirq.Circuit(cirq.I.on_each(*cirq.LineQubit.range(4)))
    layout = MovementDistillery(
        input_circuit=empty_circuit,
        num_t_factories=1,
        num_ccz_factories=1,
        architecture="SSM",
    )
    arc = DefaultMovement()
    with pytest.raises(ValueError, match="Unknown distillation resource"):
        _ = precompute_distil_cost("Toffoli", layout=layout, arc=arc)

    # Distil T and CCZ have the same critical path, but T cultivation moves further, so it should be more expensive
    # Cultivation is a subcomponent, so it should be faster than the Distillation implementations
    cult = arc._cultivate_t_cost["op_time"]
    t_distil = precompute_distil_cost("T", layout=layout, arc=arc)["op_time"]
    ccz_distil = precompute_distil_cost("CCZ", layout=layout, arc=arc)["op_time"]
    assert cult < ccz_distil < t_distil
