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
import resource_estimation.cliff_rz as cliff
from scripts.circuits import kanamori, fermi_hubbard


def test_fermi():
    # Test that Fermi-Hubbard circuit is compiled to Clifford + Rz correctly
    # For some reason, I can't do better that 1e-6
    ham_circuit = fermi_hubbard(3, verbose=0)
    compiled_circuit = cliff.compile_cliff_rz(circuit=ham_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.final_state_vector(ham_circuit),
        cirq.final_state_vector(compiled_circuit),
        atol=1e-6,
    )

    # assert only legal gates in compiled circuit
    allowed_ops = [cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.MeasurementGate, cirq.T, cirq.Rz]
    allowed_ops = [cirq.GateFamily(op) for op in allowed_ops]
    for op in compiled_circuit.all_operations():
        gate = op.gate
        truth = any([gate in allowed_op for allowed_op in allowed_ops])
        assert truth, f"{gate}"


def test_kanamori():
    # Test that Kanamori circuit is compiled to Clifford + Rz correctly
    # For some reason, I can't do better that 1e-6
    kan_circuit = kanamori(5, verbose=0)
    compiled_circuit = cliff.compile_cliff_rz(circuit=kan_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.final_state_vector(kan_circuit),
        cirq.final_state_vector(compiled_circuit),
        atol=1e-6,
    )

    # assert only legal gates in compiled circuit
    allowed_ops = [cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.MeasurementGate, cirq.T, cirq.Rz]
    allowed_ops = [cirq.GateFamily(op) for op in allowed_ops]
    for op in compiled_circuit.all_operations():
        gate = op.gate
        truth = any([gate in allowed_op for allowed_op in allowed_ops])
        assert truth, f"{gate}"


def test_already_in_gateset():
    op = cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    gateset = cliff.CliffRzGateset(cirq.LineQubit.range(2))
    assert cirq.CNOT in gateset
    print(gateset._decompose_two_qubit_operation(op))
    same_op = cirq.Circuit(gateset._decompose_two_qubit_operation(op))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        actual=same_op, reference=cirq.Circuit(op)
    )


def test_phx_to_zhzhz():
    q = cirq.GridQubit(0, 0)
    I_circuit = cirq.Circuit(cirq.PhasedXPowGate(exponent=0, phase_exponent=0.5).on(q))
    transformed = cliff.phx_to_zhzhz(circuit=I_circuit)
    assert not transformed

    SX_circuit = cirq.Circuit(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25).on(q))
    transformed = cliff.phx_to_zhzhz(circuit=SX_circuit)
    expected = str(cirq.Circuit([cirq.Z(q) ** (-0.75), cirq.H(q), cirq.Z(q) ** (-0.25)]))
    cirq.testing.assert_has_diagram(
        actual=transformed,
        desired=expected,
    )

    SX_dg_circuit = cirq.Circuit(cirq.PhasedXPowGate(exponent=-0.5, phase_exponent=0.25).on(q))
    transformed = cliff.phx_to_zhzhz(circuit=SX_dg_circuit)
    expected = str(cirq.Circuit([cirq.Z(q) ** (0.25), cirq.H(q), cirq.Z(q) ** (0.75)]))
    cirq.testing.assert_has_diagram(
        actual=transformed,
        desired=expected,
    )

    X_circuit = cirq.Circuit(cirq.PhasedXPowGate(exponent=1.0, phase_exponent=0.25).on(q))
    transformed = cliff.phx_to_zhzhz(circuit=X_circuit)
    expected = str(cirq.Circuit([cirq.Z(q) ** (-0.25), cirq.X(q), cirq.Z(q) ** (0.25)]))
    cirq.testing.assert_has_diagram(
        actual=transformed,
        desired=expected,
    )


def test_small_circuit():
    random_circuit = cirq.testing.random_circuit(8, 10, 1, random_state=17)
    compiled_circuit = cliff.compile_cliff_rz(random_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.final_state_vector(random_circuit),
        cirq.final_state_vector(compiled_circuit),
        atol=1e-6,  # For sanity
    )
