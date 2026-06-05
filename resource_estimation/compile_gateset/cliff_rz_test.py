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
import resource_estimation.compile_gateset as cliff
from scripts.circuits import kanamori, fermi_hubbard
import pytest


def _compile_cliff_rz(circuit: cirq.Circuit) -> cirq.Circuit:
    return cliff.compile_gateset(circuit, gateset=cliff.clifford_rz_gateset(atol=1e-15))


def _compile_cliff_phxz(circuit: cirq.Circuit) -> cirq.Circuit:
    return cliff.compile_gateset(circuit, gateset=cliff.clifford_phxz_gateset(atol=1e-15))


def _compile_cliff_t_direct(circuit: cirq.Circuit) -> cirq.Circuit:
    return cliff.compile_gateset(
        circuit, gateset=cliff.clifford_t_direct_gateset(eps=1e-9, atol=1e-15)
    )


@pytest.mark.parametrize(
    "func, gateset",
    [
        (_compile_cliff_rz, cliff.CliffRzGateset()),
        (_compile_cliff_phxz, cliff.CliffPhXZGateset()),
        # (_compile_cliff_t_direct, cliff.CliffTDirect(epsilon=1e-8)),
    ],
)
def test_fermi(func, gateset) -> None:
    # Test that Fermi-Hubbard circuit is compiled to Clifford + Rz correctly
    # For some reason, I can't do better that 1e-6
    ham_circuit = fermi_hubbard(3, verbose=0)
    compiled_circuit = func(circuit=ham_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.final_state_vector(ham_circuit),
        cirq.final_state_vector(compiled_circuit),
        atol=1e-6,
    )
    for op in compiled_circuit.all_operations():
        assert op in gateset


@pytest.mark.parametrize(
    "func, gateset",
    [
        (_compile_cliff_rz, cliff.CliffRzGateset()),
        (_compile_cliff_phxz, cliff.CliffPhXZGateset()),
        # (_compile_cliff_t_direct, cliff.CliffTDirect(epsilon=1e-8)),  # This runs far too slow to be a useful test
    ],
)
def test_kanamori(func, gateset) -> None:
    # Test that Kanamori circuit is compiled to Clifford + Rz correctly
    # For some reason, I can't do better that 1e-6
    kan_circuit = kanamori(5, verbose=0)
    compiled_circuit = func(circuit=kan_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.final_state_vector(kan_circuit),
        cirq.final_state_vector(compiled_circuit),
        atol=1e-6,
    )
    for op in compiled_circuit.all_operations():
        assert op in gateset


def test_already_in_gateset() -> None:
    op = cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    gateset = cliff.CliffRzGateset(cirq.LineQubit.range(2))
    assert cirq.CNOT in gateset
    print(gateset._decompose_two_qubit_operation(op))
    same_op = cirq.Circuit(gateset._decompose_two_qubit_operation(op))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        actual=same_op, reference=cirq.Circuit(op)
    )


def test_phx_to_zhzhz() -> None:
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


@pytest.mark.parametrize(
    "func",
    (
        _compile_cliff_rz,
        _compile_cliff_phxz,
        # _compile_cliff_t_direct,
    ),
)
def test_small_circuit(func) -> None:
    random_circuit = cirq.testing.random_circuit(8, 10, 1, random_state=17)
    compiled_circuit = func(random_circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(random_circuit),
        cirq.unitary(compiled_circuit),
        atol=1e-6,  # For sanity
    )


@pytest.mark.parametrize("qubits", (1, 2))
@pytest.mark.parametrize(
    "compiler", (_compile_cliff_rz, _compile_cliff_phxz, _compile_cliff_t_direct)
)
def test_random_circuits(qubits, compiler):
    U = cirq.testing.random_unitary(dim=2**qubits, random_state=7)
    circuit = cirq.Circuit(cirq.MatrixGate(U).on(*cirq.LineQubit.range(qubits)))
    compiled = compiler(circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(compiled),
        cirq.unitary(circuit),
        atol=1e-6,
    )


def test_op_not_replaced():
    q1, q2 = cirq.LineQubit.range(2)
    op1 = cirq.S.on(q1)
    decomposed_op = cliff.CliffTDirect(epsilon=1e-3)._decompose_single_qubit_operation(op=op1)
    assert op1 is decomposed_op
    op2 = cirq.CNOT.on(q1, q2)
    decomposed_op = cliff.CliffTDirect(epsilon=1e-3)._decompose_two_qubit_operation(op=op2)
    assert op2 is decomposed_op


def test_replace_op_with_pygridsynth():
    with pytest.raises(ValueError, match="Support for multi-qubit gates"):
        _ = cliff.replace_op_with_pygridsynth(
            cirq.MatrixGate(cirq.testing.random_unitary(dim=8, random_state=7)).on(
                *cirq.LineQubit.range(3)
            ),
            1e-3,
        )
    U = cirq.testing.random_unitary(dim=4, random_state=7)
    circuit = cirq.Circuit(cirq.MatrixGate(U).on(*cirq.LineQubit.range(2)))
    direct_replacement = cliff.replace_op_with_pygridsynth(next(circuit.all_operations()), 1e-7)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(direct_replacement),
        cirq.unitary(circuit),
        atol=1e-6,
    )
