import cirq
import pytest
from math import pi
import numpy as np
from resource_estimation.clifford_t import (
    compile_cirq_to_clifford_t,
    process_cirq_str,
    approx_rz,
    toffoli_decompose,
)


@pytest.mark.parametrize("theta", (1, pi / 3, pi + 1, 2 * pi - pi / 7, pi / 4))
@pytest.mark.parametrize("eps", (1e-1, 1e-3, 1e-5, 1e-7))
def test_compile_cirq_to_clifford_t(theta, eps):
    circuit = cirq.Circuit(cirq.Rz(rads=theta).on(cirq.GridQubit(0, 0)))
    comp_circuit = compile_cirq_to_clifford_t(circuit, eps=eps, verbose=False)
    u_expected = cirq.unitary(circuit)
    u_realized = cirq.unitary(comp_circuit)
    fid = abs(np.trace(u_realized.conjugate().T @ u_expected) / 2)
    err = 1 - fid
    assert err < eps


@pytest.mark.parametrize(
    "theta",
    (
        pi,
        -pi,
        pi / 2,
        -3 * pi / 2,
        pi / 4,
        -7 * pi / 4,
        3 * pi / 2,
        -pi / 2,
        2 * pi / 4,
        -5 * pi / 4,
        5 * pi / 4,
        -3 * pi / 4,
        7 * pi / 4,
        -pi / 4,
        0,
        2 * pi,
    ),
)
def test_special_cases(theta):
    eps = 0.0001
    circuit = cirq.Circuit(cirq.Rz(rads=theta).on(cirq.GridQubit(0, 0)))
    comp_circuit = compile_cirq_to_clifford_t(circuit, eps=eps, verbose=False)
    if theta in [0, 2 * pi]:
        assert not comp_circuit
    else:
        u_expected = cirq.unitary(circuit)
        u_realized = cirq.unitary(comp_circuit)
        fid = abs(np.trace(u_realized.conjugate().T @ u_expected) / 2)
        err = 1 - fid
        assert err < eps


def test_error_handling():
    bad_circuit = cirq.Circuit(cirq.Rx(rads=1).on(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError):
        _ = compile_cirq_to_clifford_t(bad_circuit, eps=0.01)

    with pytest.raises(ValueError):
        _ = process_cirq_str(bad_circuit, gates=["P"], q=cirq.GridQubit(0, 0))


def test_measure():
    circuit = cirq.Circuit(cirq.MeasurementGate(1).on(cirq.GridQubit(0, 0)))
    comp_circuit = compile_cirq_to_clifford_t(circuit, eps=0.001)
    cirq.testing.assert_same_circuits(actual=comp_circuit, expected=circuit)


def test_in_cliffs():
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(
        cirq.X.on(q),
        cirq.H.on(q),
        cirq.S.on(q),
        cirq.I.on(q),
    )
    comp_circuit = compile_cirq_to_clifford_t(circuit, eps=0.001)
    cirq.testing.assert_same_circuits(actual=comp_circuit, expected=circuit)


def test_t_synth():
    theta, eps = 1.2345678, 1e-8
    true_circuit = cirq.Circuit(cirq.Rz(rads=theta / pi).on(cirq.GridQubit(0, 0)))
    decomp = approx_rz(theta, eps)
    approx_circuit = cirq.Circuit()
    process_cirq_str(circ=approx_circuit, gates=decomp, q=cirq.GridQubit(0, 0))
    rho_true = cirq.density_matrix(cirq.final_density_matrix(true_circuit))
    rho_approx = cirq.density_matrix(cirq.final_density_matrix(approx_circuit))
    assert cirq.fidelity(rho_true, rho_approx) >= 1 - eps


def test_special_angles():
    eps = 1e-5
    for theta, expected_str in [
        (pi / 4, "T"),
        (2 * pi / 4, "S"),
        (3 * pi / 4, "ST"),
        (4 * pi / 4, "Z"),
        (5 * pi / 4, "ZT"),
        (6 * pi / 4, "ZS"),
        (7 * pi / 4, "ZST"),
        (-pi / 4, "ZST"),
        (-2 * pi / 4, "ZS"),
        (-3 * pi / 4, "ZT"),
        (-4 * pi / 4, "Z"),
        (-5 * pi / 4, "ST"),
        (-6 * pi / 4, "S"),
        (-7 * pi / 4, "T"),
    ]:
        assert approx_rz(theta, eps) == expected_str


def test_misc():
    illegal_circuit = cirq.Circuit(cirq.Rx(rads=2).on(cirq.LineQubit(0)))
    with pytest.raises(ValueError):
        _ = compile_cirq_to_clifford_t(circ=illegal_circuit, eps=1e-4)
    assert process_cirq_str(cirq.Circuit(), "I", cirq.LineQubit(0)) is None
    with pytest.raises(ValueError):
        _ = process_cirq_str(cirq.Circuit(), "M", cirq.LineQubit(0))
    M_circuit = cirq.Circuit(cirq.MeasurementGate(1, key="").on(cirq.LineQubit(0)))
    synthesized_M = compile_cirq_to_clifford_t(circ=M_circuit, eps=1e-2)
    assert synthesized_M == M_circuit


def test_toffoli_decompose():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.I.on(qubits[0]), cirq.TOFFOLI.on(*qubits))
    new_circuit = toffoli_decompose(circuit=circuit)
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        actual=new_circuit, expected=circuit, qubit_map={qubit: qubit for qubit in qubits}
    )
