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
"""Independent dense checks of full-channel equality and stabilizer partial traces."""

from __future__ import annotations

import random
from collections.abc import Sequence

import cirq
import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from resource_estimation.clifford_verification import cliffords_equal
from resource_estimation.clifford_verification.benchmarks.circuits import make_pair
from resource_estimation.clifford_verification.stabilizers import Signature, reduced_signature
from resource_estimation.clifford_verification.tableau_channels import channel_signature
from resource_estimation.compile_gateset.ppm_compiler import ResourceStateTag


def tagged_t(qubit: cirq.Qid, *, dagger: bool = False) -> cirq.Operation:
    gate = cirq.T ** (-1 if dagger else 1)
    return cirq.I(qubit).with_tags(ResourceStateTag(gate, 0, 0, 1))


def density(signature: Signature) -> np.ndarray:
    n, generators = signature
    result: NDArray[np.complex128] = np.eye(1 << n, dtype=np.complex128) / (1 << n)
    for x, z, phase in generators:
        pauli = np.array([[1]], dtype=complex)
        for j in range(n):
            factor = np.eye(2, dtype=complex)
            if x >> j & 1:
                factor = factor @ cirq.unitary(cirq.X)
            if z >> j & 1:
                factor = factor @ cirq.unitary(cirq.Z)
            pauli = np.kron(pauli, factor)
        result = result @ (np.eye(1 << n) + 1j**phase * pauli)
    return result


@pytest.mark.parametrize("n", range(1, 6))
def test_partial_trace_against_dense(n: int) -> None:
    rng = random.Random(98 + n)
    qubits = cirq.LineQubit.range(n)
    for _ in range(8):
        circuit = cirq.Circuit(
            rng.choice([cirq.H, cirq.S, cirq.X, cirq.Y])(rng.choice(qubits)) for _ in range(8)
        )
        if n > 1:
            circuit.append(cirq.CNOT(*rng.sample(qubits, 2)) for _ in range(5))
        tableau = cirq.CliffordTableau(n)
        state = cirq.CliffordTableauSimulationState(tableau, qubits=qubits)
        for op in circuit.all_operations():
            cirq.act_on(op, state)
        vector = cirq.final_state_vector(circuit, qubit_order=qubits, dtype=np.complex128)
        full = np.outer(vector, vector.conj()).reshape((2,) * (2 * n))
        keep = rng.sample(range(n), rng.randrange(n + 1))
        expected = cirq.partial_trace(full, keep).reshape(1 << len(keep), 1 << len(keep))
        np.testing.assert_allclose(density(reduced_signature(tableau, keep)), expected, atol=1e-12)


def dense_channel(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid]) -> np.ndarray:
    """Small independent Kraus enumeration oracle; never used by the backend."""
    d = 1 << len(qubits)
    branches: list[tuple[NDArray[np.complex128], dict[str, list[int]]]] = [
        (np.eye(d, dtype=np.complex128), {})
    ]
    for op in circuit.all_operations():
        gate = op.gate
        if isinstance(gate, cirq.PauliMeasurementGate):
            p = cirq.Circuit(gate.observable().on(*op.qubits)).unitary(qubit_order=qubits)
            branches = [
                (
                    ((np.eye(d) + (-1) ** m * p) / 2) @ k,
                    {**records, gate.key: [*records.get(gate.key, []), m]},
                )
                for k, records in branches
                for m in (0, 1)
            ]
        else:
            updated = []
            for k, records in branches:
                enabled = True
                for cond in op.classical_controls:
                    if isinstance(cond, cirq.KeyCondition):
                        enabled &= bool(records[str(cond.key)][cond.index])
                    else:
                        assert isinstance(cond, cirq.SympyCondition)
                        enabled &= bool(
                            cond.expr.subs({key: bool(value[-1]) for key, value in records.items()})
                        )
                unitary = cirq.Circuit(op.without_classical_controls()).unitary(qubit_order=qubits)
                updated.append((unitary @ k if enabled else k, records))
            branches = updated
    result = np.zeros((d * d, d * d), dtype=complex)
    for k, _ in branches:
        vector = k.T.reshape(-1)
        result += np.outer(vector, vector.conj()) / d
    return result


@pytest.mark.parametrize("seed", range(12))
def test_adaptive_choi_against_dense(seed: int) -> None:
    rng = random.Random(seed)
    a, b = qubits = list(cirq.LineQubit.range(2))
    circuit = cirq.Circuit()
    for layer in range(2):
        circuit.append(rng.choice([cirq.H, cirq.S, cirq.Y])(rng.choice(qubits)))
        pauli = (
            rng.choice([1, -1])
            * rng.choice([cirq.X, cirq.Y, cirq.Z])(a)
            * rng.choice([cirq.X, cirq.Y, cirq.Z])(b)
        )
        circuit.append(cirq.measure_single_paulistring(pauli, key=f"m{layer}"))
        cond = (
            sympy.Symbol("m0") if layer == 0 else sympy.Xor(sympy.Symbol("m0"), sympy.Symbol("m1"))
        )
        if seed % 2:
            cond = sympy.Not(cond)
        circuit.append(rng.choice([cirq.X, cirq.Y, cirq.Z])(b).with_classical_controls(cond))
        circuit.append(cirq.CNOT(a, b))
    signature = channel_signature(circuit, qubits, qubits)
    np.testing.assert_allclose(density(signature), dense_channel(circuit, qubits), atol=1e-12)
    altered = circuit + cirq.Circuit(cirq.X(a))
    assert cliffords_equal(circuit, altered) == np.allclose(
        dense_channel(circuit, qubits), dense_channel(altered, qubits), atol=1e-12
    )
    assert cliffords_equal(circuit, circuit)


@pytest.mark.parametrize("family", ["ghz", "random", "ppm"])
def test_benchmark_pairs(family: str) -> None:
    for n in (2, 4, 8):
        left, right = make_pair(family, n, 4)
        assert cliffords_equal(left, right)


def test_gadget_and_resources() -> None:
    a, ao, b, bo, r = map(cirq.NamedQubit, ["a_in", "a_out", "b_in", "b_out", "resource"])
    original = cirq.Circuit(
        tagged_t(r), cirq.CNOT(a, b), cirq.CNOT(b, r), cirq.CNOT(a, b), cirq.measure(r, key="s")
    )
    body = cirq.Circuit(
        tagged_t(r),
        cirq.reset(ao),
        cirq.reset(bo),
        cirq.measure_single_paulistring(cirq.X(b) * cirq.X(bo) * cirq.X(r), key="q"),
        cirq.measure_single_paulistring(cirq.X(a) * cirq.X(ao) * cirq.X(b) * cirq.X(bo), key="p"),
        cirq.measure(a, key="m"),
        cirq.measure(b, key="n"),
        cirq.measure(r, key="l"),
        cirq.Z(ao).with_classical_controls(sympy.Xor(sympy.Symbol("p"), sympy.Symbol("q"))),
        cirq.Z(bo).with_classical_controls("q"),
        cirq.X(ao).with_classical_controls("m"),
        cirq.X(bo).with_classical_controls("n"),
    )
    assert cliffords_equal(original, body)
    flag = cirq.NamedQubit("z_result")
    original = (
        cirq.Circuit(cirq.reset(flag))
        + original
        + cirq.Circuit(cirq.X(flag).with_classical_controls("s"))
    )
    body = (
        cirq.Circuit(cirq.reset(flag))
        + body
        + cirq.Circuit(
            cirq.X(flag).with_classical_controls(sympy.Xor(*map(sympy.Symbol, ["m", "n", "l"])))
        )
    )
    assert cliffords_equal(original, body)


def test_channel_not_sampled_state() -> None:
    a = cirq.LineQubit(0)
    assert not cliffords_equal(cirq.Circuit(cirq.I(a)), cirq.Circuit(cirq.Z(a)))
    assert cliffords_equal(
        cirq.Circuit(cirq.I(a)), cirq.Circuit(cirq.I(a), cirq.global_phase_operation(-1))
    )
    m = cirq.measure_single_paulistring(cirq.Z(a), key="m")
    assert not cliffords_equal(cirq.Circuit(m), cirq.Circuit(cirq.I(a)))
    assert cliffords_equal(cirq.Circuit(m), cirq.Circuit(m, m))
    assert cliffords_equal(cirq.Circuit(m), cirq.Circuit(m, cirq.Z(a).with_classical_controls("m")))


def test_preparation_and_discard_boundaries() -> None:
    a, b = cirq.LineQubit.range(2)
    assert cliffords_equal(cirq.Circuit(), cirq.Circuit(cirq.I(a)))
    assert cliffords_equal(
        cirq.Circuit(cirq.reset(a), cirq.H(a)), cirq.Circuit(cirq.reset(b), cirq.H(b))
    )
    assert cliffords_equal(
        cirq.Circuit(cirq.measure(a)), cirq.Circuit(cirq.measure(a, invert_mask=(True,)))
    )


def test_indexed_and_signed_corrections() -> None:
    a, b = cirq.LineQubit.range(2)
    prefix = cirq.Circuit(
        cirq.measure_single_paulistring(cirq.X(a), key="m"),
        cirq.measure_single_paulistring(cirq.Z(a), key="m"),
    )
    for index in (0, 1, -1, -2):
        for shift in (-0.5, 0, 0.5):
            op = cirq.YPowGate(global_shift=shift)(b).with_classical_controls(
                cirq.KeyCondition(cirq.MeasurementKey("m"), index)
            )
            circuit = prefix + cirq.Circuit(op)
            changed = circuit + cirq.Circuit(cirq.Z(b))
            np.testing.assert_allclose(
                density(channel_signature(circuit, [a, b], [a, b])),
                dense_channel(circuit, [a, b]),
                atol=1e-12,
            )
            assert cliffords_equal(circuit, changed) == np.allclose(
                dense_channel(circuit, [a, b]), dense_channel(changed, [a, b]), atol=1e-12
            )
    with pytest.raises(ValueError, match="Unavailable"):
        cliffords_equal(
            cirq.Circuit(cirq.X(a).with_classical_controls("missing")), cirq.Circuit(cirq.I(a))
        )
    with pytest.raises(ValueError, match="Pauli"):
        cliffords_equal(prefix + cirq.Circuit(cirq.H(b).with_classical_controls("m")), prefix)


def test_cirq_clifford_gates_against_dense() -> None:
    """Check several native Cirq Clifford gate families against dense evolution."""
    a, b = qubits = cirq.LineQubit.range(2)
    for operation in (
        cirq.ISWAP(a, b),
        cirq.XX(a, b) ** 0.5,
        cirq.SingleQubitCliffordGate.H(a),
        cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5)(a),
    ):
        circuit = cirq.Circuit(cirq.H(a), operation, cirq.S(b))
        np.testing.assert_allclose(
            density(channel_signature(circuit, qubits, qubits)),
            dense_channel(circuit, qubits),
            atol=1e-12,
        )
        assert cliffords_equal(circuit, circuit)
        assert not cliffords_equal(circuit, circuit + cirq.Circuit(cirq.X(a)))


def test_public_boundary_validation() -> None:
    a = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="counts"):
        cliffords_equal(cirq.Circuit(cirq.I(a)), cirq.Circuit(cirq.measure(a)))
    assert cliffords_equal(cirq.Circuit(), cirq.Circuit())
    assert cliffords_equal(cirq.Circuit(cirq.reset(a), cirq.H(a), cirq.measure(a)), cirq.Circuit())


@pytest.mark.parametrize(
    "keyword", ["left_inputs", "left_outputs", "right_inputs", "right_outputs"]
)
def test_boundary_overrides_are_rejected(keyword: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        cliffords_equal(cirq.Circuit(), cirq.Circuit(), **{keyword: []})


def test_non_cliffords_are_rejected_even_when_identical() -> None:
    from resource_estimation.clifford_verification.resources import NonCliffordError

    a, b, c = cirq.LineQubit.range(3)
    for operation in (cirq.T(a), cirq.CCZ(a, b, c), cirq.Z(a) ** 0.5000000001):
        circuit = cirq.Circuit(operation)
        with pytest.raises(NonCliffordError, match="called with non-Cliffords"):
            cliffords_equal(circuit, circuit)


def test_signed_pauli_measurements_against_dense() -> None:
    from itertools import product

    for n in range(1, 4):
        qubits = cirq.LineQubit.range(n)
        for word in product("XYZ", repeat=n):
            for sign in (-1, 1):
                gate = cirq.PauliMeasurementGate(
                    cirq.DensePauliString("".join(word), coefficient=sign), key="m"
                )
                circuit = cirq.Circuit(gate.on(*qubits))
                np.testing.assert_allclose(
                    density(channel_signature(circuit, qubits, qubits)),
                    dense_channel(circuit, qubits),
                    atol=1e-12,
                )


@pytest.mark.parametrize("seed", range(12))
def test_conditional_s_mixtures_against_dense(seed: int) -> None:
    from resource_estimation.clifford_verification.stabilizers import channel_mixtures_equal
    from resource_estimation.clifford_verification.tableau_channels import channel_mixture

    a, b = qubits = cirq.LineQubit.range(2)
    rng = random.Random(seed)
    circuit = cirq.Circuit()
    for j in range(2):
        circuit.append(
            cirq.measure_single_paulistring(
                rng.choice([1, -1]) * rng.choice([cirq.X, cirq.Y, cirq.Z])(a), key=f"s{j}"
            )
        )
        condition = (
            sympy.Symbol("s0") if j == 0 else sympy.Xor(sympy.Symbol("s0"), sympy.Symbol("s1"))
        )
        circuit.append((cirq.S(b) ** rng.choice([1, -1])).with_classical_controls(condition))
        circuit.append(cirq.CNOT(b, a))
        circuit.append(cirq.H(a))
    mixture = channel_mixture(circuit, qubits, qubits)
    actual = sum(float(w) * density(s) for w, s in mixture)
    expected = dense_channel(circuit, qubits)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert sum(w for w, _ in mixture) == 1
    changed = circuit + cirq.Circuit(cirq.X(b))
    assert channel_mixtures_equal(mixture, channel_mixture(changed, qubits, qubits)) == np.allclose(
        expected, dense_channel(changed, qubits)
    )
    assert cliffords_equal(circuit, circuit)


def test_s_branch_decompositions_and_correlated_controls() -> None:
    a, b = cirq.LineQubit.range(2)
    measurement = cirq.measure_single_paulistring(cirq.Z(a), key="m")
    # On its measured eigenstate, S changes only an unobservable branch phase.
    plain = cirq.Circuit(measurement)
    assert cliffords_equal(plain, plain + cirq.Circuit(cirq.S(a).with_classical_controls("m")))
    # Two S corrections with the same bit are Z; only two cases are reachable.
    prefix = cirq.Circuit(measurement, cirq.I(b))
    assert cliffords_equal(
        prefix
        + cirq.Circuit(
            cirq.S(b).with_classical_controls("m"), cirq.S(b).with_classical_controls("m")
        ),
        prefix + cirq.Circuit(cirq.Z(b).with_classical_controls("m")),
    )
    assert cliffords_equal(
        prefix
        + cirq.Circuit(
            cirq.S(b).with_classical_controls("m"), (cirq.S(b) ** -1).with_classical_controls("m")
        ),
        prefix,
    )
    assert not cliffords_equal(
        prefix + cirq.Circuit(cirq.S(b).with_classical_controls("m")), prefix
    )


def test_stabilizer_overlap_against_dense() -> None:
    from resource_estimation.clifford_verification.stabilizers import stabilizer_overlap

    rng = random.Random(918)
    qubits = cirq.LineQubit.range(3)
    signatures = []
    for _ in range(15):
        tableau = cirq.CliffordTableau(3)
        state = cirq.CliffordTableauSimulationState(tableau, qubits=qubits)
        for _ in range(8):
            cirq.act_on(rng.choice([cirq.H, cirq.S, cirq.X])(rng.choice(qubits)), state)
            cirq.act_on(cirq.CNOT(*rng.sample(qubits, 2)), state)
        signatures.append(reduced_signature(tableau, [0, 2]))
    for a in signatures:
        for b in signatures:
            np.testing.assert_allclose(
                float(stabilizer_overlap(a, b)), np.trace(density(a) @ density(b)), atol=1e-12
            )


def test_t_injection_conditional_s() -> None:
    a, r = cirq.LineQubit.range(2)
    prefix = cirq.Circuit(tagged_t(r), cirq.CNOT(a, r), cirq.measure(r, key="m"))
    original = prefix + cirq.Circuit(cirq.S(a).with_classical_controls("m"))
    rewritten = prefix + cirq.Circuit(
        (cirq.S(a) ** -1).with_classical_controls("m"),
        cirq.Z(a).with_classical_controls("m"),
    )
    assert cliffords_equal(original, rewritten)
    from resource_estimation.clifford_verification.resources import NotCertifiedError

    with pytest.raises(NotCertifiedError, match="open-resource"):
        cliffords_equal(original, prefix + cirq.Circuit(cirq.I(a)))


@pytest.mark.parametrize("outcome", [0, 1])
def test_deterministic_s_condition(outcome: int) -> None:
    from resource_estimation.clifford_verification.tableau_channels import channel_mixture

    a, b = cirq.LineQubit.range(2)
    prefix = cirq.Circuit(cirq.reset(a), cirq.X(a) ** outcome, cirq.measure(a, key="m"))
    circuit = prefix + cirq.Circuit(cirq.S(b).with_classical_controls("m"))
    fixed = prefix + cirq.Circuit((cirq.S(b) if outcome else cirq.I(b)))
    assert cliffords_equal(circuit, fixed)
    mixture = channel_mixture(circuit, [b], [b])
    assert len(mixture) == 1
    assert mixture[0][0] == 1
