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
from __future__ import annotations

import cirq
import numpy as np
import pytest

from resource_estimation.clifford_verification import cliffords_equal
from resource_estimation.clifford_verification.resources import NonCliffordError, NotCertifiedError
from resource_estimation.compile_gateset.ppm_compiler import ResourceStateTag


def tagged_t(qubit: cirq.Qid, *, dagger: bool = False) -> cirq.Operation:
    gate = cirq.T ** (-1 if dagger else 1)
    return cirq.I(qubit).with_tags(ResourceStateTag(gate, 0, 0, 1))


def test_tagged_resources() -> None:
    a, b = cirq.LineQubit.range(2)
    for dagger in (False, True):
        assert cliffords_equal(
            cirq.Circuit(tagged_t(a, dagger=dagger)), cirq.Circuit(tagged_t(b, dagger=dagger))
        )


def test_numerical_preparations_are_rejected() -> None:
    a = cirq.LineQubit(0)
    preparation = cirq.StatePreparationChannel(np.array([1, np.exp(1j * np.pi / 4)]))
    with pytest.raises(ValueError, match="ResourceStateTag"):
        cliffords_equal(cirq.Circuit(preparation(a)), cirq.Circuit(tagged_t(a)))


def test_non_cliffords_are_not_preparations() -> None:
    a, b, c = cirq.LineQubit.range(3)
    for operation in (
        cirq.T(a),
        cirq.CCZ(a, b, c),
        cirq.rx(0.123)(a),
        cirq.T(a).with_classical_controls("m"),
    ):
        with pytest.raises(NonCliffordError, match="called with non-Cliffords"):
            cliffords_equal(cirq.Circuit(operation), cirq.Circuit())
    with pytest.raises(NonCliffordError, match="non-Cliffords"):
        cliffords_equal(
            cirq.Circuit(cirq.reset(a), cirq.H(a), cirq.T(a)), cirq.Circuit(tagged_t(a))
        )


def test_not_certified_and_preparation_validation() -> None:
    a = cirq.LineQubit(0)
    with pytest.raises(NotCertifiedError, match="preparations"):
        cliffords_equal(cirq.Circuit(tagged_t(a)), cirq.Circuit(tagged_t(a, dagger=True)))
    # S maps T-dagger to T, but the conservative preparation matcher does not certify it.
    with pytest.raises(NotCertifiedError):
        cliffords_equal(
            cirq.Circuit(tagged_t(a)), cirq.Circuit(tagged_t(a, dagger=True), cirq.S(a))
        )
    with pytest.raises(NotCertifiedError, match="open-resource"):
        cliffords_equal(cirq.Circuit(tagged_t(a)), cirq.Circuit(tagged_t(a), cirq.Z(a)))
    for prefix in (cirq.X(a), cirq.reset(a), tagged_t(a)):
        with pytest.raises(ValueError, match="first use"):
            cliffords_equal(cirq.Circuit(prefix, tagged_t(a)), cirq.Circuit(tagged_t(a)))
    with pytest.raises(ValueError, match="Unsupported resource"):
        cliffords_equal(
            cirq.Circuit(cirq.I(a).with_tags(ResourceStateTag(cirq.S, 0, 0, 1))), cirq.Circuit()
        )


def test_resource_ghz_rewrite() -> None:
    a, ao, b, bo, r = map(cirq.NamedQubit, ["a_in", "a_out", "b_in", "b_out", "resource"])
    import sympy

    original = cirq.Circuit(
        tagged_t(r), cirq.CNOT(a, b), cirq.CNOT(b, r), cirq.CNOT(a, b), cirq.measure(r, key="s")
    )
    optimized = cirq.Circuit(
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
    assert cliffords_equal(original, optimized)
