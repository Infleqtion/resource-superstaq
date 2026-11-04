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
"""Checks of affine controls independent of quantum evolution."""

import cirq
import pytest
import sympy

from resource_estimation.clifford_verification.controls import control_parity


def test_record_lookup_and_boolean_parity() -> None:
    key = cirq.MeasurementKey("m", path=("scope",))
    outcomes = [key, key]
    for index, expected in [(0, 0), (1, 1), (-1, 1), (-2, 0)]:
        assert control_parity(cirq.KeyCondition(key, index), outcomes) == ({expected}, 0)
    symbol = sympy.Symbol(str(key))
    assert control_parity(cirq.SympyCondition(sympy.Not(symbol)), outcomes) == ({1}, 1)
    for value in (sympy.true, sympy.false):
        assert control_parity(cirq.SympyCondition(value), outcomes) == (set(), int(bool(value)))
    other = cirq.MeasurementKey("n")
    outcomes.append(other)
    expr = sympy.Xor(symbol, sympy.Symbol("n"))
    assert control_parity(cirq.SympyCondition(expr), outcomes) == ({1, 2}, 0)


def test_invalid_controls() -> None:
    key = cirq.MeasurementKey("m")
    outcomes = [key]
    for condition in (
        cirq.KeyCondition(key, 1),
        cirq.KeyCondition(key, -2),
        cirq.KeyCondition(cirq.MeasurementKey("missing")),
    ):
        with pytest.raises(ValueError, match="Unavailable"):
            control_parity(condition, outcomes)
    expr = sympy.And(sympy.Symbol("m"), sympy.Symbol("n"))
    with pytest.raises(ValueError, match="XOR"):
        control_parity(cirq.SympyCondition(expr), outcomes)
