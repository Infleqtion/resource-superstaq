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

from resource_estimation.compile_gateset import (
    CliffRzGateset,
    clifford_t_gateset,
    compile_gateset,
)


def test_compile_cliff_rz_gateset() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Rx(rads=pi / 7).on(q))

    compiled = compile_gateset(circuit, gateset=CliffRzGateset())
    allowed = cirq.Gateset(cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.Rz)

    assert all(op.gate in allowed for op in compiled.all_operations())


def test_compile_clifford_t_gateset() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Rz(rads=pi / 7).on(q))

    compiled = compile_gateset(circuit, gateset=clifford_t_gateset(atol=1e-3), verbose=False)
    allowed = cirq.Gateset(cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.T)

    assert all(op.gate in allowed for op in compiled.all_operations())


def test_compile_clifford_t_gateset_requires_atol() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Rz(rads=pi / 7).on(q))
    gateset = cirq.Gateset(cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.T)

    with pytest.raises(ValueError, match="_atol"):
        compile_gateset(circuit, gateset=gateset, verbose=False)
