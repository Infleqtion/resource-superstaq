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
import pytest

from resource_estimation.typing import _require_gate_operation


def test_require_gate_operation() -> None:
    already_gate_operation = cirq.X.on(cirq.LineQubit(0))
    assert _require_gate_operation(already_gate_operation) is already_gate_operation
    not_a_gate_operation = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X.on(cirq.LineQubit(0)), cirq.Y.on(cirq.LineQubit(1)))
    )
    with pytest.raises(TypeError, match="Expected GateOperation"):
        _ = _require_gate_operation(not_a_gate_operation)
