import pytest
import cirq
from resource_estimation.ftqc.cost_types import _require_gate_operation

def test_require_gate_operation():
    already_gate_operation = cirq.X.on(cirq.LineQubit(0))
    assert _require_gate_operation(already_gate_operation) is already_gate_operation
    not_a_gate_operation = cirq.CircuitOperation(
        cirq.FrozenCircuit(
            cirq.X.on(cirq.LineQubit(0)),
            cirq.Y.on(cirq.LineQubit(1))
        )
    )
    with pytest.raises(TypeError, match="Expected GateOperation"):
        _ = _require_gate_operation(not_a_gate_operation)