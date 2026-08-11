from dataclasses import dataclass
from typing import TypeAlias

import cirq

GateKey: TypeAlias = type[cirq.Gate] | cirq.Gate
GateCounts: TypeAlias = dict[GateKey, int]
StrCounts: TypeAlias = dict[str, int]

@dataclass
class CountsDict:
    serial: GateCounts
    parallel: GateCounts

@dataclass
class CostDict:
    op_time: float
    gate_cost: GateCounts
    moment_cost: GateCounts

def _require_gate_operation(op: cirq.Operation) -> cirq.GateOperation:
    if not isinstance(op, cirq.GateOperation):
        raise TypeError(f"Expected GateOperation, got {type(op).__name__}")
    return op