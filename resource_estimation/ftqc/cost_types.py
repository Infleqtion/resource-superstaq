import cirq
from typing import TypeAlias
from dataclasses import dataclass


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