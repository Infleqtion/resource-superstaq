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
