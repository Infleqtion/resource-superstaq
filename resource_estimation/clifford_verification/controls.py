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
"""Measurement record identities and affine classical-control parsing."""

from __future__ import annotations

from collections.abc import Sequence

import cirq
import sympy

# Building xors for each control set, lets us not branch measurements.
def control_parity(
    condition: cirq.Condition, outcomes: Sequence[cirq.MeasurementKey]
) -> tuple[set[int], int]:
    """Resolve a single-bit key or Boolean XOR/NOT expression to record indices.

    Keys appear in measurement order; repeated keys identify separate records.
    """

    # Look up keys in out outcomes sequence
    def lookup(key: cirq.MeasurementKey, occurrence: int = -1) -> int:
        matches = [j for j, outcome in enumerate(outcomes) if outcome == key]
        if not -len(matches) <= occurrence < len(matches):
            raise ValueError(f"Unavailable measurement record {key}[{occurrence}]")
        return matches[occurrence]

    # Turn sympx xor expression into set of terms to be XOR'd and a bit to flip the value
    def parse(expression: sympy.Basic) -> tuple[set[int], int]:
        if expression == sympy.true:
            return set(), 1
        if expression == sympy.false:
            return set(), 0
        if isinstance(expression, sympy.Symbol):
            return {lookup(cirq.MeasurementKey.parse_serialized(str(expression)))}, 0
        if expression.func == sympy.Not:
            bits, constant = parse(expression.args[0])
            return bits, constant ^ 1
        if expression.func == sympy.Xor:
            bits = set()
            constant = 0
            for argument in expression.args:
                other, offset = parse(argument)
                bits ^= other
                constant ^= offset
            return bits, constant
        raise ValueError(
            "Feed-forward expressions must use Boolean symbols, XOR, NOT, or constants"
        )

    if isinstance(condition, cirq.KeyCondition):
        return {lookup(condition.key, condition.index)}, 0
    if isinstance(condition, cirq.SympyCondition):
        return parse(condition.expr)
    raise ValueError("Unsupported feed-forward condition")
