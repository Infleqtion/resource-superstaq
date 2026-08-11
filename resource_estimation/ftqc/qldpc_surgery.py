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

import collections
import typing

import cirq
import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.stim_functions import count_stim_resources


class LogicalPPMResourceCost(typing.TypedDict):
    gate_cost: collections.Counter[cirq.Gate]
    moment_cost: collections.Counter[cirq.Gate]
    num_physical_qubits: int


def _logical_operator(
    code: codes.CSSCode,
    pauli: Pauli,
    logical_index: int,
) -> np.ndarray:
    if not isinstance(logical_index, int) or isinstance(logical_index, bool):
        raise TypeError("logical indices must be integers")
    if logical_index < 0:
        raise ValueError("logical indices must be nonnegative")

    logical_operators = code.get_logical_ops(pauli)
    if logical_index >= len(logical_operators):
        raise ValueError(
            f"logical index {logical_index} is out of range for a code with "
            f"{len(logical_operators)} logical qubits",
        )
    return logical_operators[logical_index]


def logical_ppm_resource_cost(
    operation: cirq.Operation,
    left_code: codes.CSSCode,
    right_code: codes.CSSCode,
    *,
    rounds: int,
    left_logical_index: int = 0,
    right_logical_index: int = 0,
) -> LogicalPPMResourceCost:
    """Return the raw qLDPC surgery resources for a logical XX or ZZ measurement.

    This cost excludes architecture-specific movement, post-detachment syndrome
    extraction, and measurement-conditioned Pauli-frame updates.
    """
    if not isinstance(operation.gate, lsp.LogicalPPM):
        raise TypeError("operation must use a LogicalPPM gate")
    if not isinstance(left_code, codes.CSSCode) or not isinstance(right_code, codes.CSSCode):
        raise TypeError("left_code and right_code must be qLDPC CSSCode objects")
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("rounds must be an integer")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    try:
        from qldpc.circuits.surgery import (
            build_bridge,
            build_gadget,
            build_joint_ppm_resource_circuit,
        )
    except ImportError as ex:  # pragma: no cover - requires an incompatible qLDPC installation
        raise ImportError(
            "LogicalPPM costing requires a qLDPC version with the surgery resource-circuit API.",
        ) from ex

    pauli = Pauli.X if operation.gate.pauli_product == "XX" else Pauli.Z
    left_operator = _logical_operator(left_code, pauli, left_logical_index)
    right_operator = _logical_operator(right_code, pauli, right_logical_index)
    left_gadget = build_gadget(left_code, left_operator, basis=pauli)
    right_gadget = build_gadget(right_code, right_operator, basis=pauli)
    bridge = build_bridge(left_gadget, right_gadget)
    resource = build_joint_ppm_resource_circuit(
        left_gadget,
        right_gadget,
        bridge,
        rounds=rounds,
    )
    resources = count_stim_resources(resource.circuit)
    return {
        "gate_cost": resources["serial"],
        "moment_cost": resources["parallel"],
        "num_physical_qubits": resource.circuit.num_qubits,
    }
