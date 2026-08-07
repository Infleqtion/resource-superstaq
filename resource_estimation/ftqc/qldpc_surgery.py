# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
from typing import Any, Literal, cast

import numpy as np

from resource_estimation.ftqc.lattice_surgery_primitives import CodePatch

LogicalPauliBasis = Literal["X", "Z"]


def _logical_vector(
    patch: CodePatch,
    logical_index: int,
    basis: LogicalPauliBasis,
) -> np.ndarray:
    if not isinstance(logical_index, int) or isinstance(logical_index, bool):
        raise TypeError("logical index must be an integer")
    if logical_index < 0:
        raise ValueError("logical index must be nonnegative")
    if logical_index >= len(patch.logical_qubits):
        raise ValueError(
            f"logical index {logical_index} is out of range for a patch with "
            f"{len(patch.logical_qubits)} logical qubits"
        )

    logical_qubit = patch.logical_qubits[logical_index]
    support = logical_qubit.x_support if basis == "X" else logical_qubit.z_support
    vector = np.zeros(patch.num_data_qubits, dtype=np.uint8)
    vector[list(support)] = 1
    return vector


def build_joint_logical_pauli_measurement_circuit(
    left_patch: CodePatch,
    right_patch: CodePatch,
    *,
    basis: LogicalPauliBasis,
    rounds: int,
    left_logical_index: int = 0,
    right_logical_index: int = 0,
) -> Any:
    """Build qLDPC's cost-only bridge circuit for a two-patch logical PPM.

    This intentionally uses the public ``build_gadget`` -> ``build_bridge`` ->
    ``build_joint_ppm_resource_circuit`` path used by commit 88079892. The returned circuit
    includes the joint-code rounds and temporary-qubit detachment, but not ordinary syndrome
    extraction on the separated input patches.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be positive")

    try:
        from qldpc.circuits.surgery import (
            build_bridge,
            build_gadget,
            build_joint_ppm_resource_circuit,
        )
        from qldpc.codes import CSSCode
        from qldpc.objects import Pauli
    except ImportError as ex:  # pragma: no cover - incompatible qLDPC installations only
        raise ImportError(
            "Magic-state code teleportation requires a qLDPC version with the public "
            "surgery resource-circuit API."
        ) from ex

    left_code = left_patch.qldpc_code
    right_code = right_patch.qldpc_code
    if not isinstance(left_code, CSSCode) or not isinstance(right_code, CSSCode):
        raise TypeError("Logical Pauli measurement requires CSS CodePatch inputs")

    # A CodePatch can be used as the prototype for several logical locations. Clone one code
    # when both operands share that prototype so qLDPC constructs an inter-code adapter.
    if left_code is right_code:
        right_code = copy.deepcopy(right_code)

    pauli = cast(Any, Pauli.X if basis == "X" else Pauli.Z)
    left_vector = _logical_vector(left_patch, left_logical_index, basis)
    right_vector = _logical_vector(right_patch, right_logical_index, basis)
    left_gadget = build_gadget(left_code, left_vector, basis=pauli)
    right_gadget = build_gadget(right_code, right_vector, basis=pauli)
    bridge = build_bridge(left_gadget, right_gadget)
    return build_joint_ppm_resource_circuit(
        left_gadget,
        right_gadget,
        bridge,
        rounds=rounds,
    )
