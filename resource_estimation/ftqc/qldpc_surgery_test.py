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

import pytest

from resource_estimation.ftqc.lattice_surgery_primitives import CodePatch
from resource_estimation.ftqc.qldpc_surgery import (
    _logical_vector,
    build_joint_logical_pauli_measurement_circuit,
)


def test_build_joint_logical_pauli_measurement_circuit_supports_cross_code_pair() -> None:
    surface_patch = CodePatch("surface", d=3)
    steane_patch = CodePatch("steane")

    resource = build_joint_logical_pauli_measurement_circuit(
        surface_patch,
        steane_patch,
        basis="Z",
        rounds=1,
    )

    assert len(resource.left_data_ids) == surface_patch.num_data_qubits
    assert len(resource.right_data_ids) == steane_patch.num_data_qubits
    assert set(resource.left_data_ids).isdisjoint(resource.right_data_ids)
    assert (
        resource.circuit.num_qubits > surface_patch.num_data_qubits + steane_patch.num_data_qubits
    )


def test_logical_vector_rejects_invalid_logical_index() -> None:
    patch = CodePatch("surface", d=3)

    with pytest.raises(TypeError, match="must be an integer"):
        _logical_vector(patch, True, "Z")
    with pytest.raises(ValueError, match="nonnegative"):
        _logical_vector(patch, -1, "Z")
    with pytest.raises(ValueError, match="out of range"):
        _logical_vector(patch, 1, "Z")


def test_build_joint_logical_pauli_measurement_circuit_validates_arguments() -> None:
    patch = CodePatch("surface", d=3)

    with pytest.raises(ValueError, match="basis"):
        build_joint_logical_pauli_measurement_circuit(
            patch,
            patch,
            basis="Y",  # type: ignore[arg-type]
            rounds=1,
        )
    with pytest.raises(ValueError, match="rounds"):
        build_joint_logical_pauli_measurement_circuit(patch, patch, basis="Z", rounds=0)
