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
import sys
import types
import typing

import cirq
import numpy as np
import pytest
import stim
from qldpc import codes
from qldpc.objects import Pauli

import resource_estimation.ftqc.lattice_surgery_primitives as lsp
import resource_estimation.ftqc.qldpc_surgery as qldpc_surgery


@pytest.fixture
def fake_surgery_module(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, typing.Any]:
    calls: dict[str, typing.Any] = {"gadgets": []}
    module = types.ModuleType("qldpc.circuits.surgery")

    def build_gadget(
        code: codes.CSSCode,
        operator: np.ndarray,
        *,
        basis: Pauli,
    ) -> types.SimpleNamespace:
        gadget = types.SimpleNamespace(code=code, operator=operator, basis=basis)
        calls["gadgets"].append(gadget)
        return gadget

    def build_bridge(
        left_gadget: types.SimpleNamespace,
        right_gadget: types.SimpleNamespace,
    ) -> types.SimpleNamespace:
        bridge = types.SimpleNamespace(left=left_gadget, right=right_gadget)
        calls["bridge"] = bridge
        return bridge

    def build_joint_ppm_resource_circuit(
        left_gadget: types.SimpleNamespace,
        right_gadget: types.SimpleNamespace,
        bridge: types.SimpleNamespace,
        *,
        rounds: int,
    ) -> types.SimpleNamespace:
        calls["resource_args"] = (left_gadget, right_gadget, bridge, rounds)
        circuit = stim.Circuit(
            """
            R 0 1
            TICK
            H 0
            TICK
            CX 0 1
            TICK
            M 0 1
            TICK
            """,
        )
        return types.SimpleNamespace(circuit=circuit)

    module.build_gadget = build_gadget  # type: ignore[attr-defined]
    module.build_bridge = build_bridge  # type: ignore[attr-defined]
    module.build_joint_ppm_resource_circuit = (  # type: ignore[attr-defined]
        build_joint_ppm_resource_circuit
    )
    monkeypatch.setitem(sys.modules, "qldpc.circuits.surgery", module)
    return calls


@pytest.mark.parametrize(
    ("pauli_product", "expected_pauli"),
    (("XX", Pauli.X), ("ZZ", Pauli.Z)),
)
def test_logical_ppm_resource_cost(
    fake_surgery_module: dict[str, typing.Any],
    pauli_product: lsp.LogicalPauliProduct,
    expected_pauli: Pauli,
) -> None:
    code = codes.CSSCode(
        np.zeros((0, 2), dtype=int),
        np.zeros((0, 2), dtype=int),
    )
    q0, q1 = cirq.LineQubit.range(2)
    operation = lsp.LogicalPPM(pauli_product).on(q0, q1)

    cost = qldpc_surgery.logical_ppm_resource_cost(
        operation,
        code,
        code,
        rounds=3,
        left_logical_index=1,
        right_logical_index=0,
    )

    left_gadget, right_gadget = fake_surgery_module["gadgets"]
    assert left_gadget.basis is expected_pauli
    assert right_gadget.basis is expected_pauli
    np.testing.assert_array_equal(left_gadget.operator, [0, 1])
    np.testing.assert_array_equal(right_gadget.operator, [1, 0])
    assert fake_surgery_module["resource_args"][3] == 3
    assert cost == {
        "gate_cost": collections.Counter(
            {
                cirq.ResetChannel: 2,
                cirq.PhasedXZGate: 1,
                cirq.CZ: 1,
                cirq.MeasurementGate: 2,
            },
        ),
        "moment_cost": collections.Counter(
            {
                cirq.ResetChannel: 1,
                cirq.PhasedXZGate: 1,
                cirq.CZ: 1,
                cirq.MeasurementGate: 1,
            },
        ),
        "num_physical_qubits": 2,
    }


def test_logical_ppm_resource_cost_validation() -> None:
    code = codes.SteaneCode()
    q0, q1 = cirq.LineQubit.range(2)
    operation = lsp.LogicalPPM("XX").on(q0, q1)

    with pytest.raises(TypeError, match="LogicalPPM"):
        qldpc_surgery.logical_ppm_resource_cost(cirq.X(q0), code, code, rounds=1)
    with pytest.raises(TypeError, match="CSSCode"):
        qldpc_surgery.logical_ppm_resource_cost(
            operation,
            typing.cast(codes.CSSCode, object()),
            code,
            rounds=1,
        )
    with pytest.raises(TypeError, match="rounds must be an integer"):
        qldpc_surgery.logical_ppm_resource_cost(operation, code, code, rounds=True)
    with pytest.raises(ValueError, match="rounds must be positive"):
        qldpc_surgery.logical_ppm_resource_cost(operation, code, code, rounds=0)
    with pytest.raises(TypeError, match="logical indices must be integers"):
        qldpc_surgery.logical_ppm_resource_cost(
            operation,
            code,
            code,
            rounds=1,
            left_logical_index=True,
        )
    with pytest.raises(ValueError, match="logical indices must be nonnegative"):
        qldpc_surgery.logical_ppm_resource_cost(
            operation,
            code,
            code,
            rounds=1,
            left_logical_index=-1,
        )
    with pytest.raises(ValueError, match="out of range"):
        qldpc_surgery.logical_ppm_resource_cost(
            operation,
            code,
            code,
            rounds=1,
            left_logical_index=1,
        )
