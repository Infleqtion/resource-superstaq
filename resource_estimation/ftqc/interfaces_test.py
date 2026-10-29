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

from collections.abc import Mapping, Sequence
from unittest.mock import Mock

import cirq
import pytest

from resource_estimation.ftqc.architecture import DefaultMovement
from resource_estimation.ftqc.codepatch import CSSCodePatch
from resource_estimation.ftqc.estimate import ResourceEstimator
from resource_estimation.ftqc.interfaces import PhysicalImplementation
from resource_estimation.ftqc.layout import MovementLayout


class ReferenceCX(PhysicalImplementation):
    """A test provider, independent of any PL package."""

    def build_circuit(
        self,
        primitive: cirq.Gate,
        encoding: CSSCodePatch,
        registers: Mapping[str, Sequence[cirq.Qid]],
    ) -> cirq.Circuit:
        assert primitive == cirq.CNOT
        control, target = registers["control_data"], registers["target_data"]
        assert len(control) == len(target) == encoding.num_data_qubits
        return cirq.Circuit(
            cirq.Moment(cirq.H.on_each(*target)),
            cirq.Moment(cirq.CZ(c, t) for c, t in zip(control, target)),
            cirq.Moment(cirq.H.on_each(*target)),
        )


@pytest.mark.parametrize("distance", [3, 5])
def test_optional_implementation_counts_physical_gates(distance: int) -> None:
    implementation = Mock(wraps=ReferenceCX())
    architecture = DefaultMovement(d=distance)
    layout = MovementLayout(cirq.Circuit(cirq.CX(*cirq.LineQubit.range(2))), num_t_factories=0)
    estimator = ResourceEstimator(architecture, physical_implementation=implementation)
    assert estimator.serial_circuit_cost(layout.mapped_circuit, layout) == {
        cirq.CZ: distance**2,
        cirq.H: 2 * distance**2,
    }
    assert ResourceEstimator(architecture).serial_circuit_cost(layout.mapped_circuit, layout) == {
        cirq.CZ: distance**2,
        cirq.PhasedXZGate: 2 * distance**2,
    }
    implementation.build_circuit.assert_called_once()
    assert implementation.build_circuit.call_args.kwargs["encoding"] is architecture.patch


def test_counts_follow_returned_circuit() -> None:
    architecture = DefaultMovement(d=3)
    wires = cirq.LineQubit.range(18)
    physical = ReferenceCX().build_circuit(
        cirq.CX,
        architecture.patch,
        {"control_data": wires[:9], "target_data": wires[9:]},
    )
    # Two additional Hadamards preserve the action but incur two physical gates.
    physical += cirq.Circuit(cirq.H(wires[0]), cirq.H(wires[0]))
    implementation = Mock()
    implementation.build_circuit.return_value = physical
    estimator = ResourceEstimator(architecture, physical_implementation=implementation)
    layout = MovementLayout(cirq.Circuit(cirq.CX(*cirq.LineQubit.range(2))), num_t_factories=0)
    assert estimator.serial_circuit_cost(layout.mapped_circuit, layout) == {cirq.CZ: 9, cirq.H: 20}


def test_timing_and_paths_do_not_use_pl() -> None:
    architecture = DefaultMovement(d=3)
    layout = MovementLayout(cirq.Circuit(cirq.CX(*cirq.LineQubit.range(2))), num_t_factories=0)
    implementation = Mock()
    implementation.build_circuit.side_effect = AssertionError("Only serial counting may call PL")
    estimator = ResourceEstimator(architecture, physical_implementation=implementation)
    baseline = ResourceEstimator(architecture)
    for method in ("serial_circuit_time", "parallel_circuit_time", "parallel_circuit_cost"):
        assert getattr(estimator, method)(layout.mapped_circuit, layout) == getattr(
            baseline, method
        )(layout.mapped_circuit, layout)
    with pytest.warns(UserWarning, match="expensive"):
        assert estimator.critical_path(layout.mapped_circuit, layout) == baseline.critical_path(
            layout.mapped_circuit, layout
        )
    implementation.build_circuit.assert_not_called()


def test_undeclared_workspace() -> None:
    implementation = Mock()
    implementation.build_circuit.return_value = cirq.Circuit(cirq.H(cirq.NamedQubit("extra")))
    estimator = ResourceEstimator(DefaultMovement(d=3), physical_implementation=implementation)
    layout = MovementLayout(cirq.Circuit(cirq.CX(*cirq.LineQubit.range(2))), num_t_factories=0)
    with pytest.raises(ValueError, match="workspace"):
        estimator.serial_circuit_cost(layout.mapped_circuit, layout)


def test_implementation_failure_is_not_hidden() -> None:
    implementation = Mock()
    implementation.build_circuit.side_effect = NotImplementedError("CX not available")
    estimator = ResourceEstimator(DefaultMovement(d=3), physical_implementation=implementation)
    layout = MovementLayout(cirq.Circuit(cirq.CX(*cirq.LineQubit.range(2))), num_t_factories=0)
    with pytest.raises(NotImplementedError, match="CX not available"):
        estimator.serial_circuit_cost(layout.mapped_circuit, layout)


def test_other_primitives_retain_cost_models() -> None:
    architecture = DefaultMovement(d=3)
    layout = MovementLayout(cirq.Circuit(cirq.H(cirq.LineQubit(0))), num_t_factories=0)
    implementation = Mock()
    estimator = ResourceEstimator(architecture, physical_implementation=implementation)
    assert estimator.serial_circuit_cost(layout.mapped_circuit, layout) == ResourceEstimator(
        architecture
    ).serial_circuit_cost(layout.mapped_circuit, layout)
    implementation.build_circuit.assert_not_called()
