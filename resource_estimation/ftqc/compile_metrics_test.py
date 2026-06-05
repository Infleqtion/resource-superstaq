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
import cirq
import pytest

import resource_estimation.ftqc as ftqc
import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.compile_ftqc as comp
import resource_estimation.ftqc.compile_metrics as compile_metrics
import resource_estimation.ftqc.factory_specs as factory_specs
from resource_estimation.ftqc import Column, MovementLayout


def test_ft_compile_result():
    circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0)))
    metrics = {"operation_count": 1}

    result = compile_metrics.FTCompileResult(circuit=circuit, metrics=metrics)

    assert result.circuit is circuit
    assert result.metrics is metrics


def test_ft_compile_metric_collector_defaults():
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    layout = Column(circuit)
    architecture = arch.DefaultLattice()
    collector = compile_metrics.FTCompileMetricCollector()

    input_op = cirq.CNOT(q0, q1)
    replacement_ops = [cirq.CNOT(q0, q1)]
    collector.on_logical_operation(input_op, layout, architecture)
    collector.on_replacement(input_op, replacement_ops, layout, architecture)
    collector.on_state_prep([cirq.H(q0)], layout, architecture)
    collector.on_post_op_correction(input_op, [cirq.S(q0)], layout, architecture)
    collector.on_idling(1, [cirq.I(q0)], layout, architecture)
    collector.on_moves(input_op, [cirq.SWAP(q0, q1)], layout, architecture)

    assert collector.finalize(circuit, layout, architecture) is None


def test_public_compile_metric_exports():
    assert ftqc.FTCompileResult is compile_metrics.FTCompileResult
    assert ftqc.FTCompileMetricCollector is compile_metrics.FTCompileMetricCollector
    assert ftqc.ReactionDepthMetricCollector is compile_metrics.ReactionDepthMetricCollector
    assert comp.FTCompileResult is compile_metrics.FTCompileResult
    assert comp.FTCompileMetricCollector is compile_metrics.FTCompileMetricCollector


def test_reaction_depth_metric_uses_logical_operation_hook():
    assert "on_logical_operation" in compile_metrics.ReactionDepthMetricCollector.__dict__
    assert "on_replacement" not in compile_metrics.ReactionDepthMetricCollector.__dict__


def test_reaction_depth_metric_propagates_h_clifford_axes():
    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()
    collector.reaction_depth[qubit].update({"X": 2, "Z": 5})

    collector.on_logical_operation(
        cirq.H(qubit),
        MovementLayout(cirq.Circuit(cirq.H(qubit)), factory_specs={}),
        arch.DefaultMovement(),
    )

    assert collector.reaction_depth[qubit] == {"X": 5, "Z": 2}


def test_reaction_depth_metric_splits_y_from_s_clifford():
    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()
    collector.reaction_depth[qubit].update({"X": 7, "Z": 5})

    collector.on_logical_operation(
        cirq.S(qubit),
        MovementLayout(cirq.Circuit(cirq.S(qubit)), factory_specs={}),
        arch.DefaultMovement(),
    )

    assert collector.reaction_depth[qubit] == {"X": 7, "Z": 7}


def test_reaction_depth_metric_propagates_cnot_clifford_products():
    control, target = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    collector = compile_metrics.ReactionDepthMetricCollector()
    collector.reaction_depth[control].update({"X": 2, "Z": 3})
    collector.reaction_depth[target].update({"X": 5, "Z": 7})

    collector.on_logical_operation(
        cirq.CNOT(control, target),
        MovementLayout(cirq.Circuit(cirq.CNOT(control, target)), factory_specs={}),
        arch.DefaultMovement(),
    )

    assert collector.reaction_depth[control] == {"X": 2, "Z": 7}
    assert collector.reaction_depth[target] == {"X": 5, "Z": 7}


def test_reaction_depth_metric_prefers_factory_dynamic_for_factory_gate():
    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()
    collector.reaction_depth[qubit].update({"X": 2, "Z": 1})

    collector.on_logical_operation(
        cirq.T(qubit),
        MovementLayout(cirq.Circuit(cirq.T(qubit)), num_t_factories=1),
        arch.DefaultMovement(),
    )

    assert collector.reaction_depth[qubit] == {"X": 2, "Z": 3}


def test_reaction_depth_metric_rejects_wrong_arity_factory_dynamic():
    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()

    def reaction_dynamic(
        old_depths: factory_specs.ReactionDepthState,
    ) -> factory_specs.ReactionDepthState:
        return []

    layout = MovementLayout(
        cirq.Circuit(cirq.T(qubit)),
        num_t_factories=1,
        factory_specs={
            "t": factory_specs.FactorySpec(
                name="bad-t",
                ftype="t",
                produced_gate=cirq.T,
                correction_policy=factory_specs.CorrectionPolicy(
                    name="bad-correction",
                    reaction_dynamic=reaction_dynamic,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="returned 0 updates for 1 qubits"):
        collector.on_logical_operation(cirq.T(qubit), layout, arch.DefaultMovement())


def test_reaction_depth_metric_rejects_non_factory_non_clifford():
    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()

    with pytest.raises(ValueError, match="non-Clifford operation without a factory spec"):
        collector.on_logical_operation(
            cirq.T(qubit),
            MovementLayout(cirq.Circuit(cirq.T(qubit)), num_t_factories=1, factory_specs={}),
            arch.DefaultMovement(),
        )


def test_reaction_depth_metric_wraps_clifford_conjugation_errors(monkeypatch):
    class _FailingPauliString:
        def conjugated_by(self, input_op: cirq.Operation) -> cirq.PauliString:
            raise ValueError("cannot conjugate")

    qubit = cirq.GridQubit(0, 0)
    collector = compile_metrics.ReactionDepthMetricCollector()
    collector.reaction_depth[qubit].update({"X": 1, "Z": 0})
    monkeypatch.setattr(
        collector,
        "_pauli_string_for_basis",
        lambda qubit, basis: _FailingPauliString(),
    )

    with pytest.raises(ValueError, match="non-Clifford operation without a factory spec"):
        collector.on_logical_operation(
            cirq.H(qubit),
            MovementLayout(cirq.Circuit(cirq.H(qubit)), factory_specs={}),
            arch.DefaultMovement(),
        )


def test_reaction_depth_metric_rejects_unsupported_pauli_factor():
    with pytest.raises(ValueError, match="Unsupported Pauli factor"):
        compile_metrics.ReactionDepthMetricCollector._reaction_bases_for_pauli(cirq.I)
