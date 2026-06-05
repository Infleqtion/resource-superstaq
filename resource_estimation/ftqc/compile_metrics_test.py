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

import resource_estimation.ftqc as ftqc
import resource_estimation.ftqc.architecture as arch
import resource_estimation.ftqc.compile_ftqc as comp
import resource_estimation.ftqc.compile_metrics as compile_metrics
from resource_estimation.ftqc import Column


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
