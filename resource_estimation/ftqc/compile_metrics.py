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

import cirq

from . import architecture as arch
from .layout import Layout


@dataclass
class FTCompileResult:
    """Container returned by `ft_compile`.

    `metrics` maps each collector name passed to `ft_compile` to that collector's
    finalized metric value.
    """

    circuit: cirq.Circuit
    metrics: dict[str, object]


class FTCompileMetricCollector:
    """Base class for compile-time metric collectors.

    Subclasses override whichever hooks are relevant to their metric and use
    `finalize` to return the metric value stored in `FTCompileResult.metrics`.
    """

    def on_replacement(
        self,
        input_op: cirq.Operation,
        replacement_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe compiler replacement of one input operation with primitive operations."""
        pass

    def on_state_prep(
        self,
        ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe initial state-preparation operations added before compiled operations."""
        pass

    def on_post_op_correction(
        self,
        input_op: cirq.Operation,
        correction_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe post-operation correction operations added after an input operation."""
        pass

    def on_idling(
        self,
        moment_idx: int,
        idling_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe idling operations added to a compiled circuit moment."""
        pass

    def on_moves(
        self,
        input_op: cirq.Operation,
        move_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe movement operations added around an input operation."""
        pass

    def finalize(
        self,
        circuit: cirq.Circuit,
        layout: Layout,
        arc: arch.Architecture,
    ) -> object:
        """Return the final metric value for this collector."""
        return None
