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

from collections import defaultdict
from dataclasses import dataclass

import cirq
from typing_extensions import override

from . import architecture as arch
from .factory_specs import ReactionBasis, ReactionDepth, factory_type_for_gate
from .layout import Layout


@dataclass
class FTCompileResult:
    """Compiled FTQC circuit and any metrics collected during compilation.

    Attributes:
        circuit: Fault-tolerant circuit produced by `ft_compile`.
        metrics: Final metric values keyed by the names supplied for each metric
            collector passed to `ft_compile`.
    """

    circuit: cirq.Circuit
    """Fault-tolerant circuit produced by `ft_compile`."""

    metrics: dict[str, object]
    """Final metric values keyed by collector name."""


class FTCompileMetricCollector:
    """Base class for compile-time metric collectors.

    Subclasses override whichever hooks are relevant to their metric and use
    `finalize` to return the value stored in `FTCompileResult.metrics`. Hook
    methods are called by the compiler as it emits each category of operation.
    The base hook implementations intentionally do nothing.
    """

    def on_logical_operation(
        self,
        input_op: cirq.Operation,
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe one logical operation from the mapped input circuit.

        This hook fires once for every operation in `layout.mapped_circuit`, in
        circuit order, before primitive decomposition. It is the right hook for
        metrics that care about the logical operation stream regardless of
        whether an operation is later replaced or kept as a primitive.

        Args:
            input_op: Operation from the mapped input circuit.
            layout: Layout state visible before primitive decomposition.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def on_replacement(
        self,
        input_op: cirq.Operation,
        replacement_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe replacement of a logical input operation with primitive operations.

        Args:
            input_op: Operation from the input circuit before compiler replacement.
            replacement_ops: Primitive operations emitted for `input_op`.
            layout: Layout state visible at the replacement point.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def on_state_prep(
        self,
        ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe state-preparation operations inserted before compiled operations.

        Args:
            ops: Operations used to prepare initial logical states.
            layout: Layout state visible when state preparation is emitted.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def on_post_op_correction(
        self,
        input_op: cirq.Operation,
        correction_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe correction operations inserted after an input operation.

        Args:
            input_op: Operation from the input circuit that caused the correction.
            correction_ops: Compiler-emitted correction operations.
            layout: Layout state visible when corrections are emitted.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def on_idling(
        self,
        moment_idx: int,
        idling_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe idling operations inserted for a compiled circuit moment.

        Args:
            moment_idx: Index of the compiled moment receiving idling operations.
            idling_ops: Compiler-emitted idling operations for that moment.
            layout: Layout state visible when idling is emitted.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def on_moves(
        self,
        input_op: cirq.Operation,
        move_ops: list[cirq.Operation],
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Observe movement operations inserted around an input operation.

        Args:
            input_op: Operation from the input circuit that caused movement.
            move_ops: Compiler-emitted movement operations.
            layout: Layout state visible when movement is emitted.
            arc: Architecture used for the compilation.

        Returns:
            None. Collectors should mutate their own internal state.
        """
        pass

    def finalize(
        self,
        circuit: cirq.Circuit,
        layout: Layout,
        arc: arch.Architecture,
    ) -> object:
        """Return the final metric value for this collector.

        Args:
            circuit: Fully compiled circuit.
            layout: Final layout state after compilation.
            arc: Architecture used for the compilation.

        Returns:
            The value to store in `FTCompileResult.metrics` for this collector.
            The base collector has no metric state, so it returns `None`.
        """
        return None


class ReactionDepthMetricCollector(FTCompileMetricCollector):
    """Collect per-qubit reaction depths across the logical operation stream.

    Factory gates are identified by converting `input_op.gate` to an `ftype`.
    If `layout.factory_specs` has a spec for that `ftype`, the collector applies
    the factory correction policy's reaction dynamic. Otherwise, the collector
    treats the operation as a Clifford and propagates tracked Pauli depths
    through that Clifford.

    Attributes:
        reaction_depth: Current reaction-depth state keyed by logical qubit. Each
            depth map stores the largest known `"X"` and `"Z"` reaction depths for
            that qubit.
    """

    reaction_depth: defaultdict[cirq.Qid, ReactionDepth]

    def __init__(self) -> None:
        """Initialize every newly observed qubit with zero X and Z reaction depth."""
        super().__init__()
        self.reaction_depth = defaultdict(lambda: {"X": 0, "Z": 0})

    @override
    def on_logical_operation(
        self,
        input_op: cirq.Operation,
        layout: Layout,
        arc: arch.Architecture,
    ) -> None:
        """Apply reaction-depth updates for one logical operation.

        The collector first uses the gate's `ftype` string to select this layout's
        static factory spec for that factory type. Gates without specs are treated
        as Clifford operations and update the stored Pauli reaction-depth axes by
        conjugation.

        Args:
            input_op: Logical operation being observed. Its qubits define the
                positional order passed into the factory reaction dynamic.
            layout: Layout whose `factory_specs` map identifies reaction-depth-aware
                factories by `ftype`.
            arc: Architecture used for the compilation. It is part of the hook
                contract but is not needed by the current reaction-depth formulas.

        Raises:
            ValueError: If the selected reaction dynamic returns a number of qubit
                updates different from the arity of `input_op`, or if a non-factory
                operation is not Clifford.
        """
        factory_type = factory_type_for_gate(input_op.gate)
        factory_spec = layout.factory_specs.get(factory_type)
        if factory_spec is None:
            self._apply_clifford_reaction_depth(input_op)
            return

        reaction_dynamic = factory_spec.correction_policy.reaction_dynamic
        old_depths = [dict(self.reaction_depth[qubit]) for qubit in input_op.qubits]
        new_depths = reaction_dynamic(old_depths)
        if len(new_depths) != len(input_op.qubits):
            raise ValueError(
                "Reaction dynamic returned "
                f"{len(new_depths)} updates for {len(input_op.qubits)} qubits."
            )
        for qubit, new_depth in zip(input_op.qubits, new_depths, strict=True):
            self.reaction_depth[qubit].update(new_depth)

    def _apply_clifford_reaction_depth(self, input_op: cirq.Operation) -> None:
        """Propagate reaction-depth axes through a non-factory Clifford operation.

        Each tracked single-qubit Pauli basis on the operation's qubits is
        conjugated through `input_op`. The source depth is copied to every Pauli
        factor in the conjugated product, and collisions are combined with
        `max`. A resulting Y factor updates both X and Z reaction bases. Qubits
        with no existing nonzero reaction depth are ignored so a Clifford-only
        circuit does not create zero-valued metric entries.

        Args:
            input_op: Non-factory operation to treat as a Clifford.

        Raises:
            ValueError: If `input_op` is not Clifford in the supported Cirq model.
        """
        if input_op.gate is None or not cirq.has_stabilizer_effect(input_op.gate):
            raise ValueError(
                "Reaction-depth metric encountered a non-Clifford operation without a "
                f"factory spec: {input_op!r}. Add a matching factory spec to "
                "`layout.factory_specs` to define its reaction dynamics."
            )

        old_depths: dict[cirq.Qid, ReactionDepth] = {}
        new_depths: defaultdict[cirq.Qid, ReactionDepth] = defaultdict(lambda: {"X": 0, "Z": 0})
        for qubit in input_op.qubits:
            old_depth = self.reaction_depth.get(qubit)
            if old_depth is None or not any(old_depth.values()):
                continue
            old_depths[qubit] = dict(old_depth)
            new_depths[qubit] = {"X": 0, "Z": 0}

        try:
            for source_qubit, source_depth in old_depths.items():
                for source_basis, depth in source_depth.items():
                    source_pauli = self._pauli_string_for_basis(source_qubit, source_basis)
                    propagated_pauli = source_pauli.conjugated_by(input_op)
                    for target_qubit in propagated_pauli.qubits:
                        target_pauli = propagated_pauli.get(target_qubit)
                        for target_basis in self._reaction_bases_for_pauli(target_pauli):
                            target_depth = new_depths[target_qubit]
                            target_depth[target_basis] = max(target_depth[target_basis], depth)
        except ValueError as exc:
            raise ValueError(
                "Reaction-depth metric encountered a non-Clifford operation without a "
                f"factory spec: {input_op!r}. Add a matching factory spec to "
                "`layout.factory_specs` to define its reaction dynamics."
            ) from exc

        for qubit, new_depth in new_depths.items():
            self.reaction_depth[qubit].update(new_depth)

    @staticmethod
    def _pauli_string_for_basis(
        qubit: cirq.Qid,
        basis: ReactionBasis,
    ) -> cirq.PauliString:
        """Return a single-qubit Pauli string for one tracked reaction basis.

        Args:
            qubit: Qubit carrying the tracked reaction-depth basis.
            basis: Reaction basis to convert to a Cirq Pauli string.

        Returns:
            A single-qubit `cirq.PauliString` containing X or Z on `qubit`.
        """
        return cirq.PauliString(cirq.X(qubit) if basis == "X" else cirq.Z(qubit))

    @staticmethod
    def _reaction_bases_for_pauli(pauli: cirq.Pauli) -> tuple[ReactionBasis, ...]:
        """Return the reaction bases represented by a Cirq Pauli factor.

        Args:
            pauli: Pauli factor produced by Clifford conjugation.

        Returns:
            `("X",)` for X, `("Z",)` for Z, and `("X", "Z")` for Y.

        Raises:
            ValueError: If an unexpected Pauli factor is supplied.
        """
        if pauli == cirq.X:
            return ("X",)
        if pauli == cirq.Z:
            return ("Z",)
        if pauli == cirq.Y:
            return ("X", "Z")
        raise ValueError(f"Unsupported Pauli factor for reaction-depth tracking: {pauli!r}")

    @override
    def finalize(
        self, circuit: cirq.Circuit, layout: Layout, arc: arch.Architecture
    ) -> dict[cirq.Qid, ReactionDepth]:
        """Return the accumulated reaction-depth metric.

        Args:
            circuit: Fully compiled circuit. The reaction-depth metric is already
                accumulated during hooks, so this method does not inspect it.
            layout: Final layout state after compilation.
            arc: Architecture used for the compilation.

        Returns:
            A plain dictionary copy of the current per-qubit reaction-depth state.
            Returning copies keeps callers from mutating collector internals after
            compilation.
        """
        return {qubit: dict(depth) for qubit, depth in self.reaction_depth.items()}
