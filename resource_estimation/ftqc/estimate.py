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

import copy
import warnings
from collections import Counter, defaultdict

import cirq

from resource_estimation.ftqc.architecture import Architecture
from resource_estimation.ftqc.factory_specs import (
    ReactionBasis,
    ReactionDepth,
    factory_type_for_gate,
)
from resource_estimation.ftqc.layout import Layout
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ResourceEstimator:
    """
    Class for resource estimator objects defined by the given architecture
    """

    def __init__(self, arc: Architecture) -> None:
        self.arc = arc

    def reaction_depth(self, layout: Layout) -> dict[cirq.Qid, ReactionDepth]:
        """Compute reaction depth for the logical operation stream of a layout.

        Reaction depth is a logical post-processing metric: it uses the mapped
        logical circuit and the layout's static `factory_specs`, not the compiled
        primitive circuit. The layout is copied and reset before analysis so this
        method does not consume factories or mutate caller-owned layout state.

        Args:
            layout: Layout containing the logical circuit and factory specs to
                use for reaction-depth dynamics.

        Returns:
            Per-qubit reaction-depth state. Each value contains the current `"X"`
            and `"Z"` reaction depths for that logical qubit.
        """
        layout = copy.deepcopy(layout)
        layout.reset_graph()
        reaction_depth: defaultdict[cirq.Qid, ReactionDepth] = defaultdict(lambda: {"X": 0, "Z": 0})

        for input_op in layout.mapped_circuit.all_operations():
            self._update_reaction_depth_for_logical_operation(
                input_op=input_op,
                layout=layout,
                reaction_depth=reaction_depth,
            )

        return {qubit: dict(depth) for qubit, depth in reaction_depth.items()}

    def _update_reaction_depth_for_logical_operation(
        self,
        input_op: cirq.Operation,
        layout: Layout,
        reaction_depth: defaultdict[cirq.Qid, ReactionDepth],
    ) -> None:
        """Apply one logical operation to accumulated reaction-depth state.

        Args:
            input_op: Logical operation from `layout.mapped_circuit`.
            layout: Layout whose `factory_specs` map defines factory-backed
                reaction dynamics.
            reaction_depth: Mutable per-qubit reaction-depth state to update.

        Raises:
            ValueError: If a factory dynamic returns the wrong number of qubit
                updates, or if a non-factory operation is not Clifford.
        """
        factory_type = factory_type_for_gate(input_op.gate)
        factory_spec = layout.factory_specs.get(factory_type)
        if factory_spec is None:
            self._apply_clifford_reaction_depth(input_op, reaction_depth)
            return

        reaction_dynamic = factory_spec.correction_policy.reaction_dynamic
        old_depths = [dict(reaction_depth[qubit]) for qubit in input_op.qubits]
        new_depths = reaction_dynamic(old_depths)
        if len(new_depths) != len(input_op.qubits):
            raise ValueError(
                "Reaction dynamic returned "
                f"{len(new_depths)} updates for {len(input_op.qubits)} qubits."
            )
        for qubit, new_depth in zip(input_op.qubits, new_depths, strict=True):
            reaction_depth[qubit].update(new_depth)

    def _apply_clifford_reaction_depth(
        self,
        input_op: cirq.Operation,
        reaction_depth: defaultdict[cirq.Qid, ReactionDepth],
    ) -> None:
        """Propagate tracked Pauli reaction depths through a Clifford operation.

        Args:
            input_op: Non-factory operation to treat as a Clifford.
            reaction_depth: Mutable per-qubit reaction-depth state to update.

        Raises:
            ValueError: If `input_op` is not Clifford in the supported Cirq
                model.
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
            old_depth = reaction_depth.get(qubit)
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
            reaction_depth[qubit].update(new_depth)

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

    def validate_circuit_ops(self, circuit: cirq.Circuit) -> None:
        """
        Checks that the input circuit contains only valid operations and warns of operations still in progress
        """
        unrecognized = [
            op
            for op in dict(Counter([op_.gate for op_ in circuit.all_operations()])).keys()
            if op not in self.arc.primitives
        ]
        if unrecognized:
            error_message = f"""This circuit has gates that are incompatible with the input architecture parameters.\nThe following gates in this circuit are not recognized:"""
            for op in unrecognized:
                error_message += f"\n{str(op)}"
            raise ValueError(error_message)

    def serial_circuit_cost(
        self, circuit: cirq.Circuit, verbose: int = 0, pretty: bool = False
    ) -> dict[cirq.Gate | str, int]:
        """
        Counts up the total physical gates from all logical primitives in the input circuit
        """
        self.validate_circuit_ops(circuit=circuit)
        cost = Counter()
        for op in tqdm(
            circuit.all_operations(),
            total=len(list(circuit.all_operations())),
            colour="cyan",
            disable=not bool(verbose),
        ):
            cost += Counter(self.arc.gate_cost(op))
        if pretty:
            return {
                obj.__name__ if hasattr(obj, "__name__") else str(obj): val
                for obj, val in cost.items()
            }
        else:
            return {op: val for op, val in cost.items()}

    def serial_circuit_time(self, circuit: cirq.Circuit) -> float:
        """
        Adds up the total physical time from all logical primitives in the input circuit
        """
        self.validate_circuit_ops(circuit=circuit)
        return sum(
            map(lambda x: self.arc.total_time(self.arc.gate_cost(x)), circuit.all_operations())
        )

    def parallel_circuit_time(self, circuit: cirq.Circuit, verbose: int = 0) -> float:
        """
        Estimation of the critical path in the input circuit according to the most expensive operation per moment
        """
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            big_time = max(qubit_times[q] for q in op.qubits)
            big_time += self.arc.op_time(op)
            for qubit in op.qubits:
                qubit_times[qubit] = big_time
        return max(qubit_times.values())

    def critical_path(self, circuit: cirq.Circuit, verbose: int = 0) -> list[cirq.Operation]:
        """
        Returns the circuit's critical path in terms of the logical primitive operations
        Is very slow and expensive
        """
        warnings.warn(
            "This function can be very expensive.\nIf you just want the physical operations or circuit time, use `critical_path_ops` or `parallel_circuit_time` instead."
        )
        qubit_paths = {qubit: [] for qubit in circuit.all_qubits()}
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(),
            disable=not verbose,
            total=total_ops,
            colour="cyan",
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.get)
            big_path = qubit_paths[big_qubit]
            big_time = qubit_times[big_qubit]
            big_path.append(op)
            big_time += self.arc.op_time(op)
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path.copy()
                qubit_times[qubit] = big_time
        critical_qubit = max(qubit_times, key=qubit_times.get)
        critical_path = qubit_paths[critical_qubit]
        return critical_path

    def parallel_circuit_cost(
        self, circuit: cirq.Circuit, verbose: int = 0, pretty: bool = False
    ) -> dict[cirq.Gate | str, int]:
        """
        Estimation of the physical operations in critical path of the input circuit according to the most expensive operation per moment
        """
        qubit_paths = {qubit: Counter() for qubit in circuit.all_qubits()}
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.get)
            big_time = qubit_times[big_qubit] + self.arc.op_time(op)
            big_path = qubit_paths[big_qubit] + Counter(self.arc.moment_cost(op))
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path
                qubit_times[qubit] = big_time

        big_qubit = max(op_qubits, key=qubit_times.get)
        big_time = qubit_times[big_qubit]
        big_path = qubit_paths[big_qubit]

        if pretty:
            big_path = {
                obj.__name__ if hasattr(obj, "__name__") else str(obj): val
                for obj, val in big_path.items()
            }
        return big_path

    def physical_qubits(self, circuit: cirq.Circuit) -> int:
        """
        Calculates the physical qubit cost of the requested circuit
        """
        return cirq.num_qubits(circuit) * self.arc.patch.num_physical_qubits
