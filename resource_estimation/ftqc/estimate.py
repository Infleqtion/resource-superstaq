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

import warnings
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal

import cirq
from tqdm import tqdm

if TYPE_CHECKING:
    from resource_estimation.ftqc.architecture import Architecture

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ResourceEstimator:
    """Class for resource estimator objects defined by the given architecture"""

    def __init__(self, arc: Architecture) -> None:
        self.arc = arc

    def validate_circuit_ops(self, circuit: cirq.Circuit) -> None:
        """Checks that the input circuit contains only valid operations and warns of operations still in progress"""
        unrecognized = [
            op
            for op in dict(Counter([op_.gate for op_ in circuit.all_operations()])).keys()
            if op not in self.arc.primitives
        ]
        if unrecognized:
            error_message = """This circuit has gates that are incompatible with the input architecture parameters.\nThe following gates in this circuit are not recognized:"""
            for op in unrecognized:
                error_message += f"\n{op!s}"
            raise ValueError(error_message)

    def serial_circuit_cost(
        self, circuit: cirq.Circuit, verbose: int = 0, pretty: bool = False
    ) -> dict[cirq.Gate | str, int]:
        """Counts up the total physical gates from all logical primitives in the input circuit"""
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
        return {op: val for op, val in cost.items()}

    def serial_circuit_time(self, circuit: cirq.Circuit) -> float:
        """Adds up the total physical time from all logical primitives in the input circuit"""
        self.validate_circuit_ops(circuit=circuit)
        return sum(
            map(lambda x: self.arc.total_time(self.arc.gate_cost(x)), circuit.all_operations())
        )

    def parallel_circuit_time(self, circuit: cirq.Circuit, verbose: int = 0) -> float:
        """Estimation of the critical path in the input circuit according to the most expensive operation per moment"""
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
        """Returns the circuit's critical path in terms of the logical primitive operations
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
        """Estimation of the physical operations in critical path of the input circuit according to the most expensive operation per moment"""
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
        """Calculates the physical qubit cost of the requested circuit"""
        return cirq.num_qubits(circuit) * self.arc.patch.num_physical_qubits


ReactionDepth = dict[Literal["X", "Z"], int]
_ReactionDepthState = list[ReactionDepth]
_ReactionDynamic = Callable[[_ReactionDepthState], _ReactionDepthState]


class ReactionDepthEstimator:
    """Estimator for logical reaction depth in a Clifford+T circuit.

    The factory map defines which gates are factory-backed. Operations whose
    gates are absent from that map are treated as Clifford operations and
    propagate tracked Pauli reaction depths.

    Attributes:
        factories: Gate-to-bool map selecting factory-backed gates and whether
            each factory dynamic is auto-corrected (`True`) or
            non-auto-corrected (`False`).
    """

    @staticmethod
    def _t_reaction_dynamic(
        old_depths: _ReactionDepthState,
        auto_corrected: bool,
    ) -> _ReactionDepthState:
        """Return reaction-depth updates for T factories.

        Args:
            old_depths: Single-qubit reaction-depth state before the T
                correction.
            auto_corrected: Whether to use auto-corrected T dynamics.

        Returns:
            Single-qubit update applying `newZ = max(oldZ, oldX + 1)` for
            auto-corrected T dynamics or `newX = oldX + 1` and
            `newZ = oldZ + 1` for non-auto-corrected T dynamics.
        """
        old_depth = old_depths[0]
        if auto_corrected:
            return [{"Z": max(old_depth.get("X", 0) + 1, old_depth.get("Z", 0))}]
        return [{"X": old_depth.get("X", 0) + 1, "Z": old_depth.get("Z", 0) + 1}]

    @staticmethod
    def _s_reaction_dynamic(old_depths: _ReactionDepthState) -> _ReactionDepthState:
        """Return reaction-depth updates for standard S factories.

        Args:
            old_depths: Single-qubit reaction-depth state before the S
                correction.

        Returns:
            Single-qubit update applying `newZ = max(oldZ, oldX + 1)`.
        """
        old_depth = old_depths[0]
        return [{"Z": max(old_depth.get("X", 0) + 1, old_depth.get("Z", 0))}]

    _FACTORY_REACTION_DYNAMICS: ClassVar[dict[tuple[cirq.Gate, bool], _ReactionDynamic]] = {
        (cirq.T, True): partial(_t_reaction_dynamic.__func__, auto_corrected=True),
        (cirq.T, False): partial(_t_reaction_dynamic.__func__, auto_corrected=False),
        (cirq.S, False): _s_reaction_dynamic.__func__,
    }

    def __init__(
        self,
        factories: dict[cirq.Gate, bool] | None = None,
    ) -> None:
        """Initialize reaction-depth dynamics for a factory-backed gate set.

        Args:
            factories: Optional gate-to-bool map. Each key is treated as a
                factory-backed gate, and each value selects auto-corrected
                (`True`) or non-auto-corrected (`False`) dynamics. When omitted,
                defaults are T auto-corrected and S non-auto-corrected.

        Raises:
            ValueError: If any supplied `(gate, auto_corrected)` pair has no
                defined reaction dynamic.
        """
        if factories is None:
            self.factories = {cirq.T: True, cirq.S: False}
        else:
            self.factories = factories

        unsupported_pairs = [
            (gate, auto_corrected)
            for gate, auto_corrected in self.factories.items()
            if (gate, auto_corrected) not in self._FACTORY_REACTION_DYNAMICS
        ]
        if unsupported_pairs:
            raise ValueError(
                "No reaction-depth factory dynamic is defined for: "
                + ", ".join(
                    f"({gate!r}, {auto_corrected!r})" for gate, auto_corrected in unsupported_pairs
                )
            )

    def reaction_depth(self, circuit: cirq.Circuit) -> dict[cirq.Qid, ReactionDepth]:
        """Compute reaction depth for a logical circuit.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            Per-qubit reaction-depth state keyed by the original circuit qubits.
            Each value contains the current `"X"` and `"Z"` reaction depths.
        """
        reaction_depth: defaultdict[cirq.Qid, ReactionDepth] = defaultdict(lambda: {"X": 0, "Z": 0})

        for input_op in circuit.all_operations():
            if input_op.gate not in self.factories:
                self._apply_clifford_reaction_depth(input_op, reaction_depth)
                continue

            reaction_dynamic = self._FACTORY_REACTION_DYNAMICS[
                (input_op.gate, self.factories[input_op.gate])
            ]
            old_depths = [dict(reaction_depth[qubit]) for qubit in input_op.qubits]
            new_depths = reaction_dynamic(old_depths)
            if len(new_depths) != len(input_op.qubits):
                raise ValueError(
                    "Reaction dynamic returned "
                    f"{len(new_depths)} updates for {len(input_op.qubits)} qubits."
                )
            for qubit, new_depth in zip(input_op.qubits, new_depths, strict=True):
                reaction_depth[qubit].update(new_depth)

        return {qubit: dict(depth) for qubit, depth in reaction_depth.items()}

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
        non_clifford_message = (
            "Reaction-depth estimator encountered a non-Clifford operation without a "
            f"factory dynamic: {input_op!r}."
        )
        if not cirq.has_stabilizer_effect(input_op.gate):
            raise ValueError(non_clifford_message)

        old_depths: dict[cirq.Qid, ReactionDepth] = {}
        new_depths: defaultdict[cirq.Qid, ReactionDepth] = defaultdict(lambda: {"X": 0, "Z": 0})
        for qubit in input_op.qubits:
            old_depth = reaction_depth.get(qubit, {"X": 0, "Z": 0})
            if not any(old_depth.values()):
                continue
            old_depths[qubit] = dict(old_depth)
            new_depths[qubit] = {"X": 0, "Z": 0}

        for source_qubit, source_depth in old_depths.items():
            for source_basis, depth in source_depth.items():
                source_pauli = cirq.PauliString(
                    cirq.X(source_qubit) if source_basis == "X" else cirq.Z(source_qubit)
                )
                try:
                    propagated_pauli = source_pauli.conjugated_by(input_op)
                except ValueError as exc:
                    raise ValueError(non_clifford_message) from exc

                for target_qubit in propagated_pauli.qubits:
                    target_pauli = propagated_pauli.get(target_qubit)
                    target_bases = {
                        cirq.X: ("X",),
                        cirq.Z: ("Z",),
                        cirq.Y: ("X", "Z"),
                    }[target_pauli]
                    target_depth = new_depths[target_qubit]
                    for target_basis in target_bases:
                        target_depth[target_basis] = max(target_depth[target_basis], depth)

        for qubit, new_depth in new_depths.items():
            reaction_depth[qubit].update(new_depth)
