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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, NoReturn

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


PauliBasis = Literal["X", "Z"]
ReactionDepth = dict[PauliBasis, int]


ReactionTreeKey = tuple[cirq.Qid, PauliBasis]
ReactionTreeVertex = tuple[PauliBasis, cirq.Qid, int]
ReactionTreeEdge = tuple[ReactionTreeVertex, ReactionTreeVertex, int]


@dataclass
class ReactionTree:
    """DAG describing reaction-depth dependencies.

    Attributes:
        operations: Circuit operations ordered by reaction-tree time. A vertex
            with `time=0` is a root and has no operation; a vertex with `time=t`
            was produced by `operations[t - 1]`.
        vertices: All `(pauli, qubit, time)` Pauli vertices in the reaction tree.
        edges: `(source, target, weight)` dependency edges between vertices.
        depths: Longest dependency path depth for each vertex.
    """

    operations: tuple[cirq.Operation, ...] = ()
    vertices: set[ReactionTreeVertex] = field(default_factory=set)
    edges: list[ReactionTreeEdge] = field(default_factory=list)
    depths: dict[ReactionTreeVertex, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReactionDynamics:
    """Describes the reaction dynamics of a factory-type gate.

    Attributes:
        source_qubit_index: Operation-local source qubit index.
        source_pauli: Source Pauli basis before the factory operation.
        target_qubit_index: Operation-local target qubit index.
        target_pauli: Target Pauli basis after the factory operation.
        weight: Reaction-depth increment added to the source depth.
    """

    source_qubit_index: int
    source_pauli: PauliBasis
    target_qubit_index: int
    target_pauli: PauliBasis
    weight: int


class ReactionDepthEstimator:
    """Estimator for logical reaction depth in a Clifford+T circuit.

    The factory map defines which gates are factory-backed. Operations whose
    gates are absent from that map are treated as Clifford operations and
    propagate tracked Pauli reaction depths.

    Attributes:
        factories: Gate-to-bool map selecting factory-backed gates and whether
            each factory dynamic is auto-corrected (`True`) or
            non-auto-corrected (`False`).
        _reaction_dynamics: Instance-owned lookup of factory reaction dynamics
            keyed by `(gate, auto_corrected)`.
    """

    _DEFAULT_REACTION_DYNAMICS: ClassVar[
        dict[tuple[cirq.Gate, bool], tuple[ReactionDynamics, ...]]
    ] = {
        (cirq.T, True): (
            ReactionDynamics(0, "Z", 0, "Z", 0),
            ReactionDynamics(0, "X", 0, "Z", 1),
        ),
        (cirq.T, False): (
            ReactionDynamics(0, "X", 0, "X", 1),
            ReactionDynamics(0, "Z", 0, "Z", 1),
        ),
        (cirq.S, False): (
            ReactionDynamics(0, "Z", 0, "Z", 0),
            ReactionDynamics(0, "X", 0, "Z", 1),
        ),
    }

    def __init__(
        self,
        factories: dict[cirq.Gate, bool] | None = None,
        reaction_dynamics: Mapping[tuple[cirq.Gate, bool], Sequence[ReactionDynamics]]
        | None = None,
    ) -> None:
        """Initialize reaction-depth dynamics for a factory-backed gate set.

        Args:
            factories: Optional gate-to-bool map. Each key is treated as a
                factory-backed gate, and each value selects auto-corrected
                (`True`) or non-auto-corrected (`False`) dynamics. When omitted,
                defaults are T auto-corrected and S non-auto-corrected.
            reaction_dynamics: Optional custom dynamics keyed by
                `(gate, auto_corrected)`. These entries override built-in
                dynamics but only make a gate factory-backed when the gate also
                appears in `factories`.

        Raises:
            ValueError: If any supplied `(gate, auto_corrected)` pair has no
                defined reaction dynamic.
        """
        if factories is None:
            self.factories = {cirq.T: True, cirq.S: False}
        else:
            self.factories = factories

        self._reaction_dynamics = dict(self._DEFAULT_REACTION_DYNAMICS)
        if reaction_dynamics is not None:
            self._reaction_dynamics.update(
                {key: tuple(value) for key, value in reaction_dynamics.items()}
            )

        unsupported_pairs = [
            (gate, auto_corrected)
            for gate, auto_corrected in self.factories.items()
            if (gate, auto_corrected) not in self._reaction_dynamics
        ]
        if unsupported_pairs:
            raise ValueError(
                "No reaction-depth factory dynamic is defined for: "
                + ", ".join(
                    f"({gate!r}, {auto_corrected!r})" for gate, auto_corrected in unsupported_pairs
                )
            )

    def _raise_for_non_clifford_operation(
        self,
        input_op: cirq.Operation,
    ) -> NoReturn:
        """Raise when a non-factory operation cannot be propagated as a Clifford.

        Args:
            input_op: Operation that is missing a factory dynamic and cannot be
                treated as a Clifford.

        Raises:
            ValueError: Always, with a message explaining that `input_op` needs
                a factory dynamic.
        """
        message = (
            "Reaction-depth estimator encountered a non-Clifford operation without a "
            f"factory dynamic: {input_op!r}."
        )
        raise ValueError(message)

    def _propagated_clifford_bases(
        self,
        input_op: cirq.Operation,
        source_qubit: cirq.Qid,
        source_basis: PauliBasis,
    ) -> tuple[tuple[cirq.Qid, PauliBasis], ...]:
        """Propagate Pauli basis through a Clifford operation.

        Args:
            input_op: Clifford operation through which the Pauli is propagated.
            source_qubit: Qubit carrying the source Pauli before `input_op`.
            source_basis: Source Pauli basis to propagate.

        Returns:
            Target qubit and basis pairs after propagation. A propagated Y Pauli
            is split into both X and Z basis dependencies.

        Raises:
            ValueError: If Cirq cannot conjugate the source Pauli by `input_op`.
        """
        source_pauli = cirq.PauliString(
            cirq.X(source_qubit) if source_basis == "X" else cirq.Z(source_qubit)
        )
        propagated_pauli = source_pauli.conjugated_by(input_op)

        propagated_bases: list[tuple[cirq.Qid, PauliBasis]] = []
        for target_qubit in propagated_pauli.qubits:
            target_pauli = propagated_pauli.get(target_qubit)
            target_bases: tuple[PauliBasis, ...] = {
                cirq.X: ("X",),
                cirq.Z: ("Z",),
                cirq.Y: ("X", "Z"),
            }[target_pauli]
            for target_basis in target_bases:
                propagated_bases.append((target_qubit, target_basis))
        return tuple(propagated_bases)

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
                if not cirq.has_stabilizer_effect(input_op.gate):
                    self._raise_for_non_clifford_operation(input_op)

                old_depths: dict[cirq.Qid, ReactionDepth] = {}
                new_depths: defaultdict[cirq.Qid, ReactionDepth] = defaultdict(
                    lambda: {"X": 0, "Z": 0}
                )
                for qubit in input_op.qubits:
                    old_depth = reaction_depth.get(qubit, {"X": 0, "Z": 0})
                    if not any(old_depth.values()):
                        continue
                    old_depths[qubit] = dict(old_depth)
                    new_depths[qubit] = {"X": 0, "Z": 0}

                for source_qubit, source_depth in old_depths.items():
                    for source_basis, depth in source_depth.items():
                        for target_qubit, target_basis in self._propagated_clifford_bases(
                            input_op,
                            source_qubit,
                            source_basis,
                        ):
                            target_depth = new_depths[target_qubit]
                            target_depth[target_basis] = max(target_depth[target_basis], depth)

                for qubit, new_depth in new_depths.items():
                    reaction_depth[qubit].update(new_depth)
                continue

            reaction_dynamic = self._reaction_dynamics[
                (input_op.gate, self.factories[input_op.gate])
            ]
            new_depths: list[ReactionDepth] = [{} for _ in input_op.qubits]
            for term in reaction_dynamic:
                source_depth = reaction_depth[input_op.qubits[term.source_qubit_index]].get(
                    term.source_pauli,
                    0,
                )
                target_depth = new_depths[term.target_qubit_index]
                target_depth[term.target_pauli] = max(
                    target_depth.get(term.target_pauli, 0),
                    source_depth + term.weight,
                )
            for qubit, new_depth in zip(input_op.qubits, new_depths, strict=True):
                reaction_depth[qubit].update(new_depth)

        return {qubit: dict(depth) for qubit, depth in reaction_depth.items()}

    def reaction_tree(self, circuit: cirq.Circuit) -> ReactionTree:
        """Build a DAG for reaction-depth dependencies.
        Tree depth is the longest path from `time=0` root vertices
        to the final time-layer vertices.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            Reaction tree with vertices, edges, final time-layer vertices,
            operation metadata, and per-vertex longest-path depths.
        """
        operations = tuple(circuit.all_operations())
        tree = ReactionTree(operations=operations)
        tracked_nodes = tuple(
            (qubit, basis) for qubit in sorted(circuit.all_qubits()) for basis in ("X", "Z")
        )
        for time in range(len(operations) + 1):
            for qubit, basis in tracked_nodes:
                vertex = (basis, qubit, time)
                tree.vertices.add(vertex)
                tree.depths[vertex] = 0

        for time, input_op in enumerate(operations, start=1):
            dependencies: list[tuple[ReactionTreeKey, ReactionTreeKey, int]] = []
            if input_op.gate in self.factories:
                reaction_dynamic = self._reaction_dynamics[
                    (input_op.gate, self.factories[input_op.gate])
                ]
                for term in reaction_dynamic:
                    dependencies.append(
                        (
                            (input_op.qubits[term.source_qubit_index], term.source_pauli),
                            (input_op.qubits[term.target_qubit_index], term.target_pauli),
                            term.weight,
                        )
                    )
            else:
                if not cirq.has_stabilizer_effect(input_op.gate):
                    self._raise_for_non_clifford_operation(input_op)
                for source_qid in input_op.qubits:
                    for source_basis in ("X", "Z"):
                        for target_node in self._propagated_clifford_bases(
                            input_op,
                            source_qid,
                            source_basis,
                        ):
                            dependencies.append(
                                (
                                    (source_qid, source_basis),
                                    target_node,
                                    0,
                                )
                            )

            target_nodes = {target_node for _, target_node, _ in dependencies}
            dependencies.extend(
                (node, node, 0) for node in tracked_nodes if node not in target_nodes
            )
            for source_node, target_node, weight in dependencies:
                source = (source_node[1], source_node[0], time - 1)
                target = (target_node[1], target_node[0], time)
                tree.edges.append((source, target, weight))
                tree.depths[target] = max(tree.depths[target], tree.depths[source] + weight)

        return tree
