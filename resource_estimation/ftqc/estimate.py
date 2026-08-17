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
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cirq
import networkx as nx
from tqdm import tqdm

if TYPE_CHECKING:
    from resource_estimation.ftqc.architecture import Architecture
    from resource_estimation.ftqc.layout import MovementLayout

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ResourceEstimator:
    """Class for resource estimator objects defined by the given architecture"""

    def __init__(self, arc: Architecture, layout: MovementLayout | None = None) -> None:
        self.arc = arc
        self.layout = layout

    def validate_circuit_ops(self, circuit: cirq.Circuit) -> None:
        """Checks that the input circuit contains only valid operations and warns of operations still in progress"""
        unrecognized = [
            op
            for op in dict(
                collections.Counter([op_.gate for op_ in circuit.all_operations()])
            ).keys()
            if op not in self.arc.primitives
        ]
        if unrecognized:
            error_message = """This circuit has gates that are incompatible with the input architecture parameters.\nThe following gates in this circuit are not recognized:"""
            for op in unrecognized:
                error_message += f"\n{op!s}"
            raise ValueError(error_message)

    def serial_circuit_cost(
        self,
        circuit: cirq.Circuit,
        verbose: int = 0,
        pretty: bool = False,
    ) -> dict[cirq.Gate | str, int]:
        """Counts up the total physical gates from all logical primitives in the input circuit"""
        self.validate_circuit_ops(circuit=circuit)
        cost = collections.Counter()
        for op in tqdm(
            circuit.all_operations(),
            total=len(list(circuit.all_operations())),
            colour="cyan",
            disable=not bool(verbose),
        ):
            cost += collections.Counter(self.arc.gate_cost(op))
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
            map(lambda x: self.arc.total_time(self.arc.gate_cost(x)), circuit.all_operations()),
        )

    def parallel_circuit_time(self, circuit: cirq.Circuit, verbose: int = 0) -> float:
        """Estimation of the critical path in the input circuit according to the most expensive operation per moment"""
        qubit_times = dict.fromkeys(circuit.all_qubits(), 0)
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
            "This function can be very expensive.\nIf you just want the physical operations or circuit time, use `critical_path_ops` or `parallel_circuit_time` instead.",
        )
        qubit_paths = {qubit: [] for qubit in circuit.all_qubits()}
        qubit_times = dict.fromkeys(circuit.all_qubits(), 0)
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
        self,
        circuit: cirq.Circuit,
        verbose: int = 0,
        pretty: bool = False,
    ) -> dict[cirq.Gate | str, int]:
        """Estimation of the physical operations in critical path of the input circuit according to the most expensive operation per moment"""
        qubit_paths = {qubit: collections.Counter() for qubit in circuit.all_qubits()}
        qubit_times = dict.fromkeys(circuit.all_qubits(), 0)
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.get)
            big_time = qubit_times[big_qubit] + self.arc.op_time(op)
            big_path = qubit_paths[big_qubit] + collections.Counter(self.arc.moment_cost(op))
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
        if self.arc.movement and self.layout is not None:
            return self.layout.num_physical_qubits
        return cirq.num_qubits(circuit) * self.arc.patch.num_physical_qubits


ReactionTreeNode = tuple[int, int]


@dataclass(frozen=True)
class ReactionDynamics:
    """Reaction dynamic for each delayed choice measurement in a factory.

    Attributes:
        dependency_paulis: Paulis which create a dependency when any one of them
            anti-commutes with a propagated Pauli.
        outputs: Pauli corrections output by the vertex. Dynamics should
            use `cirq.LineQubit(i)` for operation-local qubit `i`.
    """

    dependency_paulis: tuple[cirq.PauliString, ...]
    outputs: tuple[cirq.PauliString, ...]


class ReactionDepthEstimator:
    """Estimator for logical reaction depth in a Clifford+factory circuit.

    Factory operations become graph vertices. Their output Paulis propagate
    through later Clifford operations. An output creates a dependency when it
    anti-commutes with any of the target vertex's dependency Paulis.
    """

    _DEFAULT_FACTORIES: ClassVar[dict[cirq.Gate, bool]] = {
        cirq.T: True,
        cirq.S: True,
        cirq.CCZ: True,
    }
    _NON_CLIFFORD_ERROR: ClassVar[str] = (
        "Reaction-depth estimator encountered a non-Clifford operation without "
        "factory reaction dynamics: {operation!r}."
    )

    def __init__(
        self,
        factories: dict[cirq.Gate, bool] | None = None,
        factory_reaction_dynamics: Mapping[
            tuple[cirq.Gate, bool],
            Sequence[ReactionDynamics],
        ]
        | None = None,
    ) -> None:
        """Configure factory gates and their reaction dynamics.

        Args:
            factories: Factory gates mapped to their auto-correction setting.
                Defaults to auto-corrected T, S, and CCZ gates.
            factory_reaction_dynamics: Factory reaction dynamics keyed by
                `(gate, auto_corrected)`. Entries override defaults;
                `factories` still determines which gates are factory-backed.
                Pauli qubits are operation-local `cirq.LineQubit` indices from
                zero through the gate's arity minus one.

        Raises:
            ValueError: If a configured factory has no reaction dynamics or a
                dynamic uses a qubit index outside the factory gate's arity.
        """
        self.factories = dict(self._DEFAULT_FACTORIES if factories is None else factories)
        local_qubits = cirq.LineQubit.range(3)
        local_x = cirq.PauliString(cirq.X(local_qubits[0]))
        local_z = tuple(cirq.PauliString(cirq.Z(qubit)) for qubit in local_qubits)
        self._factory_reaction_dynamics: dict[
            tuple[cirq.Gate, bool], tuple[ReactionDynamics, ...]
        ] = {
            (cirq.T, True): (ReactionDynamics((local_z[0],), (local_z[0],)),),
            (cirq.T, False): (ReactionDynamics((local_x, local_z[0]), (local_x, local_z[0])),),
            (cirq.S, True): (ReactionDynamics((local_z[0],), (local_z[0],)),),
            (cirq.CCZ, True): (
                ReactionDynamics((local_z[0],), (local_z[1] * local_z[2],)),
                ReactionDynamics((local_z[1],), (local_z[0] * local_z[2],)),
                ReactionDynamics((local_z[2],), (local_z[0] * local_z[1],)),
            ),
        }
        if factory_reaction_dynamics is not None:
            self._factory_reaction_dynamics.update(
                {
                    key: tuple(factory_dynamics)
                    for key, factory_dynamics in factory_reaction_dynamics.items()
                }
            )

        unsupported_pairs = [
            (gate, auto_corrected)
            for gate, auto_corrected in self.factories.items()
            if (gate, auto_corrected) not in self._factory_reaction_dynamics
        ]
        if unsupported_pairs:
            raise ValueError(
                "No factory reaction dynamics are defined for: "
                + ", ".join(
                    f"({gate!r}, {auto_corrected!r})" for gate, auto_corrected in unsupported_pairs
                ),
            )

        for gate, auto_corrected in self.factories.items():
            local_qubits = tuple(cirq.LineQubit.range(cirq.num_qubits(gate)))
            for reaction_dynamics in self._factory_reaction_dynamics[(gate, auto_corrected)]:
                for pauli in (
                    *reaction_dynamics.dependency_paulis,
                    *reaction_dynamics.outputs,
                ):
                    if not set(pauli.qubits).issubset(local_qubits):
                        raise ValueError(
                            f"Reaction Pauli {pauli!r} must use only operation-local qubits "
                            f"{local_qubits} for factory gate {gate!r}."
                        )

    def reaction_depth(self, circuit: cirq.Circuit) -> int:
        """Compute the logical factory reaction depth for a circuit.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            Longest factory-vertex dependency chain. Circuits with no factory
            vertices have depth zero.
        """
        tracked_paulis: dict[cirq.PauliString, int] = {}
        reaction_depth = 0

        for operation in circuit.all_operations():
            factory_dynamics = self._factory_dynamics(operation)
            if factory_dynamics is not None:
                node_depths = tuple(
                    max(
                        (
                            source_depth
                            for source_pauli, source_depth in tracked_paulis.items()
                            if self._creates_reaction_dependency(
                                source_pauli, reaction_dynamics.dependency_paulis
                            )
                        ),
                        default=0,
                    )
                    + 1
                    for reaction_dynamics in factory_dynamics
                )
                reaction_depth = max((reaction_depth, *node_depths))

                for reaction_dynamics, node_depth in zip(
                    factory_dynamics, node_depths, strict=True
                ):
                    for output in reaction_dynamics.outputs:
                        phase_free_output = output.with_coefficient(1)
                        tracked_paulis[phase_free_output] = max(
                            tracked_paulis.get(phase_free_output, 0), node_depth
                        )
                continue

            propagated_paulis: dict[cirq.PauliString, int] = {}
            for pauli, depth in tracked_paulis.items():
                phase_free_pauli = self._propagate_pauli(pauli, operation).with_coefficient(1)
                propagated_paulis[phase_free_pauli] = max(
                    propagated_paulis.get(phase_free_pauli, 0), depth
                )
            tracked_paulis = propagated_paulis

        return reaction_depth

    def reaction_tree(self, circuit: cirq.Circuit) -> nx.DiGraph:
        """Build the transitively reduced factory-vertex dependency graph.

        Nodes are `(operation_index, vertex_index)` tuples. Edges point from a
        source factory vertex to a later factory vertex when a propagated source
        output anti-commutes with a target vertex dependency Pauli.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            NetworkX DAG with operation metadata in `graph["operations"]` and
            per-node longest-chain depths in node attribute `"depth"`.
        """
        operations = tuple(circuit.all_operations())
        reaction_tree = nx.DiGraph(operations=operations)
        factory_nodes: dict[int, tuple[ReactionTreeNode, ...]] = {}
        for operation_index, operation in enumerate(operations):
            self._add_reaction_node(
                operation_index,
                operation,
                reaction_tree,
                factory_nodes,
            )
        self._add_reaction_edges(operations, reaction_tree, factory_nodes)

        for depth, nodes in enumerate(nx.topological_generations(reaction_tree), start=1):
            for node in nodes:
                reaction_tree.nodes[node]["depth"] = depth

        return reaction_tree

    def _factory_dynamics(self, operation: cirq.Operation) -> tuple[ReactionDynamics, ...] | None:
        if operation.gate not in self.factories:
            if not cirq.has_stabilizer_effect(operation.gate):
                raise ValueError(self._NON_CLIFFORD_ERROR.format(operation=operation))
            return None

        local_qubit_map = dict(
            zip(cirq.LineQubit.range(len(operation.qubits)), operation.qubits, strict=True)
        )
        return tuple(
            ReactionDynamics(
                tuple(
                    pauli.map_qubits(local_qubit_map)
                    for pauli in reaction_dynamics.dependency_paulis
                ),
                tuple(output.map_qubits(local_qubit_map) for output in reaction_dynamics.outputs),
            )
            for reaction_dynamics in self._factory_reaction_dynamics[
                (operation.gate, self.factories[operation.gate])
            ]
        )

    def _add_reaction_node(
        self,
        operation_index: int,
        operation: cirq.Operation,
        reaction_tree: nx.DiGraph,
        factory_nodes: dict[int, tuple[ReactionTreeNode, ...]],
    ) -> None:
        factory_dynamics = self._factory_dynamics(operation)
        if factory_dynamics is None:
            return

        nodes: list[ReactionTreeNode] = []
        for vertex_index, reaction_dynamics in enumerate(factory_dynamics):
            node = (operation_index, vertex_index)
            nodes.append(node)
            reaction_tree.add_node(
                node,
                dependency_paulis=reaction_dynamics.dependency_paulis,
                outputs=reaction_dynamics.outputs,
            )
        factory_nodes[operation_index] = tuple(nodes)

    def _add_reaction_edges(
        self,
        operations: tuple[cirq.Operation, ...],
        reaction_tree: nx.DiGraph,
        factory_nodes: dict[int, tuple[ReactionTreeNode, ...]],
    ) -> None:
        node_masks = {node: 1 << index for index, node in enumerate(reaction_tree)}
        operation_masks = {
            operation_index: sum(node_masks[node] for node in nodes)
            for operation_index, nodes in factory_nodes.items()
        }
        closure_masks: dict[ReactionTreeNode, int] = {}
        future_mask = 0

        for source_operation_index in reversed(factory_nodes):
            for source_node in factory_nodes[source_operation_index]:
                covered_mask = 0
                propagated_paulis = reaction_tree.nodes[source_node]["outputs"]
                search_mask = future_mask if propagated_paulis else 0
                for target_operation_index in range(source_operation_index + 1, len(operations)):
                    if covered_mask == search_mask:
                        break

                    target_operation = operations[target_operation_index]
                    target_nodes = factory_nodes.get(target_operation_index)
                    if target_nodes is None:
                        propagated_paulis = tuple(
                            self._propagate_pauli(pauli, target_operation)
                            for pauli in propagated_paulis
                        )
                        continue

                    operation_mask = operation_masks[target_operation_index]
                    if covered_mask & operation_mask == operation_mask:
                        continue

                    for target_node in target_nodes:
                        target_mask = node_masks[target_node]
                        if covered_mask & target_mask:
                            continue
                        dependency_paulis = reaction_tree.nodes[target_node]["dependency_paulis"]
                        if any(
                            self._creates_reaction_dependency(propagated_pauli, dependency_paulis)
                            for propagated_pauli in propagated_paulis
                        ):
                            reaction_tree.add_edge(source_node, target_node)
                            covered_mask |= closure_masks[target_node]

                closure_masks[source_node] = node_masks[source_node] | covered_mask

            future_mask |= operation_masks[source_operation_index]

    @staticmethod
    def _creates_reaction_dependency(
        pauli: cirq.PauliString,
        dependency_paulis: tuple[cirq.PauliString, ...],
    ) -> bool:
        return any(
            not cirq.commutes(pauli, dependency_pauli) for dependency_pauli in dependency_paulis
        )

    @staticmethod
    def _propagate_pauli(
        pauli: cirq.PauliString,
        operation: cirq.Operation,
    ) -> cirq.PauliString:
        return (
            pauli.conjugated_by(operation)
            if any(qubit in pauli for qubit in operation.qubits)
            else pauli
        )
