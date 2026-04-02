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
import abc
import cirq
import networkx as nx
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Literal
from math import ceil, sqrt
from itertools import combinations, product


@dataclass
class Layout(abc.ABC):
    """
    Base class for layouts used by the fault tolerant compiler to track factory use and CNOT routing
    """

    input_circuit: cirq.Circuit
    num_t_factories: int = 0
    num_s_factories: int = 0

    def __post_init__(self):
        self.mapped_circuit = None
        self.layout_graph = None
        self._available_t_factories = deque()
        self._available_s_factories = deque()
        self._all_factories = set()
        self._generate()

    def set_map_circuit(self, qubit_map: dict[cirq.Qid, cirq.GridQubit]) -> None:
        """
        Apply a given mapping from qubits in the input circuit to GridQubits used for compilation
        """
        mapped_circuit = cirq.Circuit(
            moment.transform_qubits(qubit_map) for moment in self.input_circuit
        )
        self.mapped_circuit = mapped_circuit

    def reset_graph(self) -> None:
        """
        Reset the graph to its starting state by setting all factory qubits to the `used` state
        """
        G = self.layout_graph
        for node in G.nodes:
            if G.nodes[node]["patch_type"] == "factory":
                G.nodes[node]["used"] = True
        # Resets the available factories
        self._available_t_factories = deque()
        self._available_s_factories = deque()

    def reload_factories(self, ftype: Literal["t", "s"]) -> None:
        if ftype == "t":
            all_t_factories = [
                factory
                for factory in self._all_factories
                if self.layout_graph.nodes[factory]["ftype"] == "t"
            ]
            self._available_t_factories = deque(all_t_factories)
        elif ftype == "s":
            all_s_factories = [
                factory
                for factory in self._all_factories
                if self.layout_graph.nodes[factory]["ftype"] == "s"
            ]
            self._available_s_factories = deque(all_s_factories)
        else:
            raise ValueError(f"{ftype} is not a valid factory type")
        # Update graph to reflect the new status
        for node in self.layout_graph.nodes:
            if node in self.available_s_factories or node in self._available_t_factories:
                self.layout_graph.nodes[node]["used"] = False

    def _generate(self) -> None:
        """
        Private method to generate the underlying networkx graph, qubit map, and qubit placement
        This method is the core of what defines a Layout
        At this level, the graph generated has no locality, but methods in subclasses should be local (especially lattice surgery layouts)
        """
        total_qubits = (
            len(self.input_circuit.all_qubits()) + self.num_s_factories + self.num_t_factories
        )
        side_length = ceil(sqrt(total_qubits))

        def idx_to_xy(idx: int) -> tuple[int, int]:
            x = idx // side_length
            y = idx % side_length
            return x, y

        qubit_map = {
            qid: cirq.GridQubit(*idx_to_xy(idx))
            for idx, qid in enumerate(sorted(self.input_circuit.all_qubits()))
        }
        self.set_map_circuit(qubit_map=qubit_map)
        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [
                (
                    cirq.GridQubit(*idx_to_xy(idx + len(G.nodes))),
                    dict(patch_type="factory", ftype="t", used=True),
                )
                for idx in range(self.num_t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (
                    cirq.GridQubit(*idx_to_xy(idx + len(G.nodes))),
                    dict(patch_type="factory", ftype="s", used=True),
                )
                for idx in range(self.num_s_factories)
            ],
        )
        G.add_edges_from((n1, n2) for n1, n2 in combinations(G.nodes, 2))
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    @property
    def available_t_factories(self) -> deque[cirq.GridQubit]:
        return self._available_t_factories

    @property
    def available_s_factories(self) -> deque[cirq.GridQubit]:
        return self._available_s_factories

    def nearest_factory(self, qubit: cirq.GridQubit, ftype: Literal["s", "t"]) -> cirq.GridQubit:
        """
        Finds the closest factory of desired type according to the Manhattan distance using the GridQubit indices of the factory qubits that do not have the `used` status
        Removes the returned factory from the available options and sets its status to `used`
        """
        available_factories = (
            self.available_s_factories if ftype == "s" else self.available_t_factories
        )
        if not available_factories:
            raise ValueError(f"No available {ftype} factories available!")
        r, c = qubit.row, qubit.col
        # Closest factory according to the L1 distance
        factory = min(available_factories, key=lambda fact: abs(fact.row - r) + abs(fact.col - c))
        # Factory now used must be removed
        self.layout_graph.nodes[factory]["used"] = True
        available_factories.remove(factory)
        if ftype == "s":
            self._available_s_factories = available_factories
        else:
            self._available_t_factories = available_factories
        return factory

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit) -> list[cirq.GridQubit]:
        """
        Finds the patches required to perform a lattice surgery CNOT between two logical qubits
        The path returned must include at least one ancilla
        This method does not account for other CNOTs in the logical circuit, so choosing the shortest path might not correspond to the optimal path
        """
        # TODO: See if there is a way to maximize parallelism, or port over work that already does this maximization
        G = self.layout_graph

        def custom_weight(u: cirq.GridQubit, v: cirq.GridQubit, attr: dict) -> int | None:
            if (G.nodes[v]["patch_type"] == "data") or (G.nodes[v]["patch_type"] == "factory"):
                # Must go through at least one ancilla
                if (v == trgt and u == ctrl) or (u == trgt and v == ctrl):
                    return None
                elif v == trgt or v == ctrl:
                    return 1
                else:
                    return None
            return 1

        path = nx.dijkstra_path(G=G, source=ctrl, target=trgt, weight=custom_weight)
        return path

    def draw(self) -> None:  # pragma: no cover
        """
        Draw method to display layouts clearly
        Red and yellow nodes correspond to T and S factories, respectively
        Green nodes correspond to data (logical) qubits
        Blue nodes correspond to ancilla qubits
        """
        color_dict = {
            "t": "red",
            "s": "yellow",
            "data": "green",
            "ancilla": "blue",
        }
        G = self.layout_graph
        node_color = []
        for node in G.nodes:
            node_dict = G.nodes[node]
            key = node_dict["ftype"] if "ftype" in node_dict else node_dict["patch_type"]
            node_color.append(color_dict[key])
        pos = {node: (node.row, node.col) for node in G.nodes}
        nx.draw(G, with_labels=True, node_color=node_color, pos=pos)


class MovementLayout(Layout):
    """
    Layout class representing the connections available to Movement Architectures
    It does not have S factories and the number of T factories is fully configurable
    The current implementation assumes all-to-all connectivity in the logical qubit layout because the cost for nonlocal moves is handled deeper in the stack
    A better implementation might do a smart placement of qubits on the grid to minimize overall distance travelled
    """

    # TODO: build this implementation
    def __init__(self, input_circuit: cirq.Circuit, num_t_factories: int = 1):
        super().__init__(
            input_circuit=input_circuit, num_t_factories=num_t_factories, num_s_factories=0
        )

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit):
        raise NotImplementedError


class Column(Layout):
    """
    Lattice surgery Layout based on having two columns of logical qubits
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    ...
    """

    def __init__(self, input_circuit: cirq.Circuit):
        rows = ceil(len(input_circuit.all_qubits()) / 2)
        num_s_factories = 2 * rows
        num_t_factories = 2 * rows
        super().__init__(
            input_circuit=input_circuit,
            num_s_factories=num_s_factories,
            num_t_factories=num_t_factories,
        )

    def _generate(self) -> None:
        """
        Places and assigns logical qubits according to the column configuration
        In the case where the number of logical qubits is odd fill the would-be logical qubit with an ancilla
        """
        qubit_map: dict[cirq.Qid, cirq.GridQubit] = {}
        all_qubits = list(self.input_circuit.all_qubits())
        s_factories = []
        t_factories = []
        ancillas = []
        num_rows = ceil(len(all_qubits) / 2)
        for idx, qid in enumerate(sorted(all_qubits)):
            row = 2 * (idx // 2)
            col = 4 if idx % 2 else 2
            qubit_map[qid] = cirq.GridQubit(row, col)
        self.set_map_circuit(qubit_map=qubit_map)
        for row in range(2 * num_rows):
            if row % 2 == 0:
                s_factories.extend([cirq.GridQubit(row, 0), cirq.GridQubit(row, 6)])
                ancillas.extend(
                    [cirq.GridQubit(row, 1), cirq.GridQubit(row, 3), cirq.GridQubit(row, 5)]
                )
            else:
                t_factories.extend([cirq.GridQubit(row, 0), cirq.GridQubit(row, 6)])
                ancillas.extend([cirq.GridQubit(row, col) for col in range(1, 6)])
        if len(all_qubits) % 2:
            ancillas.append(cirq.GridQubit(2 * num_rows - 2, 4))

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        # Connect nearest neighbors (Manhattan distance 1) without O(n^2) pairwise checks
        for node in G.nodes:
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = cirq.GridQubit(node.row + d_row, node.col + d_col)
                if neighbor in G:
                    G.add_edge(node, neighbor)
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G


class FactorySandwich(Layout):
    """
    Lattice surgery layout based on having a line of logical qubits sandwiched by factory qubits and ancilla
    S | S | ... | S
    a | a | ... | a
    q | q | ... | q
    a | a | ... | a
    T | T | ... | T

    Because the numbers of S and T factories are configurable, the dimensions might not line up resulting in things like
    S | S | S
    a | a | a | a | a
    q | q | q | q | q
    a | a | a | a | a
    T | T | T | T
    """

    def _generate(self):
        """
        Places and assigns logical qubits according to the Sandwich configuration
        """
        qubit_map: dict[cirq.Qid, cirq.GridQubit] = {}
        all_qubits = list(self.input_circuit.all_qubits())
        length = max(len(all_qubits), self.num_t_factories, self.num_s_factories)
        s_factories = []
        t_factories = []
        ancillas = []
        for idx, qid in enumerate(sorted(all_qubits)):
            qubit_map[qid] = cirq.GridQubit(2, idx)
        self.set_map_circuit(qubit_map=qubit_map)
        ancillas = [cirq.GridQubit(row, idx) for idx in range(length) for row in (1, 3)]
        s_factories = [cirq.GridQubit(0, idx) for idx in range(self.num_s_factories)]
        t_factories = [cirq.GridQubit(4, idx) for idx in range(self.num_t_factories)]

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ]
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G


class Embedded(Layout):
    """
    Lattice surgery layout based on packing logical qubits into a rectangle with ancilla patches forming gaps between them
    Without the ancilla patches, the logical qubits would be nearest neighbor
    Factories surround the main array, alternating between S and T designation
    This Layout currently cannot increase/decrease the number of factories of either type
    The inspiration for this layout was a conversation with Ben, where he described the output of the MCM compiler being nearest-neighbor connectivity
    So I wanted a Layout that could potentially be compatible with that kind of output
    """

    # TODO: figure out a way o make the number of factories configurable
    def __init__(self, input_circuit: cirq.Circuit):
        # TODO: Find the formula for this
        super().__init__(input_circuit=input_circuit, num_s_factories=0, num_t_factories=0)

    def _generate(self):
        """
        Builds a large embedded logical qubit array by starting from a nearest neighbor array and adding rows/columns of other qubit types
        """
        all_qubits = list(self.input_circuit.all_qubits())
        num_logicals = len(all_qubits)
        side_length = ceil(sqrt(num_logicals))
        filler = side_length**2 - num_logicals

        # Build a mini array that packs the logical qubits as tightly as possible in a rectangle
        # Any leftover space in the rectangle is designated as ancilla space
        stage1 = np.array([1] * num_logicals + [0] * filler).reshape((side_length, side_length))
        stage1 = np.array([row for row in stage1 if not all(row == 0)])
        stage1 = np.array([col for col in stage1.T if not all(col == 0)]).T

        # Add ancilla space between logical qubits
        stage2 = [[0] * stage1.shape[1]]
        for row in stage1:
            stage2.append(row.tolist())
            stage2.append([0] * len(row))
        stage2 = np.array(stage2)

        stage3 = [[0] * stage2.shape[0]]
        for col in stage2.T:
            stage3.append(col.tolist())
            stage3.append([0] * len(col))
        stage3 = np.array(stage3).T

        # Wrap the resulting array in factory qubits
        factory_row = np.array([2 if i % 2 else 3 for i in range(stage3.shape[1])])
        stage4 = np.vstack((factory_row, stage3, factory_row))

        factory_col = np.array([[0] + [2 if i % 2 else 3 for i in range(stage3.shape[0])] + [0]]).T
        stage5 = np.hstack((factory_col, stage4, factory_col))
        total_rows, total_cols = stage5.shape

        # Now convert that array into logical qubits, factories, and ancilla in the qubit map and layout graph
        logical_qubit_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 1
        ]
        ancilla_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 0
        ]
        # We also trim off the corners to avoid adding useless ancilla patches
        for i, j in product([0, total_rows - 1], (0, total_cols - 1)):
            ancilla_positions.remove((i, j))
        s_factory_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 2
        ]
        t_factory_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 3
        ]
        qubit_map = {
            qid: cirq.GridQubit(row, col)
            for qid, (row, col) in zip(sorted(all_qubits), logical_qubit_positions)
        }
        self.set_map_circuit(qubit_map=qubit_map)
        ancillas = [cirq.GridQubit(row, col) for row, col in ancilla_positions]
        s_factories = [cirq.GridQubit(row, col) for row, col in s_factory_positions]
        t_factories = [cirq.GridQubit(row, col) for row, col in t_factory_positions]

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ]
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G
        self.num_s_factories = len(s_factories)
        self.num_t_factories = len(t_factories)
