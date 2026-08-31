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

import abc
import collections
import itertools
import typing
from dataclasses import dataclass
from math import ceil, sqrt

import cirq
import networkx as nx
import numpy as np
import numpy.typing as npt


@dataclass
class Layout(abc.ABC):
    """Base class for layouts used by the fault tolerant compiler to track factory use and CNOT routing"""

    input_circuit: cirq.Circuit
    architecture: typing.Literal["SSM", "MZO", "DSM", "DSNM"]
    num_t_factories: int = 0
    num_s_factories: int = 0
    num_ccz_factories: int = 0
    distil: bool = False
    site_spacing: int = 4

    def __post_init__(self) -> None:
        self.mapped_circuit: cirq.Circuit = cirq.Circuit()

        self.inplace_cnot = self.architecture in ("MZO", "DSM")
        self.measure_zones = self.architecture in ("MZO", "SSM")
        self.interaction_zones = self.architecture in ("SSM")

        self.layout_graph: nx.Graph = nx.Graph()
        self._available_t_factories: collections.deque[tuple[cirq.GridQubit, ...]] = (
            collections.deque()
        )
        self._available_s_factories: collections.deque[tuple[cirq.GridQubit, ...]] = (
            collections.deque()
        )
        self._available_ccz_factories: collections.deque[tuple[cirq.GridQubit, ...]] = (
            collections.deque()
        )
        self._all_factories: set[cirq.GridQubit] = set()
        self._generate()

    def set_map_circuit(self, qubit_map: dict[cirq.Qid, cirq.GridQubit]) -> None:
        """Apply a given mapping from qubits in the input circuit to GridQubits used for compilation"""
        # Ignoring the type is ok because cirq doesn't recognize the desired type as a subclass of dict[Qid, Qid]
        mapped_circuit = cirq.Circuit(
            moment.transform_qubits(qubit_map)  # type: ignore[arg-type]
            for moment in self.input_circuit
        )
        self.mapped_circuit = mapped_circuit

    def reset_graph(self) -> None:
        """Reset the graph to its starting state by setting all factory qubits to the `used` state"""
        G = self.layout_graph
        for node in G.nodes:
            if G.nodes[node]["patch_type"] == "factory":
                G.nodes[node]["used"] = True
        # Resets the available factories
        self._available_t_factories = collections.deque()
        self._available_s_factories = collections.deque()
        self._available_ccz_factories = collections.deque()

    def reload_factories(self, ftype: typing.Literal["t", "s", "ccz"]) -> None:
        if ftype not in ["t", "s", "ccz"]:
            raise ValueError(f"{ftype} is not a valid factory type")

        factories = self.all_factories(ftype=ftype)
        if ftype == "t":
            self._available_t_factories = collections.deque(factories)
        elif ftype == "s":
            self._available_s_factories = collections.deque(factories)
        elif ftype == "ccz":
            self._available_ccz_factories = collections.deque(factories)
        # Update graph to reflect the new status
        for factory in factories:
            for node in factory:
                self.layout_graph.nodes[node]["used"] = False

    def _generate(self) -> None:
        """Private method to generate the underlying networkx graph, qubit map, and qubit placement
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
                    dict(patch_type="factory", ftype="t", fid=idx, used=True),
                )
                for idx in range(self.num_t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (
                    cirq.GridQubit(*idx_to_xy(idx + len(G.nodes))),
                    dict(patch_type="factory", ftype="s", fid=idx, used=True),
                )
                for idx in range(self.num_s_factories)
            ],
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    def available_factories(
        self, ftype: typing.Literal["t", "s", "ccz"]
    ) -> collections.deque[tuple[cirq.GridQubit, ...]]:
        if ftype == "t":
            return self._available_t_factories
        elif ftype == "s":
            return self._available_s_factories
        elif ftype == "ccz":
            return self._available_ccz_factories
        raise ValueError(f"No factories available with type {ftype}")

    def all_factories(
        self, ftype: typing.Literal["t", "s", "ccz"]
    ) -> list[tuple[cirq.GridQubit, ...]]:
        G = self.layout_graph

        def is_ftype_factory(node: cirq.GridQubit) -> bool:
            return "ftype" in G.nodes[node] and G.nodes[node]["ftype"] == ftype

        unique_fids = np.unique(
            [G.nodes[node]["fid"] for node in G.nodes if is_ftype_factory(node)]
        )

        def has_fid(node: cirq.GridQubit, fid: int) -> bool:
            return "fid" in G.nodes[node] and G.nodes[node]["fid"] == fid

        return [
            tuple(
                sorted(
                    (q for q in self._all_factories if has_fid(q, fid) and is_ftype_factory(q)),
                    key=lambda q: (q.row, q.col),
                )
            )
            for fid in unique_fids
        ]

    def nearest_factory(
        self,
        qubits: tuple[cirq.GridQubit, ...] | cirq.GridQubit,
        ftype: typing.Literal["s", "t", "ccz"],
    ) -> tuple[cirq.GridQubit, ...]:
        """Finds the closest factory of desired type according to the Manhattan distance using the GridQubit indices of the factory qubits that do not have the `used` status
        Removes the returned factory from the available options and sets its status to `used`
        """
        target_qubits: tuple[cirq.GridQubit, ...] = (
            (qubits,) if isinstance(qubits, cirq.GridQubit) else qubits
        )

        available_factories = self.available_factories(ftype=ftype)
        if not available_factories:
            raise ValueError(f"No {ftype} factories available!")

        def movement_heuristic(factory: tuple[cirq.GridQubit, ...]) -> int:
            """Heuristic based on the closest qubit within the factory by Manhattan distance"""
            return min(
                abs(f.row - q.row) + abs(f.col - q.col) for q in target_qubits for f in factory
            )

        def lattice_heuristic(factory: tuple[cirq.GridQubit, ...]) -> int:
            """Heuristic based on the lattice surgery routing distance between the first qubit in the factory and the first qubit in the set of target qubits"""
            return len(self.route_cnot(factory[0], target_qubits[0]))

        factories = self.available_factories(ftype=ftype)
        try:
            closest_factory = min(factories, key=lattice_heuristic)
        except NotImplementedError:
            closest_factory = min(factories, key=movement_heuristic)

        # Factory now used must be removed
        for factory_qubit in closest_factory:
            self.layout_graph.nodes[factory_qubit]["used"] = True
        available_factories.remove(closest_factory)
        if ftype == "s":
            self._available_s_factories = available_factories
        elif ftype == "t":
            self._available_t_factories = available_factories
        else:
            self._available_ccz_factories = available_factories
        return closest_factory

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit) -> list[cirq.GridQubit]:
        """Finds the patches required to perform a lattice surgery CNOT between two logical qubits
        The path returned must include at least one ancilla
        This method does not account for other CNOTs in the logical circuit, so choosing the shortest path might not correspond to the optimal path
        """
        # TODO: See if there is a way to maximize parallelism, or port over work that already does this maximization
        G = self.layout_graph

        def custom_weight(
            u: cirq.GridQubit, v: cirq.GridQubit, attr: dict[str, object]
        ) -> int | None:
            # First condition not covered because Distillation has not been implemented for lattice layouts
            if (
                G.nodes[v]["patch_type"] == "block" or G.nodes[u]["patch_type"] == "block"
            ):  # pragma: no cover
                return None
            if (G.nodes[v]["patch_type"] == "data") or (G.nodes[v]["patch_type"] == "factory"):
                # Must go through at least one ancilla
                if (v == trgt and u == ctrl) or (u == trgt and v == ctrl):
                    return None
                if v == trgt or v == ctrl:
                    return 1
                return None
            return 1

        path = nx.dijkstra_path(G=G, source=ctrl, target=trgt, weight=custom_weight)
        return path

    def draw(self) -> None:  # pragma: no cover
        """Draw method to display layouts clearly
        Red and yellow nodes correspond to T and S factories, respectively
        Green nodes correspond to data (logical) qubits
        Blue nodes correspond to ancilla qubits
        """
        color_dict = {
            "t": "red",
            "s": "yellow",
            "data": "green",
            "ancilla": "blue",
            "block": "pink",
            "ccz": "orange",
            "mzone": "gray",
            "izone": "lightgray",
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
    """Layout class representing the connections available to Movement Architectures
    It does not have S factories and the number of T factories is fully configurable
    The current implementation assumes all-to-all connectivity in the logical qubit layout because the cost for nonlocal moves is handled deeper in the stack
    A better implementation might do a smart placement of qubits on the grid to minimize overall distance travelled
    """

    # TODO: build this implementation
    def __init__(
        self,
        input_circuit: cirq.Circuit,
        num_t_factories: int = 1,
        num_ccz_factories: int = 0,
        architecture: typing.Literal["SSM", "MZO", "DSM"] = "SSM",
    ) -> None:
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_ccz_factories=num_ccz_factories,
            num_s_factories=0,
            architecture=architecture,
        )

    def _add_zones(self) -> None:
        G = self.layout_graph
        cols = max(node.col for node in G.nodes) + 1
        rows = max(node.row for node in G.nodes) + 1
        if self.interaction_zones:  # Place an interaction zone in the -1 row
            G.add_nodes_from(
                [
                    (
                        cirq.GridQubit(-1, col),
                        dict(patch_type="izone"),
                    )
                    for col in range(cols)
                ],
            )
        if self.measure_zones:  # Place a measurement zone in the final row
            G.add_nodes_from(
                [
                    (
                        cirq.GridQubit(rows, col),
                        dict(patch_type="mzone"),
                    )
                    for col in range(cols)
                ],
            )
        self.layout_graph = G

    def zone_qubits(self, zone_type: typing.Literal["measure", "interact"]) -> list[cirq.GridQubit]:
        if zone_type == "measure":
            return [
                node
                for node in self.layout_graph.nodes
                if self.layout_graph.nodes[node]["patch_type"] == "mzone"
            ]
        if zone_type == "interact":
            return [
                node
                for node in self.layout_graph.nodes
                if self.layout_graph.nodes[node]["patch_type"] == "izone"
            ]
        else:
            raise ValueError(f"Not a recognized zone type: {zone_type}")

    def __post_init__(self) -> None:
        super().__post_init__()
        self._add_zones()

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit) -> list[cirq.GridQubit]:
        raise NotImplementedError


class Column(Layout):
    """Lattice surgery Layout based on having two columns of logical qubits
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    ...
    """

    def __init__(self, input_circuit: cirq.Circuit) -> None:
        rows = ceil(len(input_circuit.all_qubits()) / 2)
        num_s_factories = 2 * rows
        num_t_factories = 2 * rows
        super().__init__(
            input_circuit=input_circuit,
            num_s_factories=num_s_factories,
            num_t_factories=num_t_factories,
            architecture="DSNM",
        )

    def _generate(self) -> None:
        """Places and assigns logical qubits according to the column configuration
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
                    [cirq.GridQubit(row, 1), cirq.GridQubit(row, 3), cirq.GridQubit(row, 5)],
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
            [
                (q, dict(patch_type="factory", ftype="t", fid=i, used=True))
                for i, q in enumerate(t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (q, dict(patch_type="factory", ftype="s", fid=i, used=True))
                for i, q in enumerate(s_factories)
            ],
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
    """Lattice surgery layout based on having a line of logical qubits sandwiched by factory qubits and ancilla
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

    def __init__(
        self, input_circuit: cirq.Circuit, num_t_factories: int = 0, num_s_factories: int = 0
    ) -> None:
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_s_factories=num_s_factories,
            architecture="DSNM",
        )

    def _generate(self) -> None:
        """Places and assigns logical qubits according to the Sandwich configuration"""
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
            [
                (q, dict(patch_type="factory", ftype="t", fid=i, used=True))
                for i, q in enumerate(t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (q, dict(patch_type="factory", ftype="s", fid=i, used=True))
                for i, q in enumerate(s_factories)
            ],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in itertools.combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ],
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G


class Embedded(Layout):
    """Lattice surgery layout based on packing logical qubits into a rectangle with ancilla patches forming gaps between them
    Without the ancilla patches, the logical qubits would be nearest neighbor
    Factories surround the main array, alternating between S and T designation
    This Layout currently cannot increase/decrease the number of factories of either type
    The inspiration for this layout was a conversation with Ben, where he described the output of the MCM compiler being nearest-neighbor connectivity
    So I wanted a Layout that could potentially be compatible with that kind of output
    """

    def __init__(self, input_circuit: cirq.Circuit) -> None:
        super().__init__(input_circuit=input_circuit, architecture="DSNM")

    def _generate(self) -> None:
        """Builds a large embedded logical qubit array by starting from a nearest neighbor array and adding rows/columns of other qubit types"""
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
        stage2_rows: list[list[int]] = [[0] * stage1.shape[1]]
        for row in stage1:
            stage2_rows.append(row.tolist())
            stage2_rows.append([0] * len(row))

        stage2: npt.NDArray[np.int_] = np.array(stage2_rows)

        stage3_cols: list[list[int]] = [[0] * stage2.shape[0]]
        for col in stage2.T:
            stage3_cols.append(col.tolist())
            stage3_cols.append([0] * len(col))

        stage3: npt.NDArray[np.int_] = np.array(stage3_cols).T

        # Wrap the resulting array in factory qubits
        factory_row = np.array([2 if i % 2 else 3 for i in range(stage3.shape[1])])
        stage4 = np.vstack((factory_row, stage3, factory_row))

        factory_col = np.array([[0] + [2 if i % 2 else 3 for i in range(stage3.shape[0])] + [0]]).T
        stage5 = np.hstack((factory_col, stage4, factory_col))
        total_rows, total_cols = stage5.shape

        # Now convert that array into logical qubits, factories, and ancilla in the qubit map and layout graph
        logical_qubit_positions = [
            (i, j)
            for i, j in itertools.product(range(total_rows), range(total_cols))
            if stage5[i, j] == 1
        ]
        ancilla_positions = [
            (i, j)
            for i, j in itertools.product(range(total_rows), range(total_cols))
            if stage5[i, j] == 0
        ]
        # We also trim off the corners to avoid adding useless ancilla patches
        for i, j in itertools.product([0, total_rows - 1], (0, total_cols - 1)):
            ancilla_positions.remove((i, j))
        s_factory_positions = [
            (i, j)
            for i, j in itertools.product(range(total_rows), range(total_cols))
            if stage5[i, j] == 2
        ]
        t_factory_positions = [
            (i, j)
            for i, j in itertools.product(range(total_rows), range(total_cols))
            if stage5[i, j] == 3
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
            [
                (q, dict(patch_type="factory", ftype="t", fid=i, used=True))
                for i, q in enumerate(t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (q, dict(patch_type="factory", ftype="s", fid=i, used=True))
                for i, q in enumerate(s_factories)
            ],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in itertools.combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ],
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G
        self.num_s_factories = len(s_factories)
        self.num_t_factories = len(t_factories)


class MovementDistillery(MovementLayout):
    """
    Layout for distilling magic states using movement.  Currently handles
    T and CCZ distillation layouts.
    """

    def __init__(
        self,
        input_circuit: cirq.Circuit,
        num_t_factories: int = 0,
        num_ccz_factories: int = 0,
        architecture: typing.Literal["SSM", "MZO", "DSM"] = "SSM",
    ) -> None:
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_ccz_factories=num_ccz_factories,
            architecture=architecture,
        )
        self.distil = True

    def _generate(self) -> None:
        # Establish Important Variables
        program_qubits = len(self.input_circuit.all_qubits())
        qubits_per_t_distil = 31
        qubits_per_ccz_distil = 23
        num_output_qubits = 3
        distillation_qubits = (
            qubits_per_ccz_distil * self.num_ccz_factories
            + qubits_per_t_distil * self.num_t_factories
        )
        total_qubits = program_qubits + distillation_qubits
        side_length = ceil(sqrt(total_qubits))

        # Maps linear indices to left-r>ight + up->down grid filling
        def idx_to_xy(idx: int) -> tuple[int, int]:
            x = idx // side_length
            y = idx % side_length
            return x, y

        # Map Program Qubits to GridQubits
        qubit_map = {
            qid: cirq.GridQubit(*idx_to_xy(idx))
            for idx, qid in enumerate(sorted(self.input_circuit.all_qubits()))
        }
        self.set_map_circuit(qubit_map=qubit_map)

        # Generate Layout Graph
        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        # Add T Distillation Factories to Graph
        for factory_index in range(self.num_t_factories):
            qubit_index = factory_index * qubits_per_t_distil + program_qubits
            output_qubit = cirq.GridQubit(*idx_to_xy(qubit_index))
            G.add_node(output_qubit, patch_type="factory", ftype="t", fid=factory_index, used=True)
            block_qubits = [
                cirq.GridQubit(*idx_to_xy(qubit_index + i)) for i in range(1, qubits_per_t_distil)
            ]
            G.add_nodes_from(
                [(q, dict(patch_type="block", fid=factory_index)) for q in block_qubits],
            )
        # Add CCZ Distillation Factories to Graph
        data_plus_t = program_qubits + (qubits_per_t_distil * self.num_t_factories)
        for factory_index in range(self.num_ccz_factories):  # just builds on to the T factories
            qubit_index = factory_index * qubits_per_ccz_distil + data_plus_t
            output_qubits = [
                cirq.GridQubit(*idx_to_xy(qubit_index + i)) for i in range(num_output_qubits)
            ]
            G.add_nodes_from(
                [
                    (
                        q,
                        dict(
                            patch_type="factory",
                            ftype="ccz",
                            fid=self.num_t_factories + factory_index,
                            used=True,
                        ),
                    )
                    for q in output_qubits
                ]
            )
            block_qubits = [
                cirq.GridQubit(*idx_to_xy(qubit_index + i))
                for i in range(num_output_qubits, qubits_per_ccz_distil)
            ]
            G.add_nodes_from(
                [
                    (q, dict(patch_type="block", fid=(self.num_t_factories + factory_index)))
                    for q in block_qubits
                ]
            )
        # Movement layouts assume all-to-all connectivity; avoid storing O(n^2) edges explicitly.
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    def distillation_block(self, factory: tuple[cirq.GridQubit, ...]) -> list[cirq.GridQubit]:
        G = self.layout_graph
        fid = G.nodes[factory[0]]["fid"]
        block_qubits = [
            q
            for q in G.nodes
            if (G.nodes[q]["patch_type"] == "block") and (G.nodes[q]["fid"] == fid)
        ]
        return block_qubits + list(factory)
