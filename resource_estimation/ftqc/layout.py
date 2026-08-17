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

import resource_estimation.ftqc.codepatch as codepatch

PatchBuilder = typing.Callable[[int], codepatch.RotatedSurfaceCodePatch]


def _default_patch_builder(patch_id: int) -> codepatch.RotatedSurfaceCodePatch:
    return codepatch.RotatedSurfaceCodePatch(patch_id=patch_id, d=7)


@dataclass
class Layout(abc.ABC):
    """Base class for layouts used by the fault tolerant compiler to track factory use and CNOT routing"""

    input_circuit: cirq.Circuit
    num_t_factories: int = 0
    num_s_factories: int = 0
    num_ccz_factories: int = 0
    distil: bool = False

    def __post_init__(self) -> None:
        self.mapped_circuit = None
        self.layout_graph = None
        self._available_t_factories = collections.deque()
        self._available_s_factories = collections.deque()
        self._available_ccz_factories = collections.deque()
        self._all_factories = set()
        self._generate()

    def set_map_circuit(self, qubit_map: dict[cirq.Qid, cirq.Qid]) -> None:
        """Apply a mapping from input-circuit qubits to layout qubits used for compilation."""
        self.qubit_map = qubit_map
        mapped_circuit = cirq.Circuit(
            moment.transform_qubits(qubit_map) for moment in self.input_circuit
        )
        self.mapped_circuit = mapped_circuit

    def circuit_qubit(self, node: cirq.Qid | codepatch.CodePatch) -> cirq.Qid:
        """Return the Cirq wire associated with a layout-graph node."""
        if isinstance(node, cirq.Qid):
            return node
        if len(node.logical_qubits) != 1:
            raise ValueError("Layout patches must contain exactly one logical qubit.")
        return node.logical_qubits[0]

    def circuit_qubits(
        self, nodes: typing.Iterable[cirq.Qid | codepatch.CodePatch]
    ) -> tuple[cirq.Qid, ...]:
        """Return the Cirq wires associated with layout-graph nodes."""
        return tuple(self.circuit_qubit(node) for node in nodes)

    def position_of(self, node: cirq.Qid | codepatch.CodePatch) -> tuple[int, int]:
        """Return a layout node's row and column."""
        if isinstance(node, cirq.GridQubit):
            return node.row, node.col
        raise ValueError(f"No layout position found for {node!r}.")

    def distance(
        self,
        first: cirq.Qid | codepatch.CodePatch,
        second: cirq.Qid | codepatch.CodePatch,
    ) -> int:
        """Return the Manhattan distance between two layout objects.

        Legacy GridQubit layouts measure distance in grid hops; movement layouts override this
        method to measure across the physical dimensions of their patches.
        """
        first_row, first_col = self.position_of(first)
        second_row, second_col = self.position_of(second)
        return abs(first_row - second_row) + abs(first_col - second_col)

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

    @abc.abstractmethod
    def _generate(self) -> None:  # pragma: no cover
        """Generate the layout graph, circuit mapping, and patch placement."""

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

    def all_factories(self, ftype: typing.Literal["t", "s", "ccz"]):
        G = self.layout_graph

        def is_ftype_factory(node):
            return "ftype" in G.nodes[node] and G.nodes[node]["ftype"] == ftype

        unique_fids = np.unique(
            [G.nodes[node]["fid"] for node in G.nodes if is_ftype_factory(node)]
        )

        def has_fid(node, fid):
            return "fid" in G.nodes[node] and G.nodes[node]["fid"] == fid

        return [
            tuple(
                sorted(
                    (q for q in self._all_factories if has_fid(q, fid) and is_ftype_factory(q)),
                    key=self.position_of,
                )
            )
            for fid in unique_fids
        ]

    def nearest_factory(
        self,
        qubits: tuple[cirq.GridQubit, ...] | cirq.GridQubit,
        ftype: typing.Literal["s", "t", "ccz"],
    ) -> cirq.GridQubit | tuple[cirq.GridQubit, ...]:
        """Find the closest unused factory of the requested type.

        Distance comes from the positions and patch dimensions defined by the layout.
        Removes the returned factory from the available options and sets its status to `used`
        """
        single_qubit = isinstance(qubits, cirq.Qid)
        qubits = (qubits,) if single_qubit else qubits
        available_factories = self.available_factories(ftype=ftype)
        if not available_factories:
            raise ValueError(f"No {ftype} factories available!")

        def movement_heuristic(factory):
            """Heuristic based on the closest qubit within the factory by Manhattan distance"""
            # This replaces the legacy direct GridQubit calculation so each layout can define
            # whether its distance is measured in grid hops or physical patch dimensions.
            return min(self.distance(f, q) for q in qubits for f in factory)

        def lattice_heuristic(factory):
            """Heuristic based on the lattice surgery routing distance between the first qubit in the factory and the first qubit in the set of target qubits"""
            return len(self.route_cnot(factory[0], qubits[0]))

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
        closest_factory = closest_factory[0] if single_qubit else closest_factory
        return closest_factory

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit) -> list[cirq.GridQubit]:
        """Finds the patches required to perform a lattice surgery CNOT between two logical qubits
        The path returned must include at least one ancilla
        This method does not account for other CNOTs in the logical circuit, so choosing the shortest path might not correspond to the optimal path
        """
        # TODO: See if there is a way to maximize parallelism, or port over work that already does this maximization
        G = self.layout_graph

        def custom_weight(u: cirq.GridQubit, v: cirq.GridQubit, attr: dict) -> int | None:
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
            "ccz": "black",
            "data": "green",
            "ancilla": "blue",
            "block": "pink",
            "ccz": "orange",
        }
        G = self.layout_graph
        node_color = []
        for node in G.nodes:
            node_dict = G.nodes[node]
            key = node_dict["ftype"] if "ftype" in node_dict else node_dict["patch_type"]
            node_color.append(color_dict[key])
        pos = {node: self.position_of(node) for node in G.nodes}
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
        patch_builder: PatchBuilder = _default_patch_builder,
    ) -> None:
        self.patch_builder = patch_builder
        self.grid: dict[tuple[int, int], codepatch.RotatedSurfaceCodePatch] = {}
        self._patches_by_id: dict[int, codepatch.RotatedSurfaceCodePatch] = {}
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_ccz_factories=num_ccz_factories,
            num_s_factories=0,
        )

    def _make_patch(
        self, patch_id: int, position: tuple[int, int]
    ) -> codepatch.RotatedSurfaceCodePatch:
        patch = self.patch_builder(patch_id)
        if patch.patch_id != patch_id:
            raise ValueError(
                f"Patch builder returned patch_id {patch.patch_id}; expected {patch_id}."
            )
        if len(patch.logical_qubits) != 1:
            raise ValueError("Layout patches must contain exactly one logical qubit.")
        self.grid[position] = patch
        self._patches_by_id[patch_id] = patch
        return patch

    def _generate(self) -> None:
        """Place data and factory patches on an abstract movement grid."""
        total_patches = (
            len(self.input_circuit.all_qubits()) + self.num_s_factories + self.num_t_factories
        )
        side_length = ceil(sqrt(total_patches))

        def idx_to_position(idx: int) -> tuple[int, int]:
            return idx // side_length, idx % side_length

        self.grid = {}
        self._patches_by_id = {}
        G = nx.Graph()
        qubit_map: dict[cirq.Qid, cirq.Qid] = {}
        for patch_id, qid in enumerate(sorted(self.input_circuit.all_qubits())):
            position = idx_to_position(patch_id)
            patch = self._make_patch(patch_id, position)
            qubit_map[qid] = self.circuit_qubit(patch)
            G.add_node(patch, position=position, patch_type="data")
        self.set_map_circuit(qubit_map=qubit_map)

        first_factory_id = len(G.nodes)
        for factory_index in range(self.num_t_factories):
            patch_id = first_factory_id + factory_index
            position = idx_to_position(patch_id)
            patch = self._make_patch(patch_id, position)
            G.add_node(
                patch,
                position=position,
                patch_type="factory",
                ftype="t",
                fid=factory_index,
                used=True,
            )

        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    def patch_for(
        self, qubit_or_patch: cirq.Qid | codepatch.CodePatch
    ) -> codepatch.RotatedSurfaceCodePatch:
        """Return the patch associated with a movement-layout object."""
        if isinstance(qubit_or_patch, codepatch.CodePatch):
            return typing.cast(codepatch.RotatedSurfaceCodePatch, qubit_or_patch)
        if not isinstance(qubit_or_patch, codepatch.LogicalQubit):
            raise TypeError(f"No code patch found for {qubit_or_patch!r}.")
        return self._patches_by_id[qubit_or_patch.patch_id]

    def patch_at(self, position: tuple[int, int]) -> codepatch.RotatedSurfaceCodePatch:
        """Return the patch placed at a grid position."""
        return self.grid[position]

    @property
    def code_patches(self) -> tuple[codepatch.RotatedSurfaceCodePatch, ...]:
        """Return all code patches placed in the layout."""
        return tuple(self.grid.values())

    @property
    def num_physical_qubits(self) -> int:
        """Return the physical-qubit footprint of all patches in the layout."""
        return sum(patch.num_physical_qubits for patch in self.code_patches)

    def position_of(self, node: cirq.Qid | codepatch.CodePatch) -> tuple[int, int]:
        """Return the grid position of a patch or one of its logical qubits."""
        patch = self.patch_for(node)
        return typing.cast(tuple[int, int], self.layout_graph.nodes[patch]["position"])

    def distance(
        self,
        first: cirq.Qid | codepatch.CodePatch,
        second: cirq.Qid | codepatch.CodePatch,
    ) -> int:
        """Return physical Manhattan distance using the patches' dimensions."""
        first_patch = self.patch_for(first)
        second_patch = self.patch_for(second)
        first_row, first_col = self.position_of(first_patch)
        second_row, second_col = self.position_of(second_patch)
        # A movement-grid coordinate now represents a whole patch, so each grid step spans the
        # patch's interleaved physical-qubit dimension instead of one legacy GridQubit unit.
        return (
            abs(first_row - second_row) * first_patch.height
            + abs(first_col - second_col) * first_patch.width
        )

    def route_cnot(self, ctrl: cirq.Qid, trgt: cirq.Qid):
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

    # TODO: figure out a way o make the number of factories configurable
    def __init__(self, input_circuit: cirq.Circuit) -> None:
        # TODO: Find the formula for this
        super().__init__(input_circuit=input_circuit, num_s_factories=0, num_t_factories=0)

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
        patch_builder: PatchBuilder = _default_patch_builder,
    ) -> None:
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_ccz_factories=num_ccz_factories,
            patch_builder=patch_builder,
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

        def idx_to_position(idx: int) -> tuple[int, int]:
            return idx // side_length, idx % side_length

        self.grid = {}
        self._patches_by_id = {}
        G = nx.Graph()
        qubit_map: dict[cirq.Qid, cirq.Qid] = {}
        for patch_id, qid in enumerate(sorted(self.input_circuit.all_qubits())):
            position = idx_to_position(patch_id)
            patch = self._make_patch(patch_id, position)
            qubit_map[qid] = self.circuit_qubit(patch)
            G.add_node(patch, position=position, patch_type="data")
        self.set_map_circuit(qubit_map=qubit_map)

        # Add T Distillation Factories to Graph
        for factory_index in range(self.num_t_factories):
            qubit_index = factory_index * qubits_per_t_distil + program_qubits
            position = idx_to_position(qubit_index)
            output_patch = self._make_patch(qubit_index, position)
            G.add_node(
                output_patch,
                position=position,
                patch_type="factory",
                ftype="t",
                fid=factory_index,
                used=True,
            )
            for i in range(1, qubits_per_t_distil):
                patch_id = qubit_index + i
                position = idx_to_position(patch_id)
                patch = self._make_patch(patch_id, position)
                G.add_node(patch, position=position, patch_type="block", fid=factory_index)

        # Add CCZ Distillation Factories to Graph
        data_plus_t = program_qubits + (qubits_per_t_distil * self.num_t_factories)
        for factory_index in range(self.num_ccz_factories):  # just builds on to the T factories
            qubit_index = factory_index * qubits_per_ccz_distil + data_plus_t
            fid = self.num_t_factories + factory_index
            for i in range(num_output_qubits):
                patch_id = qubit_index + i
                position = idx_to_position(patch_id)
                patch = self._make_patch(patch_id, position)
                G.add_node(
                    patch,
                    position=position,
                    patch_type="factory",
                    ftype="ccz",
                    fid=fid,
                    used=True,
                )
            for i in range(num_output_qubits, qubits_per_ccz_distil):
                patch_id = qubit_index + i
                position = idx_to_position(patch_id)
                patch = self._make_patch(patch_id, position)
                G.add_node(patch, position=position, patch_type="block", fid=fid)

        # Movement layouts assume all-to-all connectivity; avoid storing O(n^2) edges explicitly.
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    def distillation_block(
        self, factory: tuple[codepatch.CodePatch, ...]
    ) -> list[codepatch.CodePatch]:
        G = self.layout_graph
        fid = G.nodes[factory[0]]["fid"]
        block_patches = [
            patch
            for patch in G.nodes
            if (G.nodes[patch]["patch_type"] == "block") and (G.nodes[patch]["fid"] == fid)
        ]
        return block_patches + list(factory)
