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
import functools

import cirq
import pytest

import resource_estimation.ftqc.codepatch as codepatch
from resource_estimation.ftqc.layout import (
    Column,
    Embedded,
    FactorySandwich,
    MovementDistillery,
    MovementLayout,
)


@pytest.fixture
def circuit5() -> cirq.Circuit:
    circuit = cirq.testing.random_circuit(
        cirq.LineQubit.range(5),
        10,
        0.6,
        {cirq.T: 1, cirq.S: 1, cirq.CNOT: 2, cirq.H: 1, cirq.CCZ: 3},
        42,
    )
    return circuit


def test_column(circuit5: cirq.Circuit) -> None:
    column = Column(circuit5)
    column.reload_factories(ftype="s")
    column.reload_factories(ftype="t")
    assert column.nearest_factory(qubits=cirq.GridQubit(0, 2), ftype="s") == cirq.GridQubit(0, 0)
    assert column.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t") in [
        cirq.GridQubit(3, 0),
        cirq.GridQubit(1, 0),
    ]
    assert column.nearest_factory(qubits=cirq.GridQubit(2, 4), ftype="t") in [
        cirq.GridQubit(1, 6),
        cirq.GridQubit(3, 6),
    ]
    # Now that (0, 0) is used, the nearest S factory to lq (0, 2) is (2, 0)
    assert column.nearest_factory(qubits=cirq.GridQubit(0, 2), ftype="s") == cirq.GridQubit(2, 0)
    G = column.layout_graph
    # Total number of nodes should be 7 x 6 = 42
    assert len(G.nodes) == 42
    # Of those 42 nodes, 5 should be logical qubits, 25 ancillas, and 6 of each factory type
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 25
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
        )
        == 6
    )
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
        )
        == 6
    )
    # Confirm the expected routes for a couple of qubits
    ctrl, trgt = cirq.GridQubit(2, 2), cirq.GridQubit(5, 0)
    # Both of these paths are valid
    path_a = [
        ctrl,
        cirq.GridQubit(2, 1),
        cirq.GridQubit(3, 1),
        cirq.GridQubit(4, 1),
        cirq.GridQubit(5, 1),
        trgt,
    ]
    path_b = [
        ctrl,
        cirq.GridQubit(3, 2),
        cirq.GridQubit(3, 1),
        cirq.GridQubit(4, 1),
        cirq.GridQubit(5, 1),
        trgt,
    ]
    assert column.route_cnot(ctrl=ctrl, trgt=trgt) in [path_a, path_b]
    ctrl, trgt = cirq.GridQubit(0, 4), cirq.GridQubit(5, 6)
    path_a = [
        ctrl,
        cirq.GridQubit(0, 5),
        cirq.GridQubit(1, 5),
        cirq.GridQubit(2, 5),
        cirq.GridQubit(3, 5),
        cirq.GridQubit(4, 5),
        cirq.GridQubit(5, 5),
        trgt,
    ]
    path_b = [
        ctrl,
        cirq.GridQubit(1, 4),
        cirq.GridQubit(1, 5),
        cirq.GridQubit(2, 5),
        cirq.GridQubit(3, 5),
        cirq.GridQubit(4, 5),
        cirq.GridQubit(5, 5),
        trgt,
    ]
    assert column.route_cnot(ctrl=ctrl, trgt=trgt) in [path_a, path_b]


def test_sandwich(circuit5: cirq.Circuit) -> None:
    sandwich = FactorySandwich(circuit5, num_t_factories=3, num_s_factories=5)
    sandwich.reload_factories(ftype="s")
    sandwich.reload_factories(ftype="t")
    # Check that nearest T factory is as expected and changes when used
    assert sandwich.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t") == cirq.GridQubit(4, 2)
    assert sandwich.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t") == cirq.GridQubit(4, 1)
    assert sandwich.nearest_factory(qubits=cirq.GridQubit(2, 4), ftype="s") == cirq.GridQubit(0, 4)
    # Check that there are no unexpected nodes in the layout graph
    G = sandwich.layout_graph
    assert len(G.nodes) == 23
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 10
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
        )
        == 5
    )
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
        )
        == 3
    )
    # Check that a CNOT has a reasonable path
    ctrl, trgt = cirq.GridQubit(2, 0), cirq.GridQubit(0, 4)
    expected_path = [
        ctrl,
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(1, 3),
        cirq.GridQubit(1, 4),
        trgt,
    ]
    assert sandwich.route_cnot(ctrl=ctrl, trgt=trgt) == expected_path

    sandwich.route_cnot(
        ctrl=cirq.GridQubit(2, 1),
        trgt=cirq.GridQubit(2, 2),
    )  # Hopefully this covers 116?


def test_embedded(circuit5: cirq.Circuit) -> None:
    embedded = Embedded(circuit5)
    embedded.reload_factories(ftype="s")
    embedded.reload_factories(ftype="t")
    # Check available qubits across several situations of ambiguity
    assert embedded.nearest_factory(cirq.GridQubit(2, 6), ftype="t") in [
        cirq.GridQubit(0, 5),
        cirq.GridQubit(0, 7),
        cirq.GridQubit(1, 8),
        cirq.GridQubit(3, 8),
    ]
    assert embedded.nearest_factory(cirq.GridQubit(4, 4), ftype="s") == cirq.GridQubit(6, 4)
    assert embedded.nearest_factory(cirq.GridQubit(4, 2), ftype="s") in [
        cirq.GridQubit(4, 0),
        cirq.GridQubit(6, 2),
    ]
    # Check that there are no unexpected nodes in the layout graph
    G = embedded.layout_graph
    assert len(G.nodes) == 59
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 30
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
        )
        == 10
    )
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
        )
        == 14
    )
    # Check that a CNOT has a reasonable path
    ctrl, trgt = cirq.GridQubit(4, 4), cirq.GridQubit(4, 8)
    expected_path = [ctrl, cirq.GridQubit(4, 5), cirq.GridQubit(4, 6), cirq.GridQubit(4, 7), trgt]
    assert embedded.route_cnot(ctrl=ctrl, trgt=trgt) == expected_path


def test_movement(circuit5: cirq.Circuit) -> None:
    movement = MovementLayout(circuit5, num_t_factories=3)
    movement.reload_factories(ftype="s")
    movement.reload_factories(ftype="t")
    G = movement.layout_graph
    # Check factories are used up when routed
    factories = [movement.patch_at(position) for position in [(1, 2), (2, 0), (2, 1)]]
    target = movement.patch_at((0, 2)).logical_qubits[0]
    factory_patch = movement.nearest_factory(qubits=target, ftype="t")
    assert factory_patch in factories
    factories.remove(factory_patch)
    new_target = movement.patch_at((1, 1)).logical_qubits[0]
    new_factory_patch = movement.nearest_factory(qubits=new_target, ftype="t")
    assert new_factory_patch in factories
    # Check that there are no unexpected nodes in the layout graph
    G = movement.layout_graph
    assert len(G.nodes) == 8
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
    assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 0
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
        )
        == 0
    )
    assert (
        sum(
            1
            for node in G.nodes
            if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
        )
        == 3
    )
    assert all(isinstance(node, codepatch.RotatedSurfaceCodePatch) for node in G.nodes)
    assert all(
        isinstance(qubit, codepatch.LogicalQubit) for qubit in movement.mapped_circuit.all_qubits()
    )
    assert len({patch.patch_id for patch in movement.code_patches}) == len(G.nodes)
    assert movement.distance(movement.patch_at((0, 0)), movement.patch_at((1, 1))) == 26
    assert movement.num_physical_qubits == 8 * 97


def test_movement_patch_builder(circuit5: cirq.Circuit) -> None:
    built_patch_ids = []

    def patch_builder(patch_id: int) -> codepatch.RotatedSurfaceCodePatch:
        built_patch_ids.append(patch_id)
        return codepatch.RotatedSurfaceCodePatch(patch_id=patch_id, d=5)

    movement = MovementLayout(circuit5, num_t_factories=3, patch_builder=patch_builder)

    assert built_patch_ids == list(range(8))
    assert all(patch.d == 5 for patch in movement.code_patches)
    assert movement.distance(movement.patch_at((0, 0)), movement.patch_at((1, 1))) == 18
    assert movement.num_physical_qubits == 8 * 49


def test_movement_patch_builder_validation(circuit5: cirq.Circuit) -> None:
    def wrong_id_builder(patch_id: int) -> codepatch.RotatedSurfaceCodePatch:
        return codepatch.RotatedSurfaceCodePatch(patch_id=patch_id + 1, d=3)

    with pytest.raises(ValueError, match="returned patch_id 1; expected 0"):
        MovementLayout(circuit5, patch_builder=wrong_id_builder)

    def empty_patch_builder(patch_id: int) -> codepatch.RotatedSurfaceCodePatch:
        return codepatch.CodePatch(
            patch_id=patch_id,
            n=0,
            k=0,
            d=None,
            num_measure_qubits=0,
            logical_qubits=[],
        )  # type: ignore[return-value]

    with pytest.raises(ValueError, match="exactly one logical qubit"):
        MovementLayout(circuit5, patch_builder=empty_patch_builder)


def test_general_exceptions(circuit5: cirq.Circuit) -> None:
    movement = MovementLayout(circuit5)
    with pytest.raises(ValueError, match="not a valid"):
        movement.reload_factories(ftype="q")
    ctrl = movement.patch_at((0, 0)).logical_qubits[0]
    trgt = movement.patch_at((0, 1)).logical_qubits[0]
    with pytest.raises(NotImplementedError):
        _ = movement.route_cnot(ctrl=ctrl, trgt=trgt)
    with pytest.raises(ValueError, match="No t factories available"):
        movement.reset_graph()
        _ = movement.nearest_factory(ctrl, "t")
    with pytest.raises(ValueError, match="No factories available"):
        _ = movement.available_factories(ftype="toffoli")
    with pytest.raises(TypeError, match="No code patch found"):
        movement.patch_for(cirq.GridQubit(0, 0))

    empty_patch = codepatch.CodePatch(
        patch_id=100,
        n=0,
        k=0,
        d=None,
        num_measure_qubits=0,
        logical_qubits=[],
    )
    with pytest.raises(ValueError, match="exactly one logical qubit"):
        movement.circuit_qubit(empty_patch)

    column = Column(circuit5)
    assert column.distance(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)) == 2
    with pytest.raises(ValueError, match="No layout position found"):
        column.position_of(cirq.LineQubit(0))


def test_reset_and_reload(circuit5: cirq.Circuit) -> None:
    column = Column(circuit5)
    # Assert all start with the used status
    assert all(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
        ],
    )
    # Reloading S should reload all S factories
    column.reload_factories("s")
    assert not any(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
            and column.layout_graph.nodes[node]["ftype"] == "s"
        ],
    )
    # Reloading T should reload all T factories
    column.reload_factories("t")
    assert not any(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
            and column.layout_graph.nodes[node]["ftype"] == "t"
        ],
    )
    # Resetting should unload all factories
    column.reset_graph()
    assert all(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
        ],
    )


def test_distillery(circuit5: cirq.Circuit) -> None:
    """
    Test that the distillery works with both T and CCZ Distillation
    """
    distillery = MovementDistillery(
        circuit5,
        num_t_factories=3,
        num_ccz_factories=2,
        patch_builder=functools.partial(codepatch.RotatedSurfaceCodePatch, d=3),
    )
    distillery.reload_factories(ftype="s")
    distillery.reload_factories(ftype="t")
    distillery.reload_factories(ftype="ccz")

    expected_program_positions = {(0, i) for i in range(5)}
    realized_program_positions = {
        distillery.position_of(patch)
        for patch in distillery.layout_graph.nodes
        if distillery.layout_graph.nodes[patch]["patch_type"] == "data"
    }
    assert expected_program_positions == realized_program_positions

    expected_factory_positions = {
        (0, 5),
        (3, 0),
        (5, 7),
        (8, 2),
        (8, 3),
        (8, 4),
        (10, 1),
        (10, 2),
        (10, 3),
    }
    realized_factory_positions = {
        distillery.position_of(patch) for patch in distillery._all_factories
    }
    assert expected_factory_positions == realized_factory_positions

    expected_block_positions = set(distillery.grid) - (
        expected_program_positions.union(expected_factory_positions)
    )
    realized_block_positions = {
        distillery.position_of(patch)
        for patch in distillery.layout_graph.nodes
        if distillery.layout_graph.nodes[patch]["patch_type"] == "block"
    }
    assert expected_block_positions == realized_block_positions

    ccz_factory = tuple(distillery.patch_at((8, col)) for col in (2, 3, 4))
    expected_ccz_block = {(10, 0)}
    for idx in range(2, 12):
        expected_ccz_block.add((8, idx))
    for idx in range(12):
        expected_ccz_block.add((9, idx))
    realized_ccz_block = {
        distillery.position_of(patch) for patch in distillery.distillation_block(ccz_factory)
    }
    assert expected_ccz_block == realized_ccz_block

    # Check that nearest T factory is as expected and changes when used
    t_target = distillery.patch_at((0, 0)).logical_qubits[0]
    expected_t_factory = distillery.patch_at((3, 0))
    assert distillery.nearest_factory(qubits=t_target, ftype="t") == expected_t_factory
    expected_t_factory = distillery.patch_at((0, 5))
    assert distillery.nearest_factory(qubits=t_target, ftype="t") == expected_t_factory

    # Check that the nearest Toff factory is as expected and changes when used
    ccz_target = tuple(distillery.patch_at((0, col)).logical_qubits[0] for col in (2, 3, 4))
    expected_ccz_factory = tuple(distillery.patch_at((8, col)) for col in (2, 3, 4))
    assert distillery.nearest_factory(ccz_target, ftype="ccz") == expected_ccz_factory

    expected_ccz_factory = tuple(distillery.patch_at((10, col)) for col in (1, 2, 3))
    assert distillery.nearest_factory(ccz_target, ftype="ccz") == expected_ccz_factory
