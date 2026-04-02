import cirq
import pytest
from resource_estimation.layout import (
    Column,
    FactorySandwich,
    MovementLayout,
    Embedded,
)


@pytest.fixture
def circuit5():
    circuit = cirq.testing.random_circuit(
        cirq.LineQubit.range(5),
        10,
        0.6,
        {cirq.T: 1, cirq.S: 1, cirq.CNOT: 2, cirq.H: 1},
        42,
    )
    return circuit


def test_column(circuit5: cirq.Circuit):
    column = Column(circuit5)
    column.reload_factories(ftype="s")
    column.reload_factories(ftype="t")
    assert column.nearest_factory(qubit=cirq.GridQubit(0, 2), ftype="s") == cirq.GridQubit(0, 0)
    assert column.nearest_factory(qubit=cirq.GridQubit(2, 2), ftype="t") in [
        cirq.GridQubit(3, 0),
        cirq.GridQubit(1, 0),
    ]
    assert column.nearest_factory(qubit=cirq.GridQubit(2, 4), ftype="t") in [
        cirq.GridQubit(1, 6),
        cirq.GridQubit(3, 6),
    ]
    # Now that (0, 0) is used, the nearest S factory to lq (0, 2) is (2, 0)
    assert column.nearest_factory(qubit=cirq.GridQubit(0, 2), ftype="s") == cirq.GridQubit(2, 0)
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


def test_sandwich(circuit5: cirq.Circuit):
    sandwich = FactorySandwich(circuit5, num_t_factories=3, num_s_factories=5)
    sandwich.reload_factories(ftype="s")
    sandwich.reload_factories(ftype="t")
    # Check that nearest T factory is as expected and changes when used
    assert sandwich.nearest_factory(qubit=cirq.GridQubit(2, 2), ftype="t") == cirq.GridQubit(4, 2)
    assert sandwich.nearest_factory(qubit=cirq.GridQubit(2, 2), ftype="t") == cirq.GridQubit(4, 1)
    assert sandwich.nearest_factory(qubit=cirq.GridQubit(2, 4), ftype="s") == cirq.GridQubit(0, 4)
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
        ctrl=cirq.GridQubit(2, 1), trgt=cirq.GridQubit(2, 2)
    )  # Hopefully this covers 116?


def test_embedded(circuit5: cirq.Circuit):
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


def test_movement(circuit5: cirq.Circuit):
    movement = MovementLayout(circuit5, num_t_factories=3)
    movement.reload_factories(ftype="s")
    movement.reload_factories(ftype="t")
    G = movement.layout_graph
    # Check factories are used up when routed
    factories = [cirq.GridQubit(1, 2), cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)]
    factory_qubit = movement.nearest_factory(qubit=cirq.GridQubit(0, 2), ftype="t")
    assert factory_qubit in factories
    factories.remove(factory_qubit)
    new_factory_qubit = movement.nearest_factory(qubit=cirq.GridQubit(1, 1), ftype="t")
    assert new_factory_qubit in factories
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


def test_general_exceptions(circuit5: cirq.Circuit):
    movement = MovementLayout(circuit5)
    with pytest.raises(ValueError, match="not a valid"):
        movement.reload_factories(ftype="q")
    ctrl, trgt = cirq.GridQubit(0, 2), cirq.GridQubit(2, 1)
    with pytest.raises(NotImplementedError):
        _ = movement.route_cnot(ctrl=ctrl, trgt=trgt)
    with pytest.raises(ValueError, match="No available"):
        movement.reset_graph()
        _ = movement.nearest_factory(cirq.GridQubit(0, 2), "t")


def test_reset_and_reload(circuit5: cirq.Circuit):
    column = Column(circuit5)
    # Assert all start with the used status
    assert all(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
        ]
    )
    # Reloading S should reload all S factories
    column.reload_factories("s")
    assert not any(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
            and column.layout_graph.nodes[node]["ftype"] == "s"
        ]
    )
    # Reloading T should reload all T factories
    column.reload_factories("t")
    assert not any(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
            and column.layout_graph.nodes[node]["ftype"] == "t"
        ]
    )
    # Resetting should unload all factories
    column.reset_graph()
    assert all(
        [
            column.layout_graph.nodes[node]["used"]
            for node in column.layout_graph.nodes
            if column.layout_graph.nodes[node]["patch_type"] == "factory"
        ]
    )
