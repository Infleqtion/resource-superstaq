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
import pytest

from . import codepatch as cpa
from . import layout as lyt


def placed_zone(row: int, col: int, rows: int, cols: int) -> lyt.Placement:
    """Build simple placement from rectangular footprint"""
    return lyt.Placement(lyt.Zone(rows, cols, "Test"), row, col)


# def placed_patch(row: int, col: int, pid: int = 0) -> lyt.Placement:
#     return lyt.Placement(cpa.RotatedSurfaceCodePatch(pid, 3), row, col)
# def patch(row: int, col: int, pid: int = 0, distance: int = 3) -> lyt.Placement:
#     return lyt.Placement(cpa.RotatedSurfaceCodePatch(pid, distance), row, col)


def test_rect_dimensions() -> None:
    rect = lyt.Rect(2, 3)
    assert rect.rows == 2
    assert rect.cols == 3


@pytest.mark.parametrize("rows, cols", [(0, 1), (1, 0), (-1, 1), (1, -1), (-1, -2)])
def test_rect_nonpositive(rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _ = lyt.Rect(rows, cols)


def test_basic_placement() -> None:
    p = placed_zone(2, 3, 4, 5)
    assert p.element.footprint == lyt.Rect(4, 5)
    assert p.row == 2
    assert p.col == 3

    assert p.row_end == 6
    assert p.col_end == 8


def test_basic_zone() -> None:
    z = lyt.Zone(2, 3, "Test")
    assert z.rows == 2
    assert z.cols == 3
    assert z.role == "Test"


def test_placement_overlap() -> None:
    middle = placed_zone(8, 8, 3, 3)
    non_overlap = placed_zone(11, 5, 3, 3)
    assert not middle.overlaps(non_overlap)
    assert not non_overlap.overlaps(middle)

    top_left_corner = placed_zone(5, 5, 4, 4)
    assert middle.overlaps(top_left_corner)
    assert top_left_corner.overlaps(middle)

    bottom_half = placed_zone(10, 7, 9, 7)
    assert middle.overlaps(bottom_half)
    assert bottom_half.overlaps(middle)

    thin_strip = placed_zone(4, 9, 13, 1)
    assert middle.overlaps(thin_strip)
    assert thin_strip.overlaps(middle)


@pytest.mark.parametrize(
    "row, col, truth",
    [
        (2, 0, True),  # Along the bottom edge
        (2, 1, True),  # Partially shared bottom edge
        (0, 3, True),  # Along the right edge
        (1, 3, True),  # Partially shared right edge
        (2, 3, False),  # Corner contact isn't a border
        (3, 0, False),  # Gap between rectangles
        (1, 1, False),  # Overlapping interiors
    ],
)
def test_placement_borders(row, col, truth) -> None:
    first = placed_zone(0, 0, 2, 3)
    second = placed_zone(row, col, 2, 3)
    assert first.borders(second) == truth
    assert second.borders(first) == truth


def test_placement_requires_equal_footprints() -> None:
    with pytest.raises(ValueError, match="Both placements must have the same footprint"):
        _ = placed_zone(0, 0, 3, 3).borders(placed_zone(3, 3, 4, 5))


def test_layout_add_assigns_ids_and_allows_edge_touching():
    layout = lyt.Layout(rows=10, cols=10)
    first = layout.add(cpa.RotatedSurfaceCodePatch(0, 3), 0, 0)
    second = layout.add(cpa.RotatedSurfaceCodePatch(1, 3), 0, 5)
    third = layout.add(cpa.RotatedSurfaceCodePatch(2, 3), 5, 0)

    assert (first, second, third) == (1, 2, 3)
    assert (layout.placements[first].row, layout.placements[first].col) == (0, 0)
    assert layout.placements[first].element.footprint == lyt.Rect(5, 5)
    assert layout.placements[second].col == 5
    assert layout.placements[second].col_end == 10 == layout.cols
    assert layout.placements[third].row == 5
    assert layout.placements[third].row_end == 10 == layout.rows


def test_layout_rejects_occupied_placement_without_changing_state() -> None:
    layout = lyt.Layout(8, 8)
    layout.add(cpa.RotatedSurfaceCodePatch(0, 3), 2, 2)
    before = layout.placements.copy()
    with pytest.raises(ValueError, match="This space is already occupied"):
        _ = layout.add(cpa.RotatedSurfaceCodePatch(1, 3), 3, 3)
    assert layout.placements == before


@pytest.mark.parametrize("row, col", [(-1, 0), (0, -1), (4, 0), (0, 5)])
def test_layout_rejects_out_of_bounds(row: int, col: int) -> None:
    layout = lyt.Layout(8, 8)
    with pytest.raises(ValueError, match="This placement is out of bounds"):
        layout.add(cpa.RotatedSurfaceCodePatch(0, 3), row, col)
    assert layout.placements == {}


def test_layout_remove() -> None:
    layout = lyt.Layout(5, 5)
    placed = layout.add(cpa.RotatedSurfaceCodePatch(0, 3), 0, 0)
    layout.remove(placed)
    assert layout.placements == {}

    with pytest.raises(ValueError, match="No entry found"):
        _ = layout.remove(placed)


@pytest.mark.parametrize("row, col, truth", [(0, 0, True), (4, 4, True), (5, 5, False)])
def test_layout_occupied(row: int, col: int, truth: bool) -> None:
    layout = lyt.Layout(10, 10)
    layout.add(cpa.RotatedSurfaceCodePatch(0, 3), 0, 0)
    to_place = lyt.Placement(
        cpa.RotatedSurfaceCodePatch(1, 3),
        row,
        col,
    )
    zone = layout.add(lyt.Zone(2, 3, "exclude"), 7, 6)
    assert layout.occupied(to_place, exclude=zone) == truth


@pytest.mark.parametrize(
    "row, col, truth",
    [(1, 0, True), (-1, 4, False), (3, -2, False), (3, 3, True), (4, 3, False), (3, 4, False)],
)
def test_layout_in_bounds(row: int, col: int, truth: bool) -> None:
    layout = lyt.Layout(8, 8)
    to_place = lyt.Placement(
        cpa.RotatedSurfaceCodePatch(1, 3),
        row,
        col,
    )
    assert layout.in_bounds(to_place) == truth


def test_layout_move() -> None:
    patch = cpa.RotatedSurfaceCodePatch(0, 3)
    layout = lyt.Layout(9, 9)
    pid = layout.add(patch, 1, 1)
    layout.move(pid, 1, 1)  # Self move is ok
    layout.move(pid, 2, 3)  # Overlap with original is ok
    # Check new coords have been added
    assert layout.placements[pid].row == 2
    assert layout.placements[pid].col == 3
    # Check global coords matches local coords plus (row, col)
    assert layout.global_coord(pid, (1, 2)) == (3, 5)


def test_layout_unallowed_moves() -> None:
    layout = lyt.Layout(10, 10)
    patch1 = cpa.RotatedSurfaceCodePatch(0, 3)
    patch2 = cpa.RotatedSurfaceCodePatch(1, 3)
    moving = layout.add(patch1, 0, 0)
    layout.add(patch2, 5, 5)
    before = layout.placements.copy()
    # Checl three out of bounds examples
    for dest in [(-1, 0), (1, 10), (3, 9)]:
        with pytest.raises(ValueError, match="out of bounds"):
            layout.move(moving, *dest)
        assert layout.placements == before
    # Check three examples of occupation
    for dest in [(1, 2), (4, 4), (3, 2)]:
        print(dest)
        with pytest.raises(ValueError, match="already occupied"):
            layout.move(moving, *dest)
        assert layout.placements == before


def test_layout_get_patch() -> None:
    layout = lyt.Layout(25, 25)
    d3_patch = layout.add(cpa.RotatedSurfaceCodePatch(0, 3), 0, 0)
    d5_patch = layout.add(cpa.RotatedSurfaceCodePatch(1, 5), 0, 5)
    assert layout.get_patch(0).d == 3
    assert layout.get_patch(1).d == 5
    layout.remove(d3_patch)
    with pytest.raises(ValueError, match="Found no CodePatch"):
        layout.get_patch(0)
    layout.remove(d5_patch)
    with pytest.raises(ValueError, match="Found no CodePatch"):
        layout.get_patch(1)


@pytest.mark.parametrize(
    "t_cult, t_dist, ccz_dist, expected_patches",
    [(0, 0, 0, 3), (3, 0, 0, 6), (0, 1, 0, 34), (0, 0, 2, 49), (1, 1, 1, 58)],
)
def test_generate_movement_patch_count(
    t_cult: int, t_dist: int, ccz_dist: int, expected_patches: int
) -> None:
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=3,
        surface_code_distance=5,
        t_cultivation_factories=t_cult,
        t_distillation_factories=t_dist,
        ccz_distillation_factories=ccz_dist,
        interaction_zone=False,
        measure_zone=False,
    )
    assert len(layout.placements) == expected_patches
    assert all(
        isinstance(p.element, cpa.RotatedSurfaceCodePatch) for p in layout.placements.values()
    )
    assert all(p.element.d == 5 for p in layout.placements.values())
    assert {p.element.patch_id for p in layout.placements.values()} == set(range(expected_patches))
    assert layout.regions == {}


@pytest.mark.parametrize(
    "interaction_zone, measure_zone, expected_rows, expected_zone_ids",
    [
        (False, False, 10, set()),
        (True, False, 15, {0}),
        (False, True, 15, {1}),
        (True, True, 20, {0, 1}),
    ],
)
def test_generate_movement_zones(
    interaction_zone: bool, measure_zone: bool, expected_rows: int, expected_zone_ids: set
) -> None:
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=5,
        surface_code_distance=3,
        t_cultivation_factories=1,
        t_distillation_factories=0,
        ccz_distillation_factories=0,
        interaction_zone=interaction_zone,
        measure_zone=measure_zone,
        extra_keyword=True,  # Make sure kwargs accepted
    )
    assert layout.rows == expected_rows
    assert layout.cols == 15
    assert set(layout.regions) == expected_zone_ids
    if interaction_zone:
        region = layout.regions[0]
        assert region.element.role == "Interaction"
        assert region.row == 0
        assert region.col == 0
        assert region.element.footprint == lyt.Rect(5, 15)
    if measure_zone:
        region = layout.regions[1]
        assert region.element.role == "Measure"
        assert region.row == expected_rows - 5
        assert region.col == 0
        assert region.element.footprint == lyt.Rect(5, 15)


def test_generate_movement_coordinates() -> None:
    # 5 program + 1 T Cult + 31 T Dist + 23 CCZ Dist = 60 patches
    layout = lyt.generate_movement_layout(
        surface_code_program_qubits=5,
        surface_code_distance=5,
        t_cultivation_factories=1,
        t_distillation_factories=1,
        ccz_distillation_factories=1,
        interaction_zone=True,
        measure_zone=True,
    )
    assert layout.rows == 90
    assert layout.cols == 72
    assert layout.placements[0].row == 9
    assert layout.placements[0].col == 0

    assert layout.placements[7].row == 9
    assert layout.placements[7].col == 63

    assert layout.placements[8].row == 18
    assert layout.placements[8].col == 0

    assert layout.placements[59].row == 72
    assert layout.placements[59].col == 27

    assert layout.regions[0].row == 0
    assert layout.regions[0].col == 0
    assert layout.regions[1].row == 81
    assert layout.regions[1].col == 0


@pytest.mark.parametrize("program_qubits, rows_of_patches", [(1, 2), (2, 2), (3, 4), (4, 4)])
def test_generate_column(program_qubits: int, rows_of_patches: int) -> None:
    layout = lyt.generate_column_layout(
        surface_code_program_qubits=program_qubits,
        surface_code_distance=3,
        extra_keyword=True,  # Make sure kwargs accepted
    )
    assert layout.rows == rows_of_patches * 5
    assert layout.cols == 7 * 5
    assert len(layout.placements) == rows_of_patches * 7
    assert set(layout.placements) == set(range(rows_of_patches * 7))

    for pid, p in layout.placements.items():
        assert p.row == (pid // 7) * 5
        assert p.col == (pid % 7) * 5
        assert p.element.patch_id == pid
        assert p.element.d == 3
        assert p.element.footprint == lyt.Rect(5, 5)


# @pytest.fixture
# def circuit5() -> cirq.Circuit:
#     circuit = cirq.testing.random_circuit(
#         cirq.LineQubit.range(5),
#         10,
#         0.6,
#         {cirq.T: 1, cirq.S: 1, cirq.CNOT: 2, cirq.H: 1, cirq.CCZ: 3},
#         42,
#     )
#     return circuit


# def test_column(circuit5: cirq.Circuit) -> None:
#     column = Column(circuit5)
#     column.reload_factories(ftype="s")
#     column.reload_factories(ftype="t")
#     (factory1,) = column.nearest_factory(qubits=cirq.GridQubit(0, 2), ftype="s")
#     assert factory1 == cirq.GridQubit(0, 0)
#     (factory2,) = column.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t")
#     assert factory2 in [
#         cirq.GridQubit(3, 0),
#         cirq.GridQubit(1, 0),
#     ]
#     (factory3,) = column.nearest_factory(qubits=cirq.GridQubit(2, 4), ftype="t")
#     assert factory3 in [
#         cirq.GridQubit(1, 6),
#         cirq.GridQubit(3, 6),
#     ]
#     # Now that (0, 0) is used, the nearest S factory to lq (0, 2) is (2, 0)
#     (factory4,) = column.nearest_factory(qubits=cirq.GridQubit(0, 2), ftype="s")
#     assert factory4 == cirq.GridQubit(2, 0)
#     G = column.layout_graph
#     # Total number of nodes should be 7 x 6 = 42
#     assert len(G.nodes) == 42
#     # Of those 42 nodes, 5 should be logical qubits, 25 ancillas, and 6 of each factory type
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 25
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
#         )
#         == 6
#     )
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
#         )
#         == 6
#     )
#     # Confirm the expected routes for a couple of qubits
#     ctrl, trgt = cirq.GridQubit(2, 2), cirq.GridQubit(5, 0)
#     # Both of these paths are valid
#     path_a = [
#         ctrl,
#         cirq.GridQubit(2, 1),
#         cirq.GridQubit(3, 1),
#         cirq.GridQubit(4, 1),
#         cirq.GridQubit(5, 1),
#         trgt,
#     ]
#     path_b = [
#         ctrl,
#         cirq.GridQubit(3, 2),
#         cirq.GridQubit(3, 1),
#         cirq.GridQubit(4, 1),
#         cirq.GridQubit(5, 1),
#         trgt,
#     ]
#     assert column.route_cnot(ctrl=ctrl, trgt=trgt) in [path_a, path_b]
#     ctrl, trgt = cirq.GridQubit(0, 4), cirq.GridQubit(5, 6)
#     path_a = [
#         ctrl,
#         cirq.GridQubit(0, 5),
#         cirq.GridQubit(1, 5),
#         cirq.GridQubit(2, 5),
#         cirq.GridQubit(3, 5),
#         cirq.GridQubit(4, 5),
#         cirq.GridQubit(5, 5),
#         trgt,
#     ]
#     path_b = [
#         ctrl,
#         cirq.GridQubit(1, 4),
#         cirq.GridQubit(1, 5),
#         cirq.GridQubit(2, 5),
#         cirq.GridQubit(3, 5),
#         cirq.GridQubit(4, 5),
#         cirq.GridQubit(5, 5),
#         trgt,
#     ]
#     assert column.route_cnot(ctrl=ctrl, trgt=trgt) in [path_a, path_b]


# def test_sandwich(circuit5: cirq.Circuit) -> None:
#     sandwich = FactorySandwich(circuit5, num_t_factories=3, num_s_factories=5)
#     sandwich.reload_factories(ftype="s")
#     sandwich.reload_factories(ftype="t")
#     # Check that nearest T factory is as expected and changes when used
#     (factory1,) = sandwich.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t")
#     assert factory1 == cirq.GridQubit(4, 2)
#     (factory2,) = sandwich.nearest_factory(qubits=cirq.GridQubit(2, 2), ftype="t")
#     assert factory2 == cirq.GridQubit(4, 1)
#     (factory3,) = sandwich.nearest_factory(qubits=cirq.GridQubit(2, 4), ftype="s")
#     assert factory3 == cirq.GridQubit(0, 4)
#     # Check that there are no unexpected nodes in the layout graph
#     G = sandwich.layout_graph
#     assert len(G.nodes) == 23
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 10
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
#         )
#         == 5
#     )
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
#         )
#         == 3
#     )
#     # Check that a CNOT has a reasonable path
#     ctrl, trgt = cirq.GridQubit(2, 0), cirq.GridQubit(0, 4)
#     expected_path = [
#         ctrl,
#         cirq.GridQubit(1, 0),
#         cirq.GridQubit(1, 1),
#         cirq.GridQubit(1, 2),
#         cirq.GridQubit(1, 3),
#         cirq.GridQubit(1, 4),
#         trgt,
#     ]
#     assert sandwich.route_cnot(ctrl=ctrl, trgt=trgt) == expected_path

#     sandwich.route_cnot(
#         ctrl=cirq.GridQubit(2, 1),
#         trgt=cirq.GridQubit(2, 2),
#     )  # Hopefully this covers 116?


# def test_embedded(circuit5: cirq.Circuit) -> None:
#     embedded = Embedded(circuit5)
#     embedded.reload_factories(ftype="s")
#     embedded.reload_factories(ftype="t")
#     # Check available qubits across several situations of ambiguity
#     (factory1,) = embedded.nearest_factory(cirq.GridQubit(2, 6), ftype="t")
#     assert factory1 in [
#         cirq.GridQubit(0, 5),
#         cirq.GridQubit(0, 7),
#         cirq.GridQubit(1, 8),
#         cirq.GridQubit(3, 8),
#     ]
#     (factory2,) = embedded.nearest_factory(cirq.GridQubit(4, 4), ftype="s")
#     assert factory2 == cirq.GridQubit(6, 4)
#     (factory3,) = embedded.nearest_factory(cirq.GridQubit(4, 2), ftype="s")
#     assert factory3 in [
#         cirq.GridQubit(4, 0),
#         cirq.GridQubit(6, 2),
#     ]
#     # Check that there are no unexpected nodes in the layout graph
#     G = embedded.layout_graph
#     assert len(G.nodes) == 59
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 30
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
#         )
#         == 10
#     )
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
#         )
#         == 14
#     )
#     # Check that a CNOT has a reasonable path
#     ctrl, trgt = cirq.GridQubit(4, 4), cirq.GridQubit(4, 8)
#     expected_path = [ctrl, cirq.GridQubit(4, 5), cirq.GridQubit(4, 6), cirq.GridQubit(4, 7), trgt]
#     assert embedded.route_cnot(ctrl=ctrl, trgt=trgt) == expected_path


# def test_movement(circuit5: cirq.Circuit) -> None:
#     movement = MovementLayout(circuit5, num_t_factories=3, architecture="DSM")
#     movement.reload_factories(ftype="s")
#     movement.reload_factories(ftype="t")
#     G = movement.layout_graph
#     # Check factories are used up when routed
#     factories = [cirq.GridQubit(1, 2), cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)]
#     (factory_qubit,) = movement.nearest_factory(qubits=cirq.GridQubit(0, 2), ftype="t")
#     assert factory_qubit in factories
#     factories.remove(factory_qubit)
#     (new_factory_qubit,) = movement.nearest_factory(qubits=cirq.GridQubit(1, 1), ftype="t")
#     assert new_factory_qubit in factories
#     # Check that there are no unexpected nodes in the layout graph
#     G = movement.layout_graph
#     assert len(G.nodes) == 8
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "data") == 5
#     assert sum(1 for node in G.nodes if G.nodes[node]["patch_type"] == "ancilla") == 0
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "s"
#         )
#         == 0
#     )
#     assert (
#         sum(
#             1
#             for node in G.nodes
#             if G.nodes[node]["patch_type"] == "factory" and G.nodes[node]["ftype"] == "t"
#         )
#         == 3
#     )


# def test_general_exceptions(circuit5: cirq.Circuit) -> None:
#     movement = MovementLayout(circuit5, architecture="DSM")
#     with pytest.raises(ValueError, match="not a valid"):
#         movement.reload_factories(ftype="q")  # type: ignore[arg-type]
#     ctrl, trgt = cirq.GridQubit(0, 2), cirq.GridQubit(2, 1)
#     with pytest.raises(NotImplementedError):
#         _ = movement.route_cnot(ctrl=ctrl, trgt=trgt)
#     with pytest.raises(ValueError, match="No t factories available"):
#         movement.reset_graph()
#         _ = movement.nearest_factory(cirq.GridQubit(0, 2), "t")
#     with pytest.raises(ValueError, match="No factories available"):
#         _ = movement.available_factories(ftype="toffoli")  # type: ignore[arg-type]
#     with pytest.raises(ValueError, match="Not a recognized zone type"):
#         _ = movement.zone_qubits(zone_type="not a zone type")  # type: ignore[arg-type]


# def test_reset_and_reload(circuit5: cirq.Circuit) -> None:
#     column = Column(circuit5)
#     # Assert all start with the used status
#     assert all(
#         [
#             column.layout_graph.nodes[node]["used"]
#             for node in column.layout_graph.nodes
#             if column.layout_graph.nodes[node]["patch_type"] == "factory"
#         ],
#     )
#     # Reloading S should reload all S factories
#     column.reload_factories("s")
#     assert not any(
#         [
#             column.layout_graph.nodes[node]["used"]
#             for node in column.layout_graph.nodes
#             if column.layout_graph.nodes[node]["patch_type"] == "factory"
#             and column.layout_graph.nodes[node]["ftype"] == "s"
#         ],
#     )
#     # Reloading T should reload all T factories
#     column.reload_factories("t")
#     assert not any(
#         [
#             column.layout_graph.nodes[node]["used"]
#             for node in column.layout_graph.nodes
#             if column.layout_graph.nodes[node]["patch_type"] == "factory"
#             and column.layout_graph.nodes[node]["ftype"] == "t"
#         ],
#     )
#     # Resetting should unload all factories
#     column.reset_graph()
#     assert all(
#         [
#             column.layout_graph.nodes[node]["used"]
#             for node in column.layout_graph.nodes
#             if column.layout_graph.nodes[node]["patch_type"] == "factory"
#         ],
#     )


# def test_distillery(circuit5: cirq.Circuit) -> None:
#     """
#     Test that the distillery works with both T and CCZ Distillation
#     """
#     distillery = MovementDistillery(
#         circuit5, num_t_factories=3, num_ccz_factories=2, architecture="DSM"
#     )
#     distillery.reload_factories(ftype="s")
#     distillery.reload_factories(ftype="t")
#     distillery.reload_factories(ftype="ccz")

#     expected_program_qubits = set(cirq.GridQubit(0, i) for i in range(5))
#     realized_program_qubits = set(
#         q
#         for q in distillery.layout_graph.nodes
#         if distillery.layout_graph.nodes[q]["patch_type"] == "data"
#     )
#     assert expected_program_qubits == realized_program_qubits

#     expected_factories = {
#         cirq.GridQubit(0, 5),
#         cirq.GridQubit(3, 0),
#         cirq.GridQubit(5, 7),
#         cirq.GridQubit(8, 2),
#         cirq.GridQubit(8, 3),
#         cirq.GridQubit(8, 4),
#         cirq.GridQubit(10, 1),
#         cirq.GridQubit(10, 2),
#         cirq.GridQubit(10, 3),
#     }
#     realized_factories = distillery._all_factories
#     assert expected_factories == realized_factories

#     expected_block_qubits = set(q for q in distillery.layout_graph.nodes) - (
#         expected_program_qubits.union(expected_factories)
#     )
#     realized_block_qubits = set(
#         q
#         for q in distillery.layout_graph.nodes
#         if distillery.layout_graph.nodes[q]["patch_type"] == "block"
#     )
#     assert expected_block_qubits == realized_block_qubits

#     ccz_factory = (cirq.GridQubit(8, 2), cirq.GridQubit(8, 3), cirq.GridQubit(8, 4))
#     expected_ccz_block = set([cirq.GridQubit(10, 0)])
#     for idx in range(2, 12):
#         expected_ccz_block.add(cirq.GridQubit(8, idx))
#     for idx in range(12):
#         expected_ccz_block.add(cirq.GridQubit(9, idx))
#     realized_ccz_block = set(distillery.distillation_block(ccz_factory))
#     assert expected_ccz_block == realized_ccz_block

#     # Check that nearest T factory is as expected and changes when used
#     t_target = cirq.GridQubit(0, 0)
#     expected_t_factory = cirq.GridQubit(3, 0)
#     (t_factory1,) = distillery.nearest_factory(qubits=t_target, ftype="t")
#     assert t_factory1 == expected_t_factory
#     expected_t_factory = cirq.GridQubit(0, 5)
#     (t_factory2,) = distillery.nearest_factory(qubits=t_target, ftype="t")
#     assert t_factory2 == expected_t_factory

#     # Check that the nearest Toff factory is as expected and changes when used
#     ccz_target = (cirq.GridQubit(0, 2), cirq.GridQubit(0, 3), cirq.GridQubit(0, 4))
#     expected_ccz_factory = (cirq.GridQubit(8, 2), cirq.GridQubit(8, 3), cirq.GridQubit(8, 4))
#     assert distillery.nearest_factory(ccz_target, ftype="ccz") == expected_ccz_factory

#     expected_ccz_factory = (cirq.GridQubit(10, 1), cirq.GridQubit(10, 2), cirq.GridQubit(10, 3))
#     assert distillery.nearest_factory(ccz_target, ftype="ccz") == expected_ccz_factory


# def test_zones(circuit5: cirq.Circuit) -> None:
#     layout = MovementLayout(input_circuit=circuit5, num_t_factories=1, architecture="SSM")
#     expected_interaction_zones = {cirq.GridQubit(-1, j) for j in range(3)}
#     expected_measurement_zones = {cirq.GridQubit(2, j) for j in range(3)}
#     assert set(layout.zone_qubits("interact")) == expected_interaction_zones
#     assert set(layout.zone_qubits("measure")) == expected_measurement_zones
