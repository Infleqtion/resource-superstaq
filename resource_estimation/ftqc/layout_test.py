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
def test_placement_borders(row: int, col: int, truth: bool) -> None:
    first = placed_zone(0, 0, 2, 3)
    second = placed_zone(row, col, 2, 3)
    assert first.borders(second) == truth
    assert second.borders(first) == truth


def test_placement_requires_equal_footprints() -> None:
    with pytest.raises(ValueError, match="Both placements must have the same footprint"):
        _ = placed_zone(0, 0, 3, 3).borders(placed_zone(3, 3, 4, 5))


def test_layout_add_assigns_ids_and_allows_edge_touching() -> None:
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
        layout.remove(placed)


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
    patch_ids: set[int] = set()
    for p in layout.placements.values():
        assert isinstance(p.element, cpa.RotatedSurfaceCodePatch)
        assert p.element.d == 5
        patch_ids.add(p.element.patch_id)
    assert set(range(expected_patches)) == patch_ids
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
    interaction_zone: bool, measure_zone: bool, expected_rows: int, expected_zone_ids: set[int]
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
        assert isinstance(region.element, lyt.Zone)
        assert region.element.role == "Interaction"
        assert region.row == 0
        assert region.col == 0
        assert region.element.footprint == lyt.Rect(5, 15)
    if measure_zone:
        region = layout.regions[1]
        assert isinstance(region.element, lyt.Zone)
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
        assert isinstance(p.element, cpa.RotatedSurfaceCodePatch)
        assert p.element.patch_id == pid
        assert p.element.d == 3
        assert p.element.footprint == lyt.Rect(5, 5)
