"""Behavioral tests for lattice-surgery geometry independent of DEQ text."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from surface_code_deq.rotated_surface_code import RotatedSurfaceCode
from surface_code_deq.surgery_geometry import (
    merge_stabilizers,
    merge_layout,
    retained_layout_stabilizers,
    seam_measurement_instruction,
)


def test_merge_layout_places_the_seam_between_the_two_input_patches() -> None:
    patch = RotatedSurfaceCode(3, 3)
    horizontal = merge_layout(patch, patch, "XX")
    vertical = merge_layout(patch, patch, "ZZ")

    assert (horizontal.width, horizontal.height) == (7, 3)
    assert (vertical.width, vertical.height) == (3, 7)
    assert horizontal.qubit_at(3, 0) == horizontal.seam_data_start
    assert vertical.qubit_at(0, 3) == vertical.seam_data_start


def test_retained_and_new_checks_reconstruct_the_merged_code() -> None:
    for basis in ("XX", "ZZ"):
        patch = RotatedSurfaceCode(3, 3)
        layout = merge_layout(patch, patch, basis)
        expected = RotatedSurfaceCode(layout.width, layout.height).stabilizers(
            layout.qubit_at
        )
        actual = [
            *retained_layout_stabilizers(layout),
            *merge_stabilizers(layout),
        ]
        assert set(actual) == set(expected)


def test_merge_basis_selects_the_complementary_seam_measurement() -> None:
    assert seam_measurement_instruction("XX") == "M"
    assert seam_measurement_instruction("ZZ") == "MX"


def test_mxx_layout_supports_different_widths_with_a_shared_height() -> None:
    layout = merge_layout(
        RotatedSurfaceCode(3, 5), RotatedSurfaceCode(5, 5), "XX"
    )
    assert (layout.width, layout.height) == (9, 5)
    assert layout.seam_length == 5
    assert layout.seam_data_start == 15 + 25
    assert layout.qubit_at(3, 0) == layout.seam_data_start
    assert layout.second_patch_coordinate_offset == (4, 0)


def test_mzz_layout_supports_different_heights_with_a_shared_width() -> None:
    layout = merge_layout(
        RotatedSurfaceCode(5, 3), RotatedSurfaceCode(5, 5), "ZZ"
    )
    assert (layout.width, layout.height) == (5, 9)
    assert layout.seam_length == 5
    assert layout.seam_data_start == 15 + 25
    assert layout.qubit_at(0, 3) == layout.seam_data_start
    assert layout.second_patch_coordinate_offset == (0, 4)


def test_different_size_layout_checks_reconstruct_the_merged_code() -> None:
    layouts = (
        merge_layout(RotatedSurfaceCode(3, 5), RotatedSurfaceCode(5, 5), "XX"),
        merge_layout(RotatedSurfaceCode(5, 3), RotatedSurfaceCode(5, 5), "ZZ"),
    )
    for layout in layouts:
        expected = RotatedSurfaceCode(layout.width, layout.height).stabilizers(
            layout.qubit_at
        )
        actual = [
            *retained_layout_stabilizers(layout),
            *merge_stabilizers(layout),
        ]
        assert set(actual) == set(expected)
