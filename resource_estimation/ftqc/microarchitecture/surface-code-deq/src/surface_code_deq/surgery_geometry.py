"""Geometry and measurement conventions for rotated-code lattice surgery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, TypeAlias

from .rotated_surface_code import RotatedSurfaceCode
from .types import PauliProduct


SurgeryBasis: TypeAlias = Literal["XX", "ZZ"]


class SurgeryLayout(NamedTuple):
    """Physical-wire and display-coordinate layout for one merge orientation."""

    first_patch: RotatedSurfaceCode
    second_patch: RotatedSurfaceCode
    basis: SurgeryBasis
    width: int
    height: int
    qubit_at: Callable[[int, int], int]
    second_patch_coordinate_offset: tuple[int, int]

    @property
    def first_patch_data_size(self) -> int:
        return self.first_patch.num_data_qubits

    @property
    def second_patch_data_size(self) -> int:
        return self.second_patch.num_data_qubits

    @property
    def seam_length(self) -> int:
        """Return the number of data qubits along the shared merge boundary."""
        return self.first_patch.height if self.basis == "XX" else self.first_patch.width

    @property
    def seam_data_start(self) -> int:
        """Return the first data-wire ID reserved for the d-qubit seam."""
        return self.first_patch_data_size + self.second_patch_data_size

    @property
    def merged_data_size(self) -> int:
        """Return the data-wire count of the two patches plus their seam."""
        return self.seam_data_start + self.seam_length


def merge_layout(
    first_patch: RotatedSurfaceCode,
    second_patch: RotatedSurfaceCode,
    basis: SurgeryBasis,
) -> SurgeryLayout:
    """Return geometry for merging patches with a common boundary length.

    The basis selects the joining axis: MXX joins horizontally and MZZ joins
    vertically. The patches must have the same transverse extent. The first
    patch's joining extent must be odd so the rotated-code checkerboard phase
    aligns across the inserted seam.
    """
    if basis not in {"XX", "ZZ"}:
        raise ValueError("basis must be 'XX' or 'ZZ'")
    join_axis = {"XX": 0, "ZZ": 1}[basis]
    transverse_axis = 1 - join_axis
    axis_names = ("width", "height")
    first_dimensions = (first_patch.width, first_patch.height)
    second_dimensions = (second_patch.width, second_patch.height)
    first_join_extent = first_dimensions[join_axis]
    if first_dimensions[transverse_axis] != second_dimensions[transverse_axis]:
        raise ValueError(
            f"M{basis} requires patches with equal {axis_names[transverse_axis]}s"
        )
    if first_join_extent % 2 == 0:
        raise ValueError(
            f"M{basis} requires the first patch {axis_names[join_axis]} to be odd"
        )

    merged_dimensions = list(first_dimensions)
    merged_dimensions[join_axis] += 1 + second_dimensions[join_axis]
    width, height = merged_dimensions
    first_size = first_patch.num_data_qubits
    seam_data_start = first_size + second_patch.num_data_qubits

    def qubit_at(x: int, y: int) -> int:
        coordinates = (x, y)
        join_coordinate = coordinates[join_axis]
        if join_coordinate < first_join_extent:
            return first_patch.canonical_wire(x, y)
        if join_coordinate == first_join_extent:
            return seam_data_start + coordinates[transverse_axis]
        second_coordinates = list(coordinates)
        second_coordinates[join_axis] -= first_join_extent + 1
        return first_size + second_patch.canonical_wire(*second_coordinates)

    second_patch_coordinate_offset = tuple(
        first_join_extent + 1 if axis == join_axis else 0 for axis in range(2)
    )

    return SurgeryLayout(
        first_patch,
        second_patch,
        basis,
        width,
        height,
        qubit_at,
        second_patch_coordinate_offset,
    )


def seam_measurement_instruction(basis: SurgeryBasis) -> str:
    """Return DEQ's physical instruction for measuring the complementary seam."""
    try:
        return {"XX": "M", "ZZ": "MX"}[basis]
    except KeyError as error:
        raise ValueError("basis must be 'XX' or 'ZZ'") from error


def retained_layout_stabilizers(
    layout: SurgeryLayout,
) -> list[PauliProduct]:
    """Return all input-patch checks that remain after the merge begins."""
    return [
        *_retained_patch_stabilizers(layout, patch="first"),
        *_retained_patch_stabilizers(layout, patch="second"),
    ]


def _retained_patch_stabilizers(
    layout: SurgeryLayout, *, patch: Literal["first", "second"]
) -> list[PauliProduct]:
    """Return the retained checks from one named input patch."""
    code = layout.first_patch if patch == "first" else layout.second_patch
    offset = 0 if patch == "first" else layout.first_patch_data_size

    def wire_at(x: int, y: int) -> int:
        return offset + y * code.width + x

    products = code.stabilizers(wire_at)
    if layout.basis == "XX":
        side = "right" if patch == "first" else "left"
    else:
        side = "bottom" if patch == "first" else "top"
    facing = {support for _, support in code.boundary_stabilizers(side, wire_at)}
    return [product for product in products if product[1] not in facing]


def merge_stabilizers(layout: SurgeryLayout) -> list[PauliProduct]:
    """Return checks introduced by merging the layout's two input patches."""
    merged = RotatedSurfaceCode(layout.width, layout.height).stabilizers(layout.qubit_at)
    retained = set(retained_layout_stabilizers(layout))
    return [product for product in merged if product not in retained]


def merge_readout_indices(layout: SurgeryLayout) -> tuple[int, ...]:
    """Return new-check outcomes whose product is the requested logical parity."""
    return tuple(
        index
        for index, (pauli, _) in enumerate(merge_stabilizers(layout))
        if pauli == layout.basis[0]
    )
