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

import math
import typing
from dataclasses import dataclass, field

import resource_estimation.ftqc.codepatch as cpa


@dataclass(frozen=True)
class Rect:
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rect dimensions must be positive")


class Placeable(typing.Protocol):
    @property
    def footprint(self) -> Rect:  # pragma: no cover
        ...


@dataclass(frozen=True)
class Placement:
    element: Placeable
    row: int
    col: int

    @property
    def row_end(self) -> int:
        return self.row + self.element.footprint.rows

    @property
    def col_end(self) -> int:
        return self.col + self.element.footprint.cols

    def overlaps(self, placement: Placement) -> bool:
        # TODO: This might have a bug because I think there should be and statements in here
        return not (
            placement.row_end <= self.row
            or self.row_end <= placement.row
            or placement.col_end <= self.col
            or self.col_end <= placement.col
        )

    def borders(self, other: Placement) -> bool:
        # For now only handle cases where the objects have the same shape
        if other.element.footprint != self.element.footprint:
            raise ValueError("Both placements must have the same footprint")
        return (
            (self.row_end == other.row or self.row == other.row_end)
            and self.col < other.col_end
            and other.col < self.col_end
        ) or (
            (self.col_end == other.col or self.col == other.col_end)
            and self.row < other.row_end
            and other.row < self.row_end
        )


@dataclass
class Zone:
    rows: int
    cols: int
    role: str

    @property
    def footprint(self) -> Rect:
        return Rect(self.rows, self.cols)

    # Hopefully useful later
    # def contains(self, placement: Placement) -> bool:
    #     return (
    #         0 <= placement.row
    #         and placement.row_end <= self.rows
    #         and 0 <= placement.col
    #         and placement.col_end <= self.cols
    #     )


@dataclass
class Layout:
    rows: int
    cols: int
    placements: dict[int, Placement] = field(default_factory=dict)
    regions: dict[int, Placement] = field(default_factory=dict)
    site_spacing: int = 4  # um between grid locations

    # Might want a method that checks no overlapping placements or something similar

    def add(self, element: Placeable, row: int, col: int) -> int:
        """Adds a new proposed placement if possible, returning its unique identifier"""
        candidate_placement = Placement(element=element, row=row, col=col)
        if not self.in_bounds(placement=candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement):
            raise ValueError("This space is already occupied")
        next_id = max(self.placements, default=0) + 1
        self.placements[next_id] = candidate_placement
        return next_id

    def remove(self, placement_id: int) -> None:
        """Removes the Placement at placement_id"""
        if placement_id not in self.placements:
            raise ValueError(f"No entry found for placement id {placement_id}")
        del self.placements[placement_id]

    def move(self, placement_id: int, row: int, col: int) -> None:
        """Moves the anchor of a Placement with placement_id to the given row, col coordinate"""
        prior_placement = self.placements[placement_id]
        candidate_placement = Placement(element=prior_placement.element, row=row, col=col)
        if not self.in_bounds(candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement, exclude=placement_id):
            raise ValueError("This space is already occupied")
        self.placements[placement_id] = candidate_placement

    def global_coord(self, placement_id: int, local_coord: tuple[int, int]) -> tuple[int, int]:
        """Returns global coordinate for a local object within the Placement with the given placement_id"""
        placement = self.placements[placement_id]
        r, c = local_coord
        return placement.row + r, placement.col + c

    def occupied(self, placement: Placement, exclude: int | None = None) -> bool:
        """Returns True if the proposed Placement overlaps with any existing placement, subject to excluding one placement"""
        return any(
            placement.overlaps(prior_placement)
            for pid, prior_placement in self.placements.items()
            if pid != exclude
        )

    def in_bounds(self, placement: Placement) -> bool:
        """Returns True if the proposed placement is within the bounds of the Layout"""
        return (
            placement.row >= 0
            and placement.col >= 0
            and placement.row_end <= self.rows
            and placement.col_end <= self.cols
        )

    def get_patch(self, patch_id: int) -> cpa.CodePatch:
        for placement in self.placements.values():
            element = placement.element
            if isinstance(element, cpa.CodePatch) and element.patch_id == patch_id:
                return element
        raise ValueError(f"Found no CodePatch with patch_id={patch_id} in Layout")


def generate_movement_layout(
    surface_code_program_qubits: int,
    surface_code_distance: int,
    t_cultivation_factories: int,
    t_distillation_factories: int,
    ccz_distillation_factories: int,
    interaction_zone: bool,
    measure_zone: bool,
    **kwargs: object,
) -> Layout:
    """Generate a populated `Layout` based on the input circuit and other important parameters"""
    total_surface_code_patches = (
        surface_code_program_qubits  # Logical qubits required by the circuit
        + t_cultivation_factories  # Patches dedicated to T state cultivation
        + 31 * t_distillation_factories  # 15 ancilla + 15 Cult + 1 Out
        + 23 * ccz_distillation_factories  # 8 Cult + 12 Ancilla + 3 Out
    )
    patch_grid_length = math.ceil(math.sqrt(total_surface_code_patches))
    patch_length = 2 * surface_code_distance - 1
    required_rows = math.ceil(total_surface_code_patches / patch_grid_length) * patch_length
    required_cols = patch_grid_length * patch_length
    zone_offset = 0
    zones = {}
    if interaction_zone:
        required_rows += patch_length
        zone_offset = patch_length
        zones[0] = Placement(
            element=Zone(rows=patch_length, cols=required_cols, role="Interaction"),
            row=0,
            col=0,
        )
    if measure_zone:
        required_rows += patch_length
        zones[1] = Placement(
            Zone(rows=patch_length, cols=required_cols, role="Measure"),
            row=required_rows - patch_length,
            col=0,
        )
    placements = {}
    for pid in range(total_surface_code_patches):
        patch_row = pid // patch_grid_length
        patch_col = pid % patch_grid_length
        anchor_row = patch_length * patch_row + zone_offset
        anchor_col = patch_length * patch_col
        patch_to_add = cpa.RotatedSurfaceCodePatch(pid, surface_code_distance)
        placements[pid] = Placement(patch_to_add, anchor_row, anchor_col)
    return Layout(rows=required_rows, cols=required_cols, placements=placements, regions=zones)


def generate_column_layout(
    surface_code_program_qubits: int,
    surface_code_distance: int,
    **kwargs: object,
) -> Layout:
    """Generate a populated `Layout` based on the input circuit and other important parameters"""
    rows_of_patches = surface_code_program_qubits + surface_code_program_qubits % 2
    cols_of_patches = 7
    # total_surface_code_patches = rows_of_patches * cols_of_patches
    patch_length = 2 * surface_code_distance - 1
    required_rows = rows_of_patches * patch_length
    required_cols = cols_of_patches * patch_length
    placements = {}
    pid = 0
    for patch_row in range(rows_of_patches):
        for patch_col in range(cols_of_patches):
            anchor_row = patch_row * patch_length
            anchor_col = patch_col * patch_length
            placements[pid] = Placement(
                cpa.RotatedSurfaceCodePatch(pid, surface_code_distance),
                row=anchor_row,
                col=anchor_col,
            )
            pid += 1
    return Layout(
        rows=required_rows,
        cols=required_cols,
        placements=placements,
    )
