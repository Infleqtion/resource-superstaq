"""Intermediate geometries for the transversal-H patch deformation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .types import MixedPauliProduct


def _extension_checks(
    distance: int,
    wire: Callable[[int, int], int],
    *,
    missing_corner: bool,
) -> list[MixedPauliProduct]:
    """Return the Fig. 2(c) checks, optionally including its final corner."""
    width = 2 * distance
    omitted_site = (width - 1, distance - 1) if missing_corner else None
    checks: list[MixedPauliProduct] = []

    def add(pauli: str, support: tuple[tuple[int, int], ...]) -> None:
        check = tuple(
            (pauli, wire(x, y)) for x, y in support if (x, y) != omitted_site
        )
        if len(check) >= 2:
            checks.append(check)

    for y in range(distance - 1):
        for x in range(width - 1):
            add(
                "X" if (x + y) % 2 == 0 else "Z",
                ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)),
            )

    for x in range(0, distance - 1, 2):
        add("Z", ((x, 0), (x + 1, 0)))
        add("X", ((distance + x, 0), (distance + x + 1, 0)))
        add("Z", ((distance + x, distance - 1), (distance + x + 1, distance - 1)))
    for x in range(1, distance - 1, 2):
        add("Z", ((x, distance - 1), (x + 1, distance - 1)))
    for y in range(1, distance - 1, 2):
        add("X", ((0, y), (0, y + 1)))
    for y in range(0, distance - 1, 2):
        add("Z", ((width - 1, y), (width - 1, y + 1)))
    return checks


def _corner_moved_checks(
    distance: int, wire: Callable[[int, int], int]
) -> list[MixedPauliProduct]:
    """Return Fig. 2(d), where the lower boundary's Z checks become X checks."""
    width = 2 * distance
    old_bottom_supports = {
        frozenset((wire(x, distance - 1), wire(x + 1, distance - 1)))
        for x in range(1, width - 1, 2)
    }
    checks = [
        check
        for check in _extension_checks(distance, wire, missing_corner=False)
        if frozenset(qubit for _, qubit in check) not in old_bottom_supports
    ]
    checks.extend(
        (("X", wire(x, distance - 1)), ("X", wire(x + 1, distance - 1)))
        for x in range(0, width - 1, 2)
    )
    return checks


@dataclass(frozen=True)
class HadamardDeformationLayout:
    """One intermediate rightward H-deformation layout.

    ``stage="extension"`` is Fig. 2(c), with its bottom-right data site
    absent. ``stage="corner_moved"`` is Fig. 2(d), after that site is added
    and the lower boundary has been changed.
    """

    distance: int
    stage: Literal["extension", "corner_moved"]

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be odd and at least 3")
        if self.stage not in ("extension", "corner_moved"):
            raise ValueError("stage must be extension or corner_moved")

    @property
    def width(self) -> int:
        return 2 * self.distance

    @property
    def height(self) -> int:
        return self.distance

    @property
    def has_missing_corner(self) -> bool:
        return self.stage == "extension"

    @property
    def num_data_qubits(self) -> int:
        return self.width * self.height - int(self.has_missing_corner)

    def wire(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("coordinate is outside the deformation")
        if self.has_missing_corner and (x, y) == (self.width - 1, self.height - 1):
            raise ValueError("the bottom-right site is an auxiliary, not data")
        if x < self.distance:
            return y * self.distance + x
        return self.distance * self.distance + y * self.distance + x - self.distance

    def data_coordinates(self) -> dict[int, tuple[float, float]]:
        return {
            self.wire(x, y): (x + 0.5, y + 0.5)
            for y in range(self.height)
            for x in range(self.width)
            if not (self.has_missing_corner and (x, y) == (self.width - 1, self.height - 1))
        }

    def syndrome_checks(self) -> list[MixedPauliProduct]:
        if self.has_missing_corner:
            return _extension_checks(self.distance, self.wire, missing_corner=True)
        return _corner_moved_checks(self.distance, self.wire)
