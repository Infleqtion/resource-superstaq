"""Geometry and DEQ type metadata for a rotated planar surface-code patch."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import PauliProduct

WireMap = Callable[[int, int], int]


@dataclass(frozen=True)
class RotatedSurfaceCode:
    """A one-logical-qubit rotated planar patch with explicit rectangular shape.

    DEQ needs each port type to have a fixed stabilizer group, so an emitted
    type is named by its dimensions (for example ``RotatedSurfaceCodeW7H3``).
    The *library* does not force the patch to be square or have odd dimensions:
    the transversal-Hadamard protocol temporarily uses a d-by-2d patch, while
    lattice surgery uses d-by-(2d+1) patches.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("rotated planar patches require width and height of at least 3")

    @property
    def num_data_qubits(self) -> int:
        return self.width * self.height

    @property
    def type_name(self) -> str:
        return f"RotatedSurfaceCodeW{self.width}H{self.height}"

    def canonical_wire(self, x: int, y: int) -> int:
        return y * self.width + x

    def stabilizers(self, wire_at: WireMap | None = None) -> list[PauliProduct]:
        """Return the fixed CSS stabilizer group in the requested wire layout."""
        wire = wire_at or self.canonical_wire
        products: list[PauliProduct] = []
        # This is Stim's rotated-memory checkerboard convention. Data are at
        # odd coordinates (2x+1, 2y+1) in Stim's native layout and each
        # four-body check is centered at an even coordinate (2x+2, 2y+2).
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                products.append((
                    "Z" if (x + y) % 2 == 0 else "X",
                    (wire(x, y), wire(x + 1, y), wire(x, y + 1), wire(x + 1, y + 1)),
                ))
        # Stim places X boundaries along the top/bottom and Z boundaries on
        # the left/right, with the stagger shown by its rotated-memory tasks.
        for x in range(0, self.width - 1, 2):
            products.append(("X", (wire(x, 0), wire(x + 1, 0))))
        top_start = 1 if self.height % 2 else 0
        for x in range(top_start, self.width - 1, 2):
            products.append(("X", (wire(x, self.height - 1), wire(x + 1, self.height - 1))))
        for y in range(1, self.height - 1, 2):
            products.append(("Z", (wire(0, y), wire(0, y + 1))))
        for y in range(0, self.height - 1, 2):
            products.append(("Z", (wire(self.width - 1, y), wire(self.width - 1, y + 1))))
        return products

    def boundary_stabilizers(self, side: str, wire_at: WireMap | None = None) -> list[PauliProduct]:
        """Return the staggered two-body boundary checks on one named side."""
        wire = wire_at or self.canonical_wire
        if side == "top":
            return [("X", (wire(x, 0), wire(x + 1, 0))) for x in range(0, self.width - 1, 2)]
        if side == "bottom":
            top_start = 1 if self.height % 2 else 0
            return [
                ("X", (wire(x, self.height - 1), wire(x + 1, self.height - 1)))
                for x in range(top_start, self.width - 1, 2)
            ]
        if side == "left":
            return [("Z", (wire(0, y), wire(0, y + 1))) for y in range(1, self.height - 1, 2)]
        if side == "right":
            return [
                ("Z", (wire(self.width - 1, y), wire(self.width - 1, y + 1)))
                for y in range(0, self.height - 1, 2)
            ]
        raise ValueError("side must be top, bottom, left, or right")

    def logical_x(self, wire_at: WireMap | None = None) -> str:
        wire = wire_at or self.canonical_wire
        return "*".join(f"X{wire(0, y)}" for y in range(self.height))

    def logical_z(self, wire_at: WireMap | None = None) -> str:
        wire = wire_at or self.canonical_wire
        return "*".join(f"Z{wire(x, 0)}" for x in range(self.width))
