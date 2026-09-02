"""Shared type aliases for surface-code geometry and stabilizer descriptions."""

from typing import TypeAlias


PauliProduct: TypeAlias = tuple[str, tuple[int, ...]]
MixedPauliProduct: TypeAlias = tuple[tuple[str, int], ...]
Coordinates: TypeAlias = dict[int, tuple[float, float]]
