"""Shared type aliases for surface-code geometry and stabilizer descriptions."""

from typing import TypeAlias

PauliProduct: TypeAlias = tuple[str, tuple[int, ...]]
Coordinates: TypeAlias = dict[int, tuple[float, float]]
