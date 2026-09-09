"""Shared physical-operation schedules for generated DEQ and Stim circuits."""

from __future__ import annotations

from collections.abc import Mapping

from .gadgets.core import Instruction, Schedule
from .types import Coordinates, PauliProduct

_CnotOrder = Mapping[str, tuple[tuple[int, int], ...]]

_CNOT_DIRECTIONS = {
    # Gidney's XZXZ surface-code order. Coordinates use +y downward.
    "X": ((1, -1), (-1, -1), (1, 1), (-1, 1)),
    "Z": ((1, -1), (1, 1), (-1, -1), (-1, 1)),
}


def square_data_coordinates(
    distance: int,
    *,
    qubit_offset: int = 0,
    offset: tuple[float, float] = (0, 0),
) -> Coordinates:
    """Return row-major coordinates for a square rotated-code data patch."""
    offset_x, offset_y = offset
    return {
        qubit_offset + y * distance + x: (x + 0.5 + offset_x, y + 0.5 + offset_y)
        for y in range(distance)
        for x in range(distance)
    }


def check_ancilla_coordinates(
    data_coordinates: Coordinates,
    stabilizers: list[PauliProduct],
    *,
    ancilla_offset: int,
) -> Coordinates:
    """Place Stim-style check ancillas on the integer sublattice."""
    result = dict(data_coordinates)
    data_x = [x for x, _ in data_coordinates.values()]
    data_y = [y for _, y in data_coordinates.values()]
    min_x, max_x = min(data_x), max(data_x)
    min_y, max_y = min(data_y), max(data_y)
    for index, (_, support) in enumerate(stabilizers):
        support_coordinates = [data_coordinates[qubit] for qubit in support]
        support_x_values = [point[0] for point in support_coordinates]
        support_y_values = [point[1] for point in support_coordinates]
        x = (min(support_x_values) + max(support_x_values)) / 2
        y = (min(support_y_values) + max(support_y_values)) / 2
        if len(support) == 2:
            support_x = set(support_x_values)
            support_y = set(support_y_values)
            if len(support_x) == 1:
                x += -0.5 if x == min_x else 0.5 if x == max_x else 0
            elif len(support_y) == 1:
                y += -0.5 if y == min_y else 0.5 if y == max_y else 0
        result[ancilla_offset + index] = (x, y)
    return result


def cnot_syndrome_schedule(
    stabilizers: list[PauliProduct],
    *,
    ancilla_offset: int,
    data_coordinates: Coordinates | None = None,
    cnot_directions: _CnotOrder | None = None,
    cnot_directions_by_check: Mapping[int, _CnotOrder] | None = None,
) -> tuple[Schedule, dict[int, int]]:
    """Return one four-layer syndrome round with the requested hook ordering.

    ``cnot_directions`` gives the data-qubit offset selected by each layer for
    X and Z checks. It defaults to the ordinary rotated-code ordering. Reverse
    time preparations can supply their own ordering when their hook direction
    must match an inverted readout circuit. ``cnot_directions_by_check`` can
    override that ordering for individual checks in a deformation.
    """
    if data_coordinates is None:
        data_count = max(qubit for _, support in stabilizers for qubit in support) + 1
        width = int(data_count**0.5)
        if width * width != data_count:
            raise ValueError("data coordinates are required for non-square layouts")
        data_coordinates = square_data_coordinates(width)

    default_directions = cnot_directions or _CNOT_DIRECTIONS
    all_directions = (default_directions, *(cnot_directions_by_check or {}).values())
    if any(
        set(directions) != {"X", "Z"} or any(len(directions[basis]) != 4 for basis in ("X", "Z"))
        for directions in all_directions
    ):
        raise ValueError("CNOT directions must give four layers for X and Z checks")
    if any(pauli not in ("X", "Z") for pauli, _ in stabilizers):
        raise ValueError("syndrome checks must be uniformly X or Z")

    coordinates = check_ancilla_coordinates(
        data_coordinates, stabilizers, ancilla_offset=ancilla_offset
    )
    z_ancillas = [
        ancilla_offset + index for index, (pauli, _) in enumerate(stabilizers) if pauli == "Z"
    ]
    x_ancillas = [
        ancilla_offset + index for index, (pauli, _) in enumerate(stabilizers) if pauli == "X"
    ]
    instructions: list[Instruction] = []
    if z_ancillas:
        instructions.append(Instruction.on("R", z_ancillas))
    if x_ancillas:
        instructions.append(Instruction.on("RX", x_ancillas))
    instructions.append(Instruction("TICK"))

    for layer_index in range(4):
        interactions: list[tuple[str, int, int]] = []
        for index, (pauli, support) in enumerate(stabilizers):
            ancilla = ancilla_offset + index
            check_x, check_y = coordinates[ancilla]
            directions = (cnot_directions_by_check or {}).get(index, default_directions)
            direction = directions[pauli][layer_index]
            data = next(
                (
                    qubit
                    for qubit in support
                    if (
                        1 if data_coordinates[qubit][0] > check_x else -1,
                        1 if data_coordinates[qubit][1] > check_y else -1,
                    )
                    == direction
                ),
                None,
            )
            if data is None:
                continue
            interactions.append((pauli, ancilla, data))

        batches: list[tuple[list[tuple[str, int, int]], set[int]]] = []
        for interaction in interactions:
            _, ancilla, data = interaction
            batch = next((item for item in batches if not {ancilla, data} & item[1]), None)
            if batch is None:
                batch = ([], set())
                batches.append(batch)
            batch[0].append(interaction)
            batch[1].update((ancilla, data))

        for batch, _ in batches:
            cx_targets: list[int] = []
            for pauli, ancilla, data in batch:
                cx_targets.extend((data, ancilla) if pauli == "Z" else (ancilla, data))
            instructions.append(Instruction.on("CX", cx_targets))
            instructions.append(Instruction("TICK"))
    if z_ancillas:
        instructions.append(Instruction.on("M", z_ancillas))
    if x_ancillas:
        instructions.append(Instruction.on("MX", x_ancillas))
    instructions.append(Instruction("TICK"))

    # ``M`` records all Z-check ancillas before ``MX`` records X-check
    # ancillas. Return each stabilizer's index in that measurement sequence.
    measurement_index = {
        check_index: measurement
        for measurement, check_index in enumerate(
            index for index, (pauli, _) in enumerate(stabilizers) if pauli == "Z"
        )
    }
    z_count = len(measurement_index)
    measurement_index.update(
        {
            check_index: z_count + measurement
            for measurement, check_index in enumerate(
                index for index, (pauli, _) in enumerate(stabilizers) if pauli == "X"
            )
        }
    )
    return Schedule(tuple(instructions)), measurement_index
