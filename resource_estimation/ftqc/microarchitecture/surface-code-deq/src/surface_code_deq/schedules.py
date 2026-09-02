"""Shared physical-operation schedules for generated DEQ and Stim circuits."""

from __future__ import annotations

from collections.abc import Mapping

from .types import Coordinates, MixedPauliProduct, PauliProduct


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
    include_ticks: bool = False,
    cnot_directions: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
) -> tuple[list[str], dict[int, int]]:
    """Return one four-layer syndrome round with the requested hook ordering.

    ``cnot_directions`` gives the data-qubit offset selected by each layer for
    X and Z checks. It defaults to the ordinary rotated-code ordering. Reverse
    time preparations can supply their own ordering when their hook direction
    must match an inverted readout circuit.
    """
    if data_coordinates is None:
        data_count = max(qubit for _, support in stabilizers for qubit in support) + 1
        width = int(data_count**0.5)
        if width * width != data_count:
            raise ValueError("data coordinates are required for non-square layouts")
        data_coordinates = square_data_coordinates(width)

    if cnot_directions is None:
        cnot_directions = _CNOT_DIRECTIONS
    if set(cnot_directions) != {"X", "Z"} or any(
        len(cnot_directions[basis]) != 4 for basis in ("X", "Z")
    ):
        raise ValueError("CNOT directions must give four layers for X and Z checks")

    coordinates = check_ancilla_coordinates(
        data_coordinates, stabilizers, ancilla_offset=ancilla_offset
    )
    x_ancillas = [
        ancilla_offset + index
        for index, (pauli, _) in enumerate(stabilizers)
        if pauli == "X"
    ]
    z_ancillas = [
        ancilla_offset + index
        for index, (pauli, _) in enumerate(stabilizers)
        if pauli == "Z"
    ]
    lines: list[str] = []
    if z_ancillas:
        lines.append("R " + " ".join(map(str, z_ancillas)))
    if x_ancillas:
        lines.append("RX " + " ".join(map(str, x_ancillas)))
    if include_ticks:
        lines.append("TICK")

    for layer_index in range(4):
        targets: list[int] = []
        used_qubits: set[int] = set()
        for check_index, (pauli, support) in enumerate(stabilizers):
            ancilla = ancilla_offset + check_index
            check_x, check_y = coordinates[ancilla]
            direction = cnot_directions[pauli][layer_index]
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
            if data in used_qubits or ancilla in used_qubits:
                raise AssertionError("Stim-style CNOT layer has a qubit collision")
            used_qubits.update((data, ancilla))
            targets.extend((ancilla, data) if pauli == "X" else (data, ancilla))
        lines.append("CX " + " ".join(map(str, targets)))
        if include_ticks:
            lines.append("TICK")

    if z_ancillas:
        lines.append("M " + " ".join(map(str, z_ancillas)))
    if x_ancillas:
        lines.append("MX " + " ".join(map(str, x_ancillas)))
    if include_ticks:
        lines.append("TICK")
    # ``M`` records all Z-check ancillas before ``MX`` records X-check
    # ancillas.  Return the physical-record index for each stabilizer-list
    # index; surgery READOUTs use this map to form a logical parity.
    measurement_index: dict[int, int] = {}
    for measurement, check_index in enumerate(
        index for index, (pauli, _) in enumerate(stabilizers) if pauli == "Z"
    ):
        measurement_index[check_index] = measurement
    z_count = len(measurement_index)
    for measurement, check_index in enumerate(
        index for index, (pauli, _) in enumerate(stabilizers) if pauli == "X"
    ):
        measurement_index[check_index] = z_count + measurement
    return lines, measurement_index


def mixed_syndrome_schedule(
    stabilizers: list[MixedPauliProduct],
    *,
    ancilla_offset: int,
    data_coordinates: Coordinates,
    include_ticks: bool = False,
    cnot_directions_by_check: Mapping[
        int, Mapping[str, tuple[tuple[int, int], ...]]
    ] | None = None,
) -> list[str]:
    """Measure pure checks conventionally and mixed checks with H/CNOT.

    A deformation can supply its own hook-safe CNOT order for individual
    checks. Checks omitted from ``cnot_directions_by_check`` use the ordinary
    rotated-code order.
    """
    supports = [("X", tuple(qubit for _, qubit in check)) for check in stabilizers]
    coordinates = check_ancilla_coordinates(
        data_coordinates, supports, ancilla_offset=ancilla_offset
    )
    ancillas = range(ancilla_offset, ancilla_offset + len(stabilizers))
    pure_z_ancillas = [
        ancilla_offset + index
        for index, check in enumerate(stabilizers)
        if {pauli for pauli, _ in check} == {"Z"}
    ]
    pure_z_ancilla_set = set(pure_z_ancillas)
    x_basis_ancillas = [
        ancilla for ancilla in ancillas if ancilla not in pure_z_ancilla_set
    ]
    check_paulis = [{pauli for pauli, _ in check} for check in stabilizers]
    lines: list[str] = []
    if pure_z_ancillas:
        lines.append("R " + " ".join(map(str, pure_z_ancillas)))
    if x_basis_ancillas:
        lines.append("RX " + " ".join(map(str, x_basis_ancillas)))
    if include_ticks:
        lines.append("TICK")

    for layer_index in range(4):
        interactions: list[tuple[bool, int, int, bool]] = []
        for index, check in enumerate(stabilizers):
            ancilla = ancilla_offset + index
            check_x, check_y = coordinates[ancilla]
            term_by_qubit = {qubit: pauli for pauli, qubit in check}
            paulis = check_paulis[index]
            check_kind = "Z" if paulis == {"Z"} else "X"
            directions = (cnot_directions_by_check or {}).get(
                index, _CNOT_DIRECTIONS
            )
            direction = directions[check_kind][layer_index]
            data = next(
                (
                    qubit
                    for qubit in term_by_qubit
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
            is_mixed = len(paulis) > 1
            interactions.append(
                (is_mixed, ancilla, data, is_mixed and term_by_qubit[data] == "Z")
            )

        batches: list[tuple[list[tuple[bool, int, int, bool]], set[int]]] = []
        for interaction in sorted(interactions, key=lambda item: item[0]):
            _, ancilla, data, _ = interaction
            batch = next(
                (item for item in batches if not {ancilla, data} & item[1]), None
            )
            if batch is None:
                batch = ([], set())
                batches.append(batch)
            batch[0].append(interaction)
            batch[1].update((ancilla, data))

        for batch, _ in batches:
            cx_targets: list[int] = []
            basis_changed_data: list[int] = []
            for _, ancilla, data, basis_change in batch:
                if check_paulis[ancilla - ancilla_offset] == {"Z"}:
                    cx_targets.extend((data, ancilla))
                else:
                    cx_targets.extend((ancilla, data))
                if basis_change:
                    basis_changed_data.append(data)
            if basis_changed_data:
                lines.append("H " + " ".join(map(str, basis_changed_data)))
            lines.append("CX " + " ".join(map(str, cx_targets)))
            if basis_changed_data:
                lines.append("H " + " ".join(map(str, basis_changed_data)))
            if include_ticks:
                lines.append("TICK")
    if pure_z_ancillas:
        lines.append("M " + " ".join(map(str, pure_z_ancillas)))
    if x_basis_ancillas:
        lines.append("MX " + " ".join(map(str, x_basis_ancillas)))
    if include_ticks:
        lines.append("TICK")
    return lines


def parallelize_schedules(*schedules: list[str]) -> list[str]:
    """Fuse equal-time operations from disjoint schedules into one layer."""
    if not schedules:
        return []
    if len({len(schedule) for schedule in schedules}) != 1:
        raise ValueError("parallel schedules must have the same number of layers")
    result: list[str] = []
    for layer in zip(*schedules, strict=True):
        operation = layer[0].split()[0]
        if any(line.split()[0] != operation for line in layer):
            raise ValueError("parallel schedules have incompatible layers")
        if operation == "TICK":
            result.append("TICK")
            continue
        targets = [target for line in layer for target in line.split()[1:]]
        if len(targets) != len(set(targets)):
            raise ValueError("parallel schedules reuse a qubit in one layer")
        result.append(operation + " " + " ".join(targets))
    return result


def with_seam_reset(schedule: list[str], seam_targets: str, basis: str) -> list[str]:
    """Fuse a seam-data reset into the matching first check-reset layer."""
    reset = "R" if basis == "XX" else "RX"
    result = list(schedule)
    for index, line in enumerate(result):
        if line.startswith(reset + " "):
            result[index] = line + " " + seam_targets
            return result
    raise AssertionError(f"merge schedule has no {reset} reset layer")
