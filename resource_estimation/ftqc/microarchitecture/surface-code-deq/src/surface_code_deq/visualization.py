"""Stim circuit construction used by notebooks and gadget visualizations."""

from __future__ import annotations

from .types import Coordinates, PauliProduct


def circuit_from_schedule(lines: list[str], *, coordinates: Coordinates):
    """Build a coordinate-annotated Stim circuit from a shared schedule.

    Importing Stim lazily keeps DEQ source rendering independent of the
    visualization backend.
    """
    import stim

    body = stim.Circuit("\n".join(lines))
    result = stim.Circuit()
    for qubit, (x, y) in sorted(coordinates.items()):
        if qubit < body.num_qubits:
            result.append("QUBIT_COORDS", [qubit], [x, y])
    result += body
    return result


def stim_rotated_patch_coordinates(
    distance: int, stabilizers: list[PauliProduct]
) -> Coordinates:
    """Map Stim's rotated-memory geometry into the project's wire numbering."""
    import stim

    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=1
    )
    reference_coordinates = reference.get_final_qubit_coordinates()
    data_by_stim_qubit = {
        qubit: ((int(x) - 1) // 2) + distance * ((int(y) - 1) // 2)
        for qubit, (x, y) in reference_coordinates.items()
        if int(x) % 2 and int(y) % 2
    }

    x_ancillas: set[int] = set()
    interactions: list[tuple[int, int]] = []
    measured_ancillas: list[int] = []
    for instruction in reference:
        if instruction.name == "H":
            x_ancillas.update(target.value for target in instruction.targets_copy())
        elif instruction.name == "CX":
            targets = [target.value for target in instruction.targets_copy()]
            interactions.extend(zip(targets[::2], targets[1::2]))
        elif instruction.name == "MR":
            measured_ancillas = [target.value for target in instruction.targets_copy()]
            break

    coordinates = {
        data: (
            reference_coordinates[stim_qubit][0] / 2,
            reference_coordinates[stim_qubit][1] / 2,
        )
        for stim_qubit, data in data_by_stim_qubit.items()
    }
    check_coordinates: dict[PauliProduct, tuple[float, float]] = {}
    for ancilla in measured_ancillas:
        support = {
            data_by_stim_qubit[target]
            for control, target in interactions
            if control == ancilla and target in data_by_stim_qubit
        } | {
            data_by_stim_qubit[control]
            for control, target in interactions
            if target == ancilla and control in data_by_stim_qubit
        }
        pauli = "X" if ancilla in x_ancillas else "Z"
        x, y = reference_coordinates[ancilla]
        check_coordinates[(pauli, tuple(sorted(support)))] = (x / 2, y / 2)

    if set(check_coordinates) != set(stabilizers):
        raise AssertionError("DEQ stabilizers do not match Stim's rotated-memory code")
    for ancilla_index, stabilizer in enumerate(stabilizers):
        coordinates[distance * distance + ancilla_index] = check_coordinates[stabilizer]
    return coordinates
