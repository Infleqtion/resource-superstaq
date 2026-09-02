"""Physical schedules for encoded logical ``|+i>`` preparation."""

from __future__ import annotations

from .deq_text import code_text, generated_document, indent, targets, validate_distance
from .rotated_surface_code import RotatedSurfaceCode
from .schedules import check_ancilla_coordinates, cnot_syndrome_schedule
from .visualization import circuit_from_schedule
from .y_basis import YBoundarySurfaceCode, y_transition_inverse_schedule


# The initial XXZZ round is the inverse of Gidney's final boundary-readout
# round. Its hook directions therefore differ from an ordinary forward
# surface-code round. Coordinates use +y downward.
_GIDNEY_INVERSE_BOUNDARY_DIRECTIONS = {
    "X": ((-1, 1), (1, 1), (-1, -1), (1, -1)),
    "Z": ((-1, 1), (-1, -1), (1, 1), (1, -1)),
}


def _boundary_round(
    patch: YBoundarySurfaceCode,
) -> tuple[list[str], list[int], list[int]]:
    """Return the boundary round on wires 1..d²-1, reserving wire 0 for Y."""
    data_coordinates = {
        wire + 1: coordinate for wire, coordinate in patch.data_coordinates().items()
    }
    stabilizers = [
        (pauli, tuple(wire + 1 for wire in support))
        for pauli, support in patch.stabilizers()
    ]
    schedule, _ = cnot_syndrome_schedule(
        stabilizers,
        ancilla_offset=2 * (patch.distance**2),
        data_coordinates=data_coordinates,
        cnot_directions=_GIDNEY_INVERSE_BOUNDARY_DIRECTIONS,
    )
    z_targets: list[int] = []
    x_targets: list[int] = []
    for wire, (x, y) in patch.data_coordinates().items():
        target = wire + 1
        (z_targets if patch.initial_basis(int(x), int(y)) == "Z" else x_targets).append(
            target
        )
    return schedule, z_targets, x_targets


def prepare_y_gadget_text(distance: int, *, boundary_rounds: int | None) -> str:
    """Render logical ``|+i>`` preparation as one ordinary-patch gadget."""
    validate_distance(distance)
    boundary = YBoundarySurfaceCode(distance)
    minimum_rounds = distance // 2
    if boundary_rounds is None:
        boundary_rounds = minimum_rounds
    if boundary_rounds < minimum_rounds:
        raise ValueError(
            f"PrepareY requires at least {minimum_rounds} XXZZ boundary rounds"
        )

    patch = RotatedSurfaceCode(distance, distance)
    boundary_round, z_targets, x_targets = _boundary_round(boundary)
    transition, _ = y_transition_inverse_schedule(distance)
    return "\n".join(
        (
            "# Encoded |+i> preparation via Gidney's reverse-time diagonal twist.",
            "# The XXZZ-boundary state is internal; wire 0 is restored during the transition.",
            "GADGET PrepareY {",
            "    # 1. Prepare the missing-corner XXZZ boundary state.",
            f"    R {targets(z_targets)}",
            f"    RX {targets(x_targets)}",
            indent(boundary_round),
            "    # 2. Repeat XXZZ-boundary syndrome extraction for fault tolerance.",
            f"    REPEAT {boundary_rounds} {{",
            indent(boundary_round, spaces=8),
            "    }",
            "    # 3. Restore the corner and map to the ordinary rotated patch.",
            "    #    The terminal record-controlled CX operations fix the +Y frame.",
            indent(transition),
            f"    OUTPUT {patch.type_name} {targets(range(patch.num_data_qubits))}",
            "}",
        )
    )


def y_boundary_circuits(distance: int) -> dict[str, object]:
    """Return Stim views of the internal boundary and transition schedules."""
    validate_distance(distance)
    patch = YBoundarySurfaceCode(distance)
    coordinates = patch.data_coordinates()
    stabilizers = patch.stabilizers()
    schedule, _ = cnot_syndrome_schedule(
        stabilizers,
        ancilla_offset=patch.num_data_qubits,
        data_coordinates=coordinates,
        include_ticks=True,
        cnot_directions=_GIDNEY_INVERSE_BOUNDARY_DIRECTIONS,
    )
    all_coordinates = check_ancilla_coordinates(
        coordinates, stabilizers, ancilla_offset=patch.num_data_qubits
    )
    z_targets = targets(
        [
            wire
            for wire, (x, y) in coordinates.items()
            if patch.initial_basis(int(x), int(y)) == "Z"
        ]
    )
    x_targets = targets(
        [
            wire
            for wire, (x, y) in coordinates.items()
            if patch.initial_basis(int(x), int(y)) == "X"
        ]
    )
    transition_schedule, transition_coordinates = y_transition_inverse_schedule(
        distance, include_ticks=True
    )
    return {
        "PrepareYBoundary": circuit_from_schedule(
            [f"R {z_targets}", f"RX {x_targets}", *schedule],
            coordinates=all_coordinates,
        ),
        "YBoundarySyndromeExtraction": circuit_from_schedule(
            schedule, coordinates=all_coordinates
        ),
        "YBoundaryToRotated": circuit_from_schedule(
            [line for line in transition_schedule if not line.startswith("CX rec[")],
            coordinates=transition_coordinates,
        ),
    }


def render_prepare_y(distance: int = 3, *, boundary_rounds: int | None = None) -> str:
    """Return the one-gadget logical ``|+i>`` preparation library."""
    validate_distance(distance)
    patch = RotatedSurfaceCode(distance, distance)
    return generated_document(
        code_text(
            patch.type_name,
            patch.num_data_qubits,
            patch.logical_x(),
            patch.logical_z(),
            patch.stabilizers(),
            distance=distance,
        ),
        prepare_y_gadget_text(distance, boundary_rounds=boundary_rounds),
    )
