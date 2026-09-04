"""DEQ rendering and schedules for the staged logical-H deformation."""

from __future__ import annotations

from .deq_text import code_text, document, indent, targets
from .hadamard import HadamardDeformationLayout
from .rotated_surface_code import RotatedSurfaceCode
from .schedules import (
    check_ancilla_coordinates,
    cnot_syndrome_schedule,
    mixed_syndrome_schedule,
)


def swap_qec_schedule(
    distance: int,
    direction: str,
    *,
    include_ticks: bool,
) -> tuple[list[str], dict[int, tuple[float, float]]]:
    """Move data by one native SWAP layer followed by a syndrome round."""
    patch = RotatedSurfaceCode(distance, distance)
    patch_size = patch.num_data_qubits
    if direction == "NW":
        input_coordinates = {
            y * distance + x: (x + 1.5, y + 0.5)
            for y in range(distance)
            for x in range(distance)
        }
        data_coordinates = {
            patch_size + y * distance + x: (x + 1.0, y + 1.0)
            for y in range(distance)
            for x in range(distance)
        }
    elif direction == "SW":
        input_coordinates = {
            y * distance + x: (x + 1.0, y + 1.0)
            for y in range(distance)
            for x in range(distance)
        }
        data_coordinates = {
            patch_size + y * distance + x: (x + 0.5, y + 0.5)
            for y in range(distance)
            for x in range(distance)
        }
    else:
        raise ValueError(f"unknown SWAP-QEC direction: {direction}")
    shifted_stabilizers = patch.stabilizers(lambda x, y: patch_size + y * distance + x)
    qec_schedule, _ = cnot_syndrome_schedule(
        shifted_stabilizers,
        ancilla_offset=2 * patch_size,
        data_coordinates=data_coordinates,
        include_ticks=include_ticks,
    )
    qec_coordinates = check_ancilla_coordinates(
        data_coordinates,
        shifted_stabilizers,
        ancilla_offset=2 * patch_size,
    )
    input_at_coordinate = {
        coordinate: qubit for qubit, coordinate in input_coordinates.items()
    }
    reused_ancillas = {
        2 * patch_size + index: input_at_coordinate[coordinate]
        for index in range(len(shifted_stabilizers))
        if (coordinate := qec_coordinates[2 * patch_size + index])
        in input_at_coordinate
    }

    def remap_reused_ancillas(lines: list[str]) -> list[str]:
        remapped: list[str] = []
        for line in lines:
            if line == "TICK":
                remapped.append(line)
                continue
            operation, *line_targets = line.split()
            remapped.append(
                " ".join(
                    (
                        operation,
                        *(
                            str(reused_ancillas.get(int(target), int(target)))
                            for target in line_targets
                        ),
                    )
                )
            )
        return remapped

    qec_schedule = remap_reused_ancillas(qec_schedule)
    new_targets = targets(range(patch_size, 2 * patch_size))
    swap_pairs = targets(
        [qubit for index in range(patch_size) for qubit in (index, patch_size + index)]
    )
    coordinates = dict(input_coordinates)
    coordinates.update(data_coordinates)
    coordinates.update(
        {
            qubit: coordinate
            for qubit, coordinate in qec_coordinates.items()
            if qubit < 2 * patch_size or qubit not in reused_ancillas
        }
    )
    schedule = [f"R {new_targets}"]
    if include_ticks:
        schedule.append("TICK")
    schedule.append(f"SWAP {swap_pairs}")
    if include_ticks:
        schedule.append("TICK")
    schedule.extend(qec_schedule)
    return schedule, coordinates


def _remap_schedule(schedule: list[str], wire_map: dict[int, int]) -> list[str]:
    """Rename physical wires in a schedule while preserving its layers."""
    remapped: list[str] = []
    for line in schedule:
        if line == "TICK":
            remapped.append(line)
            continue
        operation, *line_targets = line.split()
        remapped.append(
            " ".join(
                (
                    operation,
                    *(
                        str(wire_map.get(int(target), int(target)))
                        for target in line_targets
                    ),
                )
            )
        )
    return remapped


def _extension_hook_orders(
    layout: HadamardDeformationLayout,
) -> dict[int, dict[str, tuple[tuple[int, int], ...]]]:
    """Return the Figure 5(b) order for the rightward extension layout.

    The published patch is vertical. Rotating it into this horizontal layout
    sends its four CNOT layers to the two orders below. Transversal H swaps
    the X and Z orders in region A; crossing the diagonal region-B boundary
    swaps them once more. The staggered boundary is one lattice site apart for
    X and Z checks.
    """
    if layout.stage != "extension":
        raise ValueError("Figure 5(b) orders apply only to the extension layout")
    region_a = {
        "X": ((-1, 1), (-1, -1), (1, 1), (1, -1)),
        "Z": ((-1, 1), (1, 1), (-1, -1), (1, -1)),
    }
    region_b = {"X": region_a["Z"], "Z": region_a["X"]}
    checks = layout.syndrome_checks()
    data_size = len(layout.data_coordinates())
    coordinates = check_ancilla_coordinates(
        layout.data_coordinates(),
        [("X", tuple(qubit for _, qubit in check)) for check in checks],
        ancilla_offset=data_size,
    )
    orders = {}
    for index, check in enumerate(checks):
        basis = check[0][0]
        x, y = coordinates[data_size + index]
        in_region_b = x - y >= layout.distance + (basis == "X")
        orders[index] = region_b if in_region_b else region_a
    return orders


def _corner_movement_hook_orders(
    layout: HadamardDeformationLayout,
) -> dict[int, dict[str, tuple[tuple[int, int], ...]]]:
    """Return the rotated Figure 5(c) order for the corner-movement layout.

    This is the next H-deformation round after the extension. Its A/B boundary
    runs along the opposite diagonal in this rightward layout, so the local
    schedule changes with ``x + y`` instead of ``x - y``.
    """
    if layout.stage != "corner_moved":
        raise ValueError("Figure 5(c) orders apply only to the corner-movement layout")
    region_a = {
        "X": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
        "Z": ((-1, -1), (1, -1), (-1, 1), (1, 1)),
    }
    region_b = {"X": region_a["Z"], "Z": region_a["X"]}
    checks = layout.syndrome_checks()
    data_size = len(layout.data_coordinates())
    coordinates = check_ancilla_coordinates(
        layout.data_coordinates(),
        [("X", tuple(qubit for _, qubit in check)) for check in checks],
        ancilla_offset=data_size,
    )
    orders = {}
    for index, check in enumerate(checks):
        basis = check[0][0]
        x, y = coordinates[data_size + index]
        in_region_a = x + y <= layout.distance - 1 - (basis == "Z")
        orders[index] = region_a if in_region_a else region_b
    return orders


def _layout_stabilizers(
    layout: HadamardDeformationLayout,
) -> list[tuple[str, tuple[int, ...]]]:
    """Convert one mixed-layout check list into DEQ's CSS-code form."""
    return [
        (check[0][0], tuple(qubit for _, qubit in check))
        for check in layout.syndrome_checks()
    ]


def _hadamard_code_text(
    distance: int,
    extension: HadamardDeformationLayout,
    corner: HadamardDeformationLayout,
) -> str:
    """Render the four transient code types used by the H deformation.

    Each logical representative is chosen on the physical boundary that the
    following stage preserves.  DEQ can therefore derive the measurement
    frame directly from each typed boundary, without a ``PROPAGATE`` record.
    """
    patch = RotatedSurfaceCode(distance, distance)
    return_patch = RotatedSurfaceCode(2 * distance - 1, distance)
    frame_stabilizers = [
        ("Z" if pauli == "X" else "X", support)
        for pauli, support in patch.stabilizers()
    ]
    corner_logical_z = (
        corner.wire(distance - 1, 0),
        *(corner.wire(x, 0) for x in range(distance, 2 * distance)),
    )
    return document(
        code_text(
            f"HadamardFrameD{distance}",
            patch.num_data_qubits,
            "*".join(f"X{x}" for x in range(distance)),
            "*".join(f"Z{distance * y}" for y in range(distance)),
            frame_stabilizers,
            distance=distance,
        ),
        code_text(
            f"HadamardExtensionD{distance}",
            extension.num_data_qubits,
            "*".join(f"X{extension.wire(x, 0)}" for x in range(distance)),
            "*".join(f"Z{extension.wire(0, y)}" for y in range(distance)),
            _layout_stabilizers(extension),
            distance=distance,
        ),
        code_text(
            f"HadamardCornerD{distance}",
            corner.num_data_qubits,
            "*".join(f"X{corner.wire(distance, y)}" for y in range(distance)),
            "*".join(f"Z{qubit}" for qubit in corner_logical_z),
            _layout_stabilizers(corner),
            distance=distance,
        ),
        code_text(
            f"HadamardReturnD{distance}",
            return_patch.num_data_qubits,
            return_patch.logical_x(),
            return_patch.logical_z(),
            return_patch.stabilizers(),
            distance=distance,
        ),
    )


def _hadamard_stage_text(
    name: str,
    input_code: str,
    input_targets: list[int],
    body: list[str],
    output_code: str,
    output_targets: list[int],
) -> str:
    """Render one endpoint-typed H-deformation stage."""
    return "\n".join(
        (
            f"GADGET {name} {{",
            f"    INPUT {input_code} {targets(input_targets)}",
            indent(body),
            f"    OUTPUT {output_code} {targets(output_targets)}",
            "}",
        )
    )


def logical_hadamard_gadget_text(distance: int) -> str:
    """Render the complete H deformation as typed, composable DEQ stages."""
    patch = RotatedSurfaceCode(distance, distance)
    patch_size = patch.num_data_qubits
    extension = HadamardDeformationLayout(distance, "extension")
    domain_wall = HadamardDeformationLayout(distance, "corner_moved")
    patch_targets = list(range(patch_size))

    extension_round = mixed_syndrome_schedule(
        extension.syndrome_checks(),
        ancilla_offset=2 * patch_size - 1,
        data_coordinates=extension.data_coordinates(),
        include_ticks=False,
        cnot_directions_by_check=_extension_hook_orders(extension),
    )
    domain_wall_round = mixed_syndrome_schedule(
        domain_wall.syndrome_checks(),
        ancilla_offset=2 * patch_size,
        data_coordinates=domain_wall.data_coordinates(),
        include_ticks=False,
        cnot_directions_by_check=_corner_movement_hook_orders(domain_wall),
    )
    shrink_round, _ = cnot_syndrome_schedule(
        patch.stabilizers(lambda x, y: domain_wall.wire(distance + x, y)),
        ancilla_offset=2 * patch_size,
        data_coordinates={
            domain_wall.wire(distance + x, y): (distance + x + 0.5, y + 0.5)
            for y in range(distance)
            for x in range(distance)
        },
        include_ticks=False,
    )
    return_width = 2 * distance - 1

    def return_wire(x: int, y: int) -> int:
        if x < distance - 1:
            return y * (distance - 1) + x
        return patch_size + y * distance + x - (distance - 1)

    return_patch = RotatedSurfaceCode(return_width, distance)
    return_extension_round, _ = cnot_syndrome_schedule(
        return_patch.stabilizers(return_wire),
        ancilla_offset=2 * patch_size,
        data_coordinates={
            return_wire(x, y): (x + 1.5, y + 0.5)
            for y in range(distance)
            for x in range(return_width)
        },
        include_ticks=False,
    )
    return_shrink_round, _ = cnot_syndrome_schedule(
        patch.stabilizers(return_wire),
        ancilla_offset=2 * patch_size,
        data_coordinates={
            return_wire(x, y): (x + 1.5, y + 0.5)
            for y in range(distance)
            for x in range(distance)
        },
        include_ticks=False,
    )
    retained_targets = [
        return_wire(x, y) for y in range(distance) for x in range(distance)
    ]
    return_added_targets = [
        return_wire(x, y) for y in range(distance) for x in range(distance - 1)
    ]
    return_removed_targets = [
        return_wire(x, y)
        for y in range(distance)
        for x in range(distance, return_width)
    ]
    northwest_schedule, _ = swap_qec_schedule(distance, "NW", include_ticks=False)
    northwest_wires = {
        **{index: qubit for index, qubit in enumerate(retained_targets)},
        **{patch_size + index: 2 * patch_size + index for index in range(patch_size)},
        **{
            2 * patch_size + index: 3 * patch_size + index
            for index in range(patch_size)
        },
    }
    southwest_schedule, _ = swap_qec_schedule(distance, "SW", include_ticks=False)
    southwest_wires = {
        **{index: 2 * patch_size + index for index in range(patch_size)},
        **{patch_size + index: 3 * patch_size + index for index in range(patch_size)},
        **{
            2 * patch_size + index: 4 * patch_size + index
            for index in range(patch_size)
        },
    }
    frame_code = f"HadamardFrameD{distance}"
    extension_code = f"HadamardExtensionD{distance}"
    corner_code = f"HadamardCornerD{distance}"
    return_code = f"HadamardReturnD{distance}"
    extension_targets = list(range(2 * patch_size - 1))
    corner_targets = list(range(2 * patch_size))
    shifted_patch_targets = list(range(patch_size, 2 * patch_size))
    return_targets = [
        return_wire(x, y) for y in range(distance) for x in range(return_width)
    ]

    stages = (
        _hadamard_stage_text(
            f"TransversalHadamardD{distance}",
            patch.type_name,
            patch_targets,
            [
                "# Change to the H frame without changing the patch footprint.",
                f"H {targets(patch_targets)}",
            ],
            frame_code,
            patch_targets,
        ),
        _hadamard_stage_text(
            f"HadamardExtendD{distance}",
            frame_code,
            patch_targets,
            [
                "# Add the right half in |+> and establish the extension checks.",
                f"RX {targets(list(range(patch_size, 2 * patch_size - 1)))}",
                *extension_round,
            ],
            extension_code,
            extension_targets,
        ),
        _hadamard_stage_text(
            f"HadamardExtensionSED{distance}",
            extension_code,
            extension_targets,
            [
                "# One additional extraction round on the extension layout.",
                *extension_round,
            ],
            extension_code,
            extension_targets,
        ),
        _hadamard_stage_text(
            f"HadamardCornerMoveD{distance}",
            extension_code,
            extension_targets,
            [
                "# Complete the corner and move the lower boundary.",
                f"R {2 * patch_size - 1}",
                *domain_wall_round,
            ],
            corner_code,
            corner_targets,
        ),
        _hadamard_stage_text(
            f"HadamardDomainWallSED{distance}",
            corner_code,
            corner_targets,
            [
                "# One additional extraction round on the domain-wall layout.",
                *domain_wall_round,
            ],
            corner_code,
            corner_targets,
        ),
        _hadamard_stage_text(
            f"HadamardShrinkD{distance}",
            corner_code,
            corner_targets,
            [
                "# Remove the H-frame half and retain the right ordinary patch.",
                f"M {targets(patch_targets)}",
                *shrink_round,
            ],
            patch.type_name,
            shifted_patch_targets,
        ),
        _hadamard_stage_text(
            f"HadamardReturnExtendD{distance}",
            patch.type_name,
            shifted_patch_targets,
            [
                "# Extend left and establish the rectangular return patch.",
                f"R {targets(return_added_targets)}",
                *return_extension_round,
            ],
            return_code,
            return_targets,
        ),
        _hadamard_stage_text(
            f"HadamardReturnExtensionSED{distance}",
            return_code,
            return_targets,
            [
                "# One additional extraction round on the return layout.",
                *return_extension_round,
            ],
            return_code,
            return_targets,
        ),
        _hadamard_stage_text(
            f"HadamardReturnShrinkD{distance}",
            return_code,
            return_targets,
            [
                "# Remove the excess columns and restore a square ordinary patch.",
                f"M {targets(return_removed_targets)}",
                *return_shrink_round,
            ],
            patch.type_name,
            retained_targets,
        ),
        _hadamard_stage_text(
            f"HadamardSwapQECD{distance}",
            patch.type_name,
            retained_targets,
            [
                "# Two SWAP-QEC translations return to the original footprint.",
                *_remap_schedule(northwest_schedule, northwest_wires),
                *_remap_schedule(southwest_schedule, southwest_wires),
            ],
            patch.type_name,
            list(range(3 * patch_size, 4 * patch_size)),
        ),
    )
    composition = "\n".join(
        (
            "# The transient codes make each deformation boundary explicit.",
            f"COMPOSE LogicalHadamardD{distance} {{",
            f"    INPUT {patch.type_name} 0",
            f"    TransversalHadamardD{distance} IN(0) OUT(1)",
            f"    HadamardExtendD{distance} IN(1) OUT(2)",
            f"    REPEAT {distance - 1} {{",
            f"        HadamardExtensionSED{distance} 2",
            "    }",
            f"    HadamardCornerMoveD{distance} IN(2) OUT(3)",
            f"    REPEAT {distance - 1} {{",
            f"        HadamardDomainWallSED{distance} 3",
            "    }",
            f"    HadamardShrinkD{distance} IN(3) OUT(4)",
            f"    HadamardReturnExtendD{distance} IN(4) OUT(5)",
            f"    REPEAT {distance - 1} {{",
            f"        HadamardReturnExtensionSED{distance} 5",
            "    }",
            f"    HadamardReturnShrinkD{distance} IN(5) OUT(6)",
            f"    HadamardSwapQECD{distance} IN(6) OUT(7)",
            f"    OUTPUT {patch.type_name} 7",
            "}",
        )
    )
    return document(
        _hadamard_code_text(distance, extension, domain_wall), *stages, composition
    )
