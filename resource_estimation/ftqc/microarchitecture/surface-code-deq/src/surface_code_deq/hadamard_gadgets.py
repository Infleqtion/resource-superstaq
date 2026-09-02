"""DEQ rendering and schedules for the monolithic logical-H gadget."""

from __future__ import annotations

from .deq_text import indent, targets
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
    shifted_stabilizers = patch.stabilizers(
        lambda x, y: patch_size + y * distance + x
    )
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
        [
            qubit
            for index in range(patch_size)
            for qubit in (index, patch_size + index)
        ]
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
                (operation, *(str(wire_map.get(int(target), int(target))) for target in line_targets))
            )
        )
    return remapped


def _repeat_schedule(count: int, schedule: list[str]) -> list[str]:
    return [f"REPEAT {count} {{", *[f"    {line}" for line in schedule], "}"]


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


def logical_hadamard_gadget_text(distance: int) -> str:
    """Render the complete H deformation as one endpoint-typed DEQ gadget."""
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
    retained_targets = [return_wire(x, y) for y in range(distance) for x in range(distance)]
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
        **{2 * patch_size + index: 3 * patch_size + index for index in range(patch_size)},
    }
    southwest_schedule, _ = swap_qec_schedule(distance, "SW", include_ticks=False)
    southwest_wires = {
        **{index: 2 * patch_size + index for index in range(patch_size)},
        **{patch_size + index: 3 * patch_size + index for index in range(patch_size)},
        **{2 * patch_size + index: 4 * patch_size + index for index in range(patch_size)},
    }
    lines = [
        "# 1. Apply transversal H, extend right, and extract extension syndromes.",
        f"H {targets(patch_targets)}",
        f"RX {targets(list(range(patch_size, 2 * patch_size - 1)))}",
        *extension_round,
        *_repeat_schedule(distance - 1, extension_round),
        "# 2. Add the missing corner and extract domain-wall syndromes.",
        f"R {2 * patch_size - 1}",
        *domain_wall_round,
        *_repeat_schedule(distance - 1, domain_wall_round),
        "# 3. Measure away the H-frame half and extract one ordinary syndrome.",
        f"M {targets(patch_targets)}",
        *shrink_round,
        "# 4. Extend left, extract return syndromes, then remove the excess columns.",
        f"R {targets(return_added_targets)}",
        *return_extension_round,
        *_repeat_schedule(distance - 1, return_extension_round),
        f"M {targets(return_removed_targets)}",
        *return_shrink_round,
        "# 5. Use two SWAP-QEC translations to return to the original footprint.",
        *_remap_schedule(northwest_schedule, northwest_wires),
        *_remap_schedule(southwest_schedule, southwest_wires),
    ]
    return "\n".join(
        (
            "# Complete transversal-H deformation. Intermediate codes remain",
            "# generator-only geometry; this gadget exposes only ordinary patches.",
            f"GADGET LogicalHadamardD{distance} {{",
            f"    INPUT {patch.type_name} {targets(patch_targets)}",
            indent(lines),
            f"    OUTPUT {patch.type_name} {targets(list(range(3 * patch_size, 4 * patch_size)))}",
            "}",
        )
    )
