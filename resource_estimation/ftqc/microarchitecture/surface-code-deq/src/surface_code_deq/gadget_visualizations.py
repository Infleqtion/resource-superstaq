"""Stim visualizations of the physical schedules behind the DEQ library."""

from __future__ import annotations

from . import surgery_geometry
from .hadamard import HadamardDeformationLayout
from .hadamard_gadgets import swap_qec_schedule
from .rotated_surface_code import RotatedSurfaceCode
from .deq_text import targets as _targets, validate_distance as _validate_distance
from .schedules import (
    check_ancilla_coordinates as _check_ancilla_coordinates,
    cnot_syndrome_schedule as _cnot_syndrome_schedule,
    mixed_syndrome_schedule as _mixed_syndrome_schedule,
    parallelize_schedules as _parallelize_schedules,
    with_seam_reset as _with_seam_reset,
)
from .visualization import (
    circuit_from_schedule,
    stim_rotated_patch_coordinates as _stim_rotated_patch_coordinates,
)


def gadget_circuits(distance: int, basis: str = "XX") -> dict[str, object]:
    """Return coordinate-preserving Stim views of the physical DEQ gadgets.

    These diagrams intentionally omit DEQ's logical ports, checks, and frame
    metadata. They otherwise use the *same explicit reset/CX/measurement
    schedule* emitted by :func:`render_mxx`; there is no MPP-only surrogate.
    """
    _validate_distance(distance)
    patch = RotatedSurfaceCode(distance, distance)
    layout = surgery_geometry.merge_layout(patch, patch, basis)
    seam_measurement = surgery_geometry.seam_measurement_instruction(layout.basis)
    patch_size = layout.first_patch_data_size
    seam_data_start = layout.seam_data_start
    merged_data_size = layout.merged_data_size
    return_extended = RotatedSurfaceCode(2 * distance - 1, distance)
    merged = RotatedSurfaceCode(layout.width, layout.height)
    patch_stabilizers = patch.stabilizers()
    extension = HadamardDeformationLayout(distance, "extension")
    merge_stabilizers = surgery_geometry.merge_stabilizers(layout)
    merged_stabilizers = merged.stabilizers(layout.qubit_at)
    patch_targets = _targets(range(patch_size))
    seam_targets = _targets(range(seam_data_start, merged_data_size))

    # For a single patch, take the layout from Stim itself.  This makes the
    # notebook's d=3 view an exact half-scale copy of
    # ``surface_code:rotated_memory_z`` rather than a hand-drawn convention.
    patch_coordinates = _stim_rotated_patch_coordinates(distance, patch_stabilizers)
    patch_data_coordinates = {
        qubit: coordinate
        for qubit, coordinate in patch_coordinates.items()
        if qubit < patch_size
    }
    merged_data_coordinates = {
        layout.qubit_at(x, y): (x + 0.5, y + 0.5)
        for y in range(layout.height)
        for x in range(layout.width)
    }
    right_patch_stabilizers = patch.stabilizers(
        lambda x, y: patch_size + y * distance + x
    )
    right_patch_coordinates = {
        patch_size + qubit: (
            x + layout.second_patch_coordinate_offset[0],
            y + layout.second_patch_coordinate_offset[1],
        )
        for qubit, (x, y) in patch_coordinates.items()
    }
    right_patch_data_coordinates = {
        qubit: coordinate
        for qubit, coordinate in right_patch_coordinates.items()
        if qubit < 2 * patch_size
    }
    patch_schedule, _ = _cnot_syndrome_schedule(
        patch_stabilizers,
        ancilla_offset=patch_size,
        data_coordinates=patch_data_coordinates,
        include_ticks=True,
    )
    extension_schedule_coordinates = extension.data_coordinates()
    full_patch = HadamardDeformationLayout(distance, "corner_moved")
    domain_wall_schedule_coordinates = full_patch.data_coordinates()
    extension_data_coordinates = dict(extension_schedule_coordinates)
    domain_wall_coordinates = dict(domain_wall_schedule_coordinates)
    extension_checks = extension.syndrome_checks()
    extension_schedule = _mixed_syndrome_schedule(
        extension_checks,
        ancilla_offset=2 * patch_size - 1,
        data_coordinates=extension_schedule_coordinates,
        include_ticks=True,
    )
    full_patch_checks = full_patch.syndrome_checks()
    domain_wall_schedule = _mixed_syndrome_schedule(
        full_patch_checks,
        ancilla_offset=2 * patch_size,
        data_coordinates=domain_wall_schedule_coordinates,
        include_ticks=True,
    )
    lower_data_qubits = {
        qubit
        for qubit, (x, _) in domain_wall_schedule_coordinates.items()
        if x > distance
    }
    shrink_schedule, _ = _cnot_syndrome_schedule(
        patch.stabilizers(lambda x, y: full_patch.wire(distance + x, y)),
        ancilla_offset=2 * patch_size,
        data_coordinates={
            qubit: coordinate
            for qubit, coordinate in domain_wall_schedule_coordinates.items()
            if qubit in lower_data_qubits
        },
        include_ticks=True,
    )
    extension_check_coordinates = _check_ancilla_coordinates(
        extension_data_coordinates,
        [
            ("X", tuple(qubit for _, qubit in check))
            for check in extension.syndrome_checks()
        ],
        ancilla_offset=2 * patch_size - 1,
    )
    domain_wall_check_coordinates = _check_ancilla_coordinates(
        domain_wall_coordinates,
        [("X", tuple(qubit for _, qubit in check)) for check in full_patch_checks],
        ancilla_offset=2 * patch_size,
    )
    shrink_coordinates = dict(domain_wall_coordinates)
    shrink_coordinates.update(
        _check_ancilla_coordinates(
            {
                qubit: coordinate
                for qubit, coordinate in domain_wall_coordinates.items()
                if qubit in lower_data_qubits
            },
            patch.stabilizers(
            lambda x, y: full_patch.wire(distance + x, y)
            ),
            ancilla_offset=2 * patch_size,
        )
    )
    return_extended_width = 2 * distance - 1
    return_extended_size = return_extended.num_data_qubits

    def return_extended_wire(x: int, y: int) -> int:
        return y * return_extended_width + x

    return_extended_coordinates = {
        return_extended_wire(x, y): (x + 1.5, y + 0.5)
        for y in range(distance)
        for x in range(return_extended_width)
    }
    return_extension_schedule, _ = _cnot_syndrome_schedule(
        return_extended.stabilizers(),
        ancilla_offset=return_extended_size,
        data_coordinates=return_extended_coordinates,
        include_ticks=True,
    )
    return_added_targets = [
        return_extended_wire(x, y) for y in range(distance) for x in range(distance - 1)
    ]
    return_removed_targets = [
        return_extended_wire(x, y)
        for y in range(distance)
        for x in range(distance, return_extended_width)
    ]
    retained_targets = {
        return_extended_wire(x, y) for y in range(distance) for x in range(distance)
    }
    retained_coordinates = {
        qubit: coordinate
        for qubit, coordinate in return_extended_coordinates.items()
        if qubit in retained_targets
    }
    retained_stabilizers = patch.stabilizers(return_extended_wire)
    return_shrink_schedule, _ = _cnot_syndrome_schedule(
        retained_stabilizers,
        ancilla_offset=return_extended_size,
        data_coordinates=retained_coordinates,
        include_ticks=True,
    )
    return_extension_coordinates = _check_ancilla_coordinates(
        return_extended_coordinates,
        return_extended.stabilizers(),
        ancilla_offset=return_extended_size,
    )
    return_shrink_coordinates = dict(return_extended_coordinates)
    return_shrink_coordinates.update(
        _check_ancilla_coordinates(
            retained_coordinates,
            retained_stabilizers,
            ancilla_offset=return_extended_size,
        )
    )
    swap_nw_schedule, swap_nw_coordinates = swap_qec_schedule(
        distance, "NW", include_ticks=True
    )
    swap_sw_schedule, swap_sw_coordinates = swap_qec_schedule(
        distance, "SW", include_ticks=True
    )
    merge_begin_schedule, _ = _cnot_syndrome_schedule(
        merge_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=merged_data_coordinates,
        include_ticks=True,
    )
    merge_begin_schedule = _with_seam_reset(merge_begin_schedule, seam_targets, basis)
    merged_schedule, _ = _cnot_syndrome_schedule(
        merged_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=merged_data_coordinates,
        include_ticks=True,
    )
    split_left_schedule, _ = _cnot_syndrome_schedule(
        patch_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=patch_data_coordinates,
        include_ticks=True,
    )
    split_right_schedule, _ = _cnot_syndrome_schedule(
        right_patch_stabilizers,
        ancilla_offset=merged_data_size + len(patch_stabilizers),
        data_coordinates=right_patch_data_coordinates,
        include_ticks=True,
    )
    split_schedule = _parallelize_schedules(split_left_schedule, split_right_schedule)
    merge_begin_coordinates = _check_ancilla_coordinates(
        merged_data_coordinates, merge_stabilizers, ancilla_offset=merged_data_size
    )
    merged_coordinates = _check_ancilla_coordinates(
        merged_data_coordinates, merged_stabilizers, ancilla_offset=merged_data_size
    )
    merge_end_coordinates = dict(merged_data_coordinates)
    merge_end_coordinates.update(
        {
            merged_data_size + index: patch_coordinates[patch_size + index]
            for index in range(len(patch_stabilizers))
        }
    )
    merge_end_coordinates.update(
        {
            merged_data_size + len(patch_stabilizers) + index: right_patch_coordinates[
                2 * patch_size + index
            ]
            for index in range(len(patch_stabilizers))
        }
    )

    circuit = circuit_from_schedule

    return {
        "PrepareX": circuit(
            [f"RX {patch_targets}", *patch_schedule], coordinates=patch_coordinates
        ),
        "PrepareZ": circuit(
            [f"R {patch_targets}", *patch_schedule], coordinates=patch_coordinates
        ),
        "SyndromeExtraction": circuit(patch_schedule, coordinates=patch_coordinates),
        "MeasureX": circuit([f"MX {patch_targets}"], coordinates=patch_coordinates),
        "MeasureZ": circuit([f"M {patch_targets}"], coordinates=patch_coordinates),
        "TransversalHadamard": circuit(
            [f"H {patch_targets}"], coordinates=patch_coordinates
        ),
        "HadamardExtend": circuit(
            [
                f"RX {' '.join(map(str, range(patch_size, 2 * patch_size - 1)))}",
                *extension_schedule,
            ],
            coordinates=extension_check_coordinates,
        ),
        "HadamardExtensionSE": circuit(
            extension_schedule, coordinates=extension_check_coordinates
        ),
        "HadamardCornerMove": circuit(
            [
                f"R {2 * patch_size - 1}",
                *domain_wall_schedule,
            ],
            coordinates=domain_wall_check_coordinates,
        ),
        "HadamardDomainWallSE": circuit(
            domain_wall_schedule, coordinates=domain_wall_check_coordinates
        ),
        "HadamardShrink": circuit(
            [
                f"M {' '.join(map(str, range(patch_size)))}",
                *shrink_schedule,
            ],
            coordinates=shrink_coordinates,
        ),
        "HadamardReturnExtend": circuit(
            [
                f"R {' '.join(map(str, return_added_targets))}",
                *return_extension_schedule,
            ],
            coordinates=return_extension_coordinates,
        ),
        "HadamardReturnExtensionSE": circuit(
            return_extension_schedule, coordinates=return_extension_coordinates
        ),
        "HadamardReturnShrink": circuit(
            [
                f"M {' '.join(map(str, return_removed_targets))}",
                *return_shrink_schedule,
            ],
            coordinates=return_shrink_coordinates,
        ),
        "HadamardSwapQECNW": circuit(
            [
                *swap_nw_schedule,
            ],
            coordinates=swap_nw_coordinates,
        ),
        "HadamardSwapQECSW": circuit(
            [
                *swap_sw_schedule,
            ],
            coordinates=swap_sw_coordinates,
        ),
        f"MergeBegin{basis}": circuit(
            [
                *merge_begin_schedule,
            ],
            coordinates=merge_begin_coordinates,
        ),
        f"MergedSE{basis}": circuit(
            [
                *merged_schedule,
            ],
            coordinates=merged_coordinates,
        ),
        f"MergeEnd{basis}": circuit(
            [
                f"{seam_measurement} {seam_targets}",
                *split_schedule,
            ],
            coordinates=merge_end_coordinates,
        ),
    }
