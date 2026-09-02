#!/usr/bin/env python3
"""Generate a distance-independent rotated-surface-code DEQ library."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Literal, NamedTuple, TypeAlias

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface_code_deq.hadamard_gadgets import (
    logical_hadamard_gadget_text,
)
from surface_code_deq.deq_text import (
    code_text as _code_text,
    document as _deq_document,
    indent as _indent,
    targets as _targets,
    validate_distance as _validate_distance,
)
from surface_code_deq.prepare_y_gadgets import (
    prepare_y_gadget_text,
    render_prepare_y,
)
from surface_code_deq.rotated_surface_code import RotatedSurfaceCode
from surface_code_deq import surgery_geometry
from surface_code_deq.schedules import (
    check_ancilla_coordinates as _check_ancilla_coordinates,
    cnot_syndrome_schedule as _cnot_syndrome_schedule,
    mixed_syndrome_schedule as _mixed_syndrome_schedule,
    parallelize_schedules as _parallelize_schedules,
    square_data_coordinates as _square_data_coordinates,
    with_seam_reset as _with_seam_reset,
)


Operation: TypeAlias = Literal[
    "surface-code",
    "surgery-library",
    "cnot",
    "prepare-y",
]


class SurgerySections(NamedTuple):
    """Named DEQ sections for one lattice-surgery orientation."""

    common_codes: str
    merged_code: str
    ordinary_gadgets: str
    merge_gadgets: str
    memory_composition: str
    measurement_composition: str
    verification_programs: str

    def render(self) -> str:
        return _deq_document(
            self.common_codes,
            self.merged_code,
            self.ordinary_gadgets,
            self.merge_gadgets,
            self.memory_composition,
            self.measurement_composition,
            self.verification_programs,
        )

def _logical_s_gadget_text(distance: int) -> str:
    """Render logical-S injection using a prepared encoded ``|+i>`` state."""
    patch_code = RotatedSurfaceCode(distance, distance).type_name
    return f"""# Inject a logical S using an encoded |+i> = S|+> resource state.
# Port 0 is the data patch; internal port 1 is the consumed |+i> patch.
COMPOSE LogicalSD{distance} {{
    INPUT {patch_code} 0
    PrepareY 1
    FaultTolerantCNOTD{distance} 0 1
    MeasureZ 1
    # A 1 outcome leaves Z*S on the data, so update its Z frame.
    CONDITIONAL rec[-1] Z0 0
    OUTPUT {patch_code} 0
}}
"""


def render_surface_code_library(
    distance: int = 3,
    *,
    merged_rounds: int | None = None,
    boundary_rounds: int | None = None,
) -> str:
    """Return the complete surface-code gadget library in one DEQ file.

    ``render_cnot`` already contains the ordinary patch, both surgery
    orientations, CNOT, and transversal-H libraries. Insert Prepare-Y beside
    the ordinary preparation gadgets, while avoiding a second ordinary-code
    type.
    """
    _validate_distance(distance)
    base_library = render_cnot(distance, merged_rounds=merged_rounds).rstrip()
    prepare_y_gadget = prepare_y_gadget_text(
        distance, boundary_rounds=boundary_rounds
    )
    prep_gadget_start = base_library.index("GADGET PrepareZ")
    syndrome_gadget_start = base_library.index("GADGET SyndromeExtraction")
    code_definitions = base_library[:prep_gadget_start]
    ordinary_preparation = base_library[prep_gadget_start:syndrome_gadget_start]
    remaining_gadgets = base_library[syndrome_gadget_start:]
    logical_s = _logical_s_gadget_text(distance).strip()
    return (
        code_definitions
        + ordinary_preparation
        + "\n\n"
        + prepare_y_gadget
        + "\n\n"
        + remaining_gadgets
        + "\n\n"
        + logical_s
    )


def _render_surgery_sections(
    distance: int, basis: str, *, merged_rounds: int | None = None
) -> str:
    """Render either horizontal MXX or vertical MZZ surgery source."""
    _validate_distance(distance)
    patch = RotatedSurfaceCode(distance, distance)
    layout = surgery_geometry.merge_layout(patch, patch, basis)
    parity_basis = basis[0]
    byproduct_pauli = {"X": "Z", "Z": "X"}[parity_basis]
    seam_measurement = surgery_geometry.seam_measurement_instruction(layout.basis)
    if merged_rounds is None:
        merged_rounds = distance
    if merged_rounds < 0:
        raise ValueError("merged_rounds must be non-negative")

    patch_size = layout.first_patch_data_size
    seam_data_start = layout.seam_data_start
    merged_data_size = layout.merged_data_size
    merged = RotatedSurfaceCode(layout.width, layout.height)
    patch_code = patch.type_name
    merged_code = merged.type_name
    patch_stabilizers = patch.stabilizers()
    merged_stabilizers = merged.stabilizers(layout.qubit_at)
    merge_stabilizers = surgery_geometry.merge_stabilizers(layout)
    retained_stabilizers = surgery_geometry.retained_layout_stabilizers(layout)
    if set(retained_stabilizers + merge_stabilizers) != set(merged_stabilizers):
        raise AssertionError(
            "merge geometry does not reproduce the merged stabilizer group"
        )
    if len(merged_stabilizers) != merged_data_size - 1:
        raise AssertionError("merged code has the wrong number of stabilizers")

    logical_x, logical_z = patch.logical_x(), patch.logical_z()
    merged_logical_x = merged.logical_x(layout.qubit_at)
    merged_logical_z = merged.logical_z(layout.qubit_at)
    patch_targets = _targets(range(patch_size))
    right_patch_targets = _targets(range(patch_size, 2 * patch_size))
    seam_targets = _targets(range(seam_data_start, merged_data_size))
    merged_targets = _targets(range(merged_data_size))
    # The physical merge has an MZZ/MXX ambiguity: without this frame update,
    # DEQ picks the measure-and-reset (MRZZ/MRXX) interpretation and aliases
    # output B's measured-basis frame to A's.  Conditioning B on the joint
    # merge parity instead declares the nondestructive measurement semantics.
    # The standard patch's logical Z is the horizontal top-edge string.
    # (The vertical left-edge string is logical X and is used by MeasureX.)
    readout_z = " ".join(
        f"rec[-{patch_size - index}]" for index in range(distance)
    )
    patch_data_coordinates = _square_data_coordinates(distance)
    merged_data_coordinates = {
        layout.qubit_at(x, y): (x + 0.5, y + 0.5)
        for y in range(layout.height)
        for x in range(layout.width)
    }
    right_patch_stabilizers = patch.stabilizers(
        lambda x, y: patch_size + y * distance + x
    )
    right_patch_data_coordinates = _square_data_coordinates(
        distance,
        qubit_offset=patch_size,
        offset=layout.second_patch_coordinate_offset,
    )
    patch_schedule_lines, _ = _cnot_syndrome_schedule(
        patch_stabilizers,
        ancilla_offset=patch_size,
        data_coordinates=patch_data_coordinates,
    )
    merge_begin_lines, merge_measurement_index = _cnot_syndrome_schedule(
        merge_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=merged_data_coordinates,
    )
    merge_begin_lines = _with_seam_reset(merge_begin_lines, seam_targets, basis)
    merged_schedule_lines, _ = _cnot_syndrome_schedule(
        merged_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=merged_data_coordinates,
    )
    split_left_schedule_lines, _ = _cnot_syndrome_schedule(
        patch_stabilizers,
        ancilla_offset=merged_data_size,
        data_coordinates=patch_data_coordinates,
    )
    split_right_schedule_lines, _ = _cnot_syndrome_schedule(
        right_patch_stabilizers,
        ancilla_offset=merged_data_size + len(patch_stabilizers),
        data_coordinates=right_patch_data_coordinates,
    )
    merge_readout = " ".join(
        f"M{merge_measurement_index[index]}"
        for index in surgery_geometry.merge_readout_indices(layout)
    )
    patch_syndrome_schedule = _indent(patch_schedule_lines)
    merge_begin_schedule = _indent(merge_begin_lines)
    merged_syndrome_schedule = _indent(merged_schedule_lines)
    split_schedule = _indent(
        _parallelize_schedules(split_left_schedule_lines, split_right_schedule_lines)
    )

    return SurgerySections(
        common_codes=f"""# Generated by tools/generate_rotated_surface_code_deq.py; do not edit by hand.
# Distance-{distance} fault-tolerant logical M{basis}, factored into local DEQ gadgets.
# The two d×d patches are joined by a d-qubit data seam starting at {seam_data_start}.

{_code_text(patch_code, patch_size, logical_x, logical_z, patch_stabilizers, distance=distance)}
""",
        merged_code=f"""# A {layout.width}×{layout.height} merged patch with one logical qubit.
{_code_text(merged_code, merged_data_size, merged_logical_x, merged_logical_z, merged_stabilizers, distance=None)}
""",
        ordinary_gadgets=f"""GADGET PrepareZ {{
    RZ {patch_targets}
    # One ephemeral ancilla per check, following the CNOT schedule below.
{patch_syndrome_schedule}
    OUTPUT {patch_code} {patch_targets}
}}

GADGET PrepareX {{
    RX {patch_targets}
    # One ephemeral ancilla per check, following the CNOT schedule below.
{patch_syndrome_schedule}
    OUTPUT {patch_code} {patch_targets}
}}

GADGET SyndromeExtraction {{
    INPUT {patch_code} {patch_targets}
    # One ephemeral ancilla per check, following the CNOT schedule below.
{patch_syndrome_schedule}
    OUTPUT {patch_code} {patch_targets}
}}

GADGET MeasureZ {{
    INPUT {patch_code} {patch_targets}
    MZ {patch_targets}
    READOUT {readout_z}
}}

GADGET MeasureX {{
    INPUT {patch_code} {patch_targets}
    MX {patch_targets}
    # The standard patch's logical X is the vertical left-edge string.
    READOUT {" ".join(f"rec[-{patch_size - index}]" for index in range(0, patch_size, distance))}
}}

# A decoder-level Pauli-frame update. This contains no physical operation but
# flips the tracked Z-observable value, exactly as a logical X should.
GADGET LogicalX {{
    INPUT {patch_code} {patch_targets}
    OUTPUT {patch_code} {patch_targets}
    VIRTUAL LX0
}}

GADGET LogicalZ {{
    INPUT {patch_code} {patch_targets}
    OUTPUT {patch_code} {patch_targets}
    VIRTUAL LZ0
}}
""",
        merge_gadgets=f"""# MergeBegin changes two RotatedSurfaceCode ports into one merged-code port. Its
# readout is the XOR of all new {basis[0]} checks, i.e. {basis[0]}A*{basis[0]}B in the input frame.  The
# seam is initialized in the complementary basis before joining the merged-code output;
# the check-measurement ancillas used by the CNOT schedule are separate.
GADGET MergeBegin{basis} {{
    INPUT {patch_code} {patch_targets}
    INPUT {patch_code} {right_patch_targets}
{merge_begin_schedule}
    READOUT {merge_readout}
    OUTPUT {merged_code} {merged_targets}
}}

# Exactly one complete extraction round on the merged code.
GADGET MergedSE{basis} {{
    INPUT {merged_code} {merged_targets}
    # Continue extracting while retaining the seam within the merged patch.
{merged_syndrome_schedule}
    OUTPUT {merged_code} {merged_targets}
}}

# Split by measuring the seam, then restore one Stim-style extraction round on
# each patch. The restored boundary checks make both output code ports explicit.
GADGET MergeEnd{basis} {{
    INPUT {merged_code} {merged_targets}
    {seam_measurement} {seam_targets}
{split_schedule}
    OUTPUT {patch_code} {patch_targets}
    OUTPUT {patch_code} {right_patch_targets}
}}
""",
        memory_composition=f"""COMPOSE RotatedSurfaceCodeMemoryD{distance} {{
    INPUT {patch_code} 0
    REPEAT {distance} {{
        SyndromeExtraction 0
    }}
    OUTPUT {patch_code} 0
}}
""",
        measurement_composition=f"""COMPOSE FaultTolerantM{basis}D{distance} {{
    INPUT {patch_code} 0
    INPUT {patch_code} 1
    MergeBegin{basis} IN(0 1) OUT(0)
    REPEAT {merged_rounds} {{
        MergedSE{basis} 0
    }}
    MergeEnd{basis} IN(0) OUT(0 1)
    # Keep patch B's measured-basis frame, rather than DEQ's default
    # measure-and-reset interpretation. rec[-1] is MergeBegin's joint parity:
    # MXX needs a logical Z correction; MZZ needs logical X.
    CONDITIONAL rec[-1] {byproduct_pauli}0 1
    OUTPUT {patch_code} 0
    OUTPUT {patch_code} 1
}}
""",
        verification_programs=f"""PROGRAM FaultTolerantM{basis}D{distance}Memory{parity_basis} {{
    Prepare{parity_basis} 0
    Prepare{parity_basis} 1
    FaultTolerantM{basis}D{distance} 0 1
    Measure{parity_basis} 0
    Measure{parity_basis} 1
    ASSERT_EQ rec[-3] 0
    ASSERT_EQ rec[-2] 0
    ASSERT_EQ rec[-1] 0
}}

# These measured-basis eigenstate cases verify the joint parity and that both
# output frames are preserved. They catch DEQ's default measure-and-reset
# interpretation, which would copy patch A's frame onto patch B.
PROGRAM FaultTolerantM{basis}D{distance}FrameA {{
    Prepare{parity_basis} 0
    Prepare{parity_basis} 1
    Logical{byproduct_pauli} 0
    FaultTolerantM{basis}D{distance} 0 1
    Measure{parity_basis} 0
    Measure{parity_basis} 1
    ASSERT_EQ rec[-3] 1
    ASSERT_EQ rec[-2] 1
    ASSERT_EQ rec[-1] 0
}}

PROGRAM FaultTolerantM{basis}D{distance}FrameB {{
    Prepare{parity_basis} 0
    Prepare{parity_basis} 1
    Logical{byproduct_pauli} 1
    FaultTolerantM{basis}D{distance} 0 1
    Measure{parity_basis} 0
    Measure{parity_basis} 1
    ASSERT_EQ rec[-3] 1
    ASSERT_EQ rec[-2] 0
    ASSERT_EQ rec[-1] 1
}}

PROGRAM FaultTolerantM{basis}D{distance}FrameBoth {{
    Prepare{parity_basis} 0
    Prepare{parity_basis} 1
    Logical{byproduct_pauli} 0
    Logical{byproduct_pauli} 1
    FaultTolerantM{basis}D{distance} 0 1
    Measure{parity_basis} 0
    Measure{parity_basis} 1
    ASSERT_EQ rec[-3] 0
    ASSERT_EQ rec[-2] 1
    ASSERT_EQ rec[-1] 1
}}""",
    )


def render_mxx(distance: int = 3, *, merged_rounds: int | None = None) -> str:
    """Return fault-tolerant horizontal MXX DEQ source for any odd distance."""
    return _render_surgery_sections(distance, "XX", merged_rounds=merged_rounds).render()


def render_mzz(distance: int = 3, *, merged_rounds: int | None = None) -> str:
    """Return fault-tolerant vertical MZZ DEQ source for any odd distance."""
    return _render_surgery_sections(distance, "ZZ", merged_rounds=merged_rounds).render()


def render_surgery_library(
    distance: int = 3, *, merged_rounds: int | None = None
) -> str:
    """Return one DEQ library containing both MXX and MZZ surgery gadgets."""
    mxx = _render_surgery_sections(distance, "XX", merged_rounds=merged_rounds)
    mzz = _render_surgery_sections(distance, "ZZ", merged_rounds=merged_rounds)
    return _deq_document(
        mxx.common_codes,
        mxx.merged_code,
        mzz.merged_code,
        mxx.ordinary_gadgets,
        mxx.merge_gadgets,
        mzz.merge_gadgets,
        mxx.memory_composition,
        mxx.measurement_composition,
        mzz.measurement_composition,
        mxx.verification_programs,
        mzz.verification_programs,
        logical_hadamard_gadget_text(distance),
    )


def render_cnot(distance: int = 3, *, merged_rounds: int | None = None) -> str:
    """Return the shared surgery library plus a mediator-based CNOT gadget."""
    patch_code = RotatedSurfaceCode(distance, distance).type_name
    library = render_surgery_library(distance, merged_rounds=merged_rounds)
    cnot = "\n".join(
        (
            "# Port 0 is the control and port 1 is the target.",
            "# Internal port 2 is a |+> mediator patch.",
            f"COMPOSE FaultTolerantCNOTD{distance} {{",
            f"    INPUT {patch_code} 0",
            f"    INPUT {patch_code} 1",
            "    PrepareX 2",
            f"    FaultTolerantMZZD{distance} 0 2",
            f"    FaultTolerantMXXD{distance} 2 1",
            "    MeasureZ 2",
            "    # Record order: mZZ, mXX, mZ_mediator.",
            "    CONDITIONAL rec[-2] Z0 0",
            "    CONDITIONAL rec[-3] X0 1",
            "    CONDITIONAL rec[-1] X0 1",
            f"    OUTPUT {patch_code} 0",
            f"    OUTPUT {patch_code} 1",
            "}",
        )
    )
    return library + "\n" + cnot + "\n"


def inject_si1000_noise(source: str, physical_error_rate: float) -> str:
    """Add DEQ's SI1000 circuit-level noise to every generated gadget body."""
    if not 0 <= physical_error_rate <= 1:
        raise ValueError("SI1000 physical error rate must be in [0, 1]")
    try:
        from deq.noise import inject_si1000
    except Exception as error:
        raise RuntimeError(
            "SI1000 injection requires the project-local DEQ environment; run "
            ".venv-deq/bin/python tools/generate_rotated_surface_code_deq.py"
        ) from error
    return inject_si1000(source, physical_error_rate)


def render_operation(
    operation: Operation,
    *,
    distance: int,
    merged_rounds: int | None,
    boundary_rounds: int | None,
) -> str:
    """Render one command-line selectable DEQ library."""
    if operation == "surface-code":
        return render_surface_code_library(
            distance,
            merged_rounds=merged_rounds,
            boundary_rounds=boundary_rounds,
        )
    if operation == "surgery-library":
        return render_surgery_library(distance, merged_rounds=merged_rounds)
    if operation == "cnot":
        return render_cnot(distance, merged_rounds=merged_rounds)
    if operation == "prepare-y":
        return render_prepare_y(distance, boundary_rounds=boundary_rounds)
    raise AssertionError(f"unsupported operation {operation!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument(
        "--operation",
        choices=(
            "surface-code",
            "surgery-library",
            "cnot",
            "prepare-y",
        ),
        default="cnot",
    )
    parser.add_argument("--merged-rounds", type=int)
    parser.add_argument("--boundary-rounds", type=int)
    parser.add_argument(
        "--noise-model",
        choices=("none", "si1000"),
        default="none",
        help="Optional circuit-level noise model applied to generated gadget bodies.",
    )
    parser.add_argument(
        "--noise-p",
        type=float,
        help="Physical error rate for the selected noise model.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = render_operation(
        args.operation,
        distance=args.distance,
        merged_rounds=args.merged_rounds,
        boundary_rounds=args.boundary_rounds,
    )
    if args.noise_model == "si1000":
        if args.noise_p is None:
            parser.error("--noise-p is required when --noise-model si1000")
        source = inject_si1000_noise(source, args.noise_p)
    elif args.noise_p is not None:
        parser.error("--noise-p requires a non-default --noise-model")
    args.out.write_text(source)


if __name__ == "__main__":
    main()
