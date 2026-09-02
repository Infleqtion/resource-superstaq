"""Ideal terminal error correction and logical readout for test experiments."""

from __future__ import annotations

from .deq_text import indent, targets
from .rotated_surface_code import RotatedSurfaceCode
from .schedules import cnot_syndrome_schedule, square_data_coordinates


def perfect_readout_gadgets_text(distance: int) -> str:
    """Return noiseless terminal-QEC and Z-readout gadgets for a square patch.

    Append this text *after* noise injection.  The syndrome round then reveals
    correctable errors left on a tested gadget's output without introducing a
    new noisy time boundary before the final logical readout.
    """
    patch = RotatedSurfaceCode(distance, distance)
    size = patch.num_data_qubits
    schedule, _ = cnot_syndrome_schedule(
        patch.stabilizers(),
        ancilla_offset=size,
        data_coordinates=square_data_coordinates(distance),
    )
    readout = " ".join(f"rec[-{size - index}]" for index in range(distance))
    return f"""# Ideal terminal error correction and readout used only by experiments.
GADGET PerfectSyndromeExtraction {{
    INPUT {patch.type_name} {targets(range(size))}
{indent(schedule)}
    OUTPUT {patch.type_name} {targets(range(size))}
}}

GADGET PerfectMeasureZ {{
    INPUT {patch.type_name} {targets(range(size))}
    MZ {targets(range(size))}
    READOUT {readout}
}}
"""
