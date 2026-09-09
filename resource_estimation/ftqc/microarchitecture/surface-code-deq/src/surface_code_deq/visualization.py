"""Stim circuit construction used by notebooks and gadget visualizations."""

from __future__ import annotations

from .deq_text import validate_distance
from .gadgets.core import Schedule
from .gadgets.hadamard import hadamard_gadgets, swap_qec_schedule
from .gadgets.memory import patch_gadgets
from .gadgets.prepare_y import prepare_y_gadgets
from .gadgets.surgery import surgery_gadgets
from .surgery_geometry import SurgeryBasis
from .types import Coordinates


def circuit_from_schedule(schedule: Schedule, *, coordinates: Coordinates):
    """Build a coordinate-annotated Stim circuit from a shared schedule.

    Importing Stim lazily keeps DEQ source rendering independent of the
    visualization backend.
    """
    import stim

    body = stim.Circuit("\n".join(schedule.lines(include_ticks=True)))
    result = stim.Circuit()
    for qubit, (x, y) in sorted(coordinates.items()):
        if qubit < body.num_qubits:
            result.append("QUBIT_COORDS", [qubit], [x, y])
    result += body
    return result


def gadget_stim_circuits(distance: int, basis: SurgeryBasis = "XX") -> dict[str, object]:
    """Return coordinate-preserving Stim views of every physical gadget."""
    validate_distance(distance)
    ordinary = patch_gadgets(distance)
    prepare_y = prepare_y_gadgets(distance)
    surgery = surgery_gadgets(distance, basis)
    hadamard = hadamard_gadgets(distance)
    northwest_schedule, northwest_coordinates = swap_qec_schedule(distance, "NW")
    southwest_schedule, southwest_coordinates = swap_qec_schedule(distance, "SW")
    return {
        **{name: spec.stim_circuit() for name, spec in ordinary.items()},
        **{name: spec.stim_circuit() for name, spec in prepare_y.items()},
        **{
            name: spec.stim_circuit()
            for name, spec in hadamard.items()
            if name != "HadamardSwapQEC"
        },
        "HadamardSwapQECNW": circuit_from_schedule(
            northwest_schedule, coordinates=northwest_coordinates
        ),
        "HadamardSwapQECSW": circuit_from_schedule(
            southwest_schedule, coordinates=southwest_coordinates
        ),
        **{name: spec.stim_circuit() for name, spec in surgery.gadgets.items()},
    }
