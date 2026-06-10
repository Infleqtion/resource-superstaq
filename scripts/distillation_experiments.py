import cirq
import cirq_superstaq as css
from functools import partial
from pathlib import Path
from collections import Counter
from cultivate_json import GATE2STR
parent_dir = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(parent_dir))
import resource_estimation as res

# Current problems with this approach

def distilled_costs(save: bool) -> dict:
    available_arcs = [
        partial(res.architecture.DualSpeciesMovement, post_op_correction=False),
        partial(res.architecture.MeasureZonesOnly, post_op_correction=False),
        partial(res.architecture.DefaultMovement, post_op_correction=False),
    ]
    qubits = cirq.LineQubit.range(16)

    circuit = cirq.Circuit([
        cirq.ResetChannel().on_each(*qubits),
        # css.Barrier(16).on(*qubits),
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        cirq.H(qubits[3]),
        cirq.H(qubits[7]),
        cirq.H(qubits[15]),
        # css.Barrier(16).on(*qubits),
        cirq.CNOT(qubits[15], qubits[14]),
        cirq.CNOT(qubits[7], qubits[8]),
        cirq.CNOT(qubits[7], qubits[9]),
        cirq.CNOT(qubits[7], qubits[10]),
        cirq.CNOT(qubits[7], qubits[11]),
        cirq.CNOT(qubits[7], qubits[12]),
        cirq.CNOT(qubits[7], qubits[13]),
        cirq.CNOT(qubits[7], qubits[14]),
        cirq.CNOT(qubits[3], qubits[4]),
        cirq.CNOT(qubits[3], qubits[5]),
        cirq.CNOT(qubits[3], qubits[6]),
        cirq.CNOT(qubits[3], qubits[11]),
        cirq.CNOT(qubits[3], qubits[12]),
        cirq.CNOT(qubits[3], qubits[13]),
        cirq.CNOT(qubits[3], qubits[14]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.CNOT(qubits[1], qubits[5]),
        cirq.CNOT(qubits[1], qubits[6]),
        cirq.CNOT(qubits[1], qubits[9]),
        cirq.CNOT(qubits[1], qubits[10]),
        cirq.CNOT(qubits[1], qubits[13]),
        cirq.CNOT(qubits[1], qubits[14]),
        cirq.CNOT(qubits[0], qubits[2]),
        cirq.CNOT(qubits[0], qubits[4]),
        cirq.CNOT(qubits[0], qubits[6]),
        cirq.CNOT(qubits[0], qubits[8]),
        cirq.CNOT(qubits[0], qubits[10]),
        cirq.CNOT(qubits[0], qubits[12]),
        cirq.CNOT(qubits[0], qubits[14]),
        cirq.CNOT(qubits[14], qubits[2]),
        cirq.CNOT(qubits[14], qubits[4]),
        cirq.CNOT(qubits[14], qubits[5]),
        cirq.CNOT(qubits[14], qubits[8]),
        cirq.CNOT(qubits[14], qubits[9]),
        cirq.CNOT(qubits[14], qubits[11]),
        css.Barrier(16).on(*qubits),
        cirq.T.on_each(*qubits[:-1]),
        css.Barrier(16).on(*qubits),
        cirq.H.on_each(*qubits[:-1]),
        cirq.MeasurementGate(1).on_each(*qubits[:-1]),
    ])
    circuit = cirq.align_left(circuit)
    sub_circuit = circuit[:18] + circuit[21:]  # Remove the t gate moment
    layout = res.layout.MovementLayout(input_circuit=sub_circuit, num_t_factories=0)
    saved_data = {arc().__name__: {} for arc in available_arcs}
    for d in range(7, 20, 2):
        for arc in available_arcs:
            arc = arc(d=d)
            primitive_circuit = cirq.synchronize_terminal_measurements(
                cirq.align_left(
                    res.compile_ftqc.ft_compile(layout=layout, arc=arc)
                )
            )[1:]
            estimator = res.estimate.ResourceEstimator(arc)
            _time, _moments, _gates = estimator.parallel_circuit_time(primitive_circuit), estimator.parallel_circuit_cost(primitive_circuit), estimator.serial_circuit_cost(primitive_circuit)
            data = {
                "op_time": _time,
                "moment_cost": {GATE2STR[gate]: num for gate, num in _moments.items()}, 
                "gate_cost": {GATE2STR[gate]: num for gate, num in _gates.items()}
            }
            saved_data[arc.__name__][d] = data
    return saved_data

from pprint import pprint
pprint(distilled_costs(save=False))
saved_costs = distilled_costs(save=False)

def distil_moments(arc: res.architecture.Architecture, reps: int, depth: int) -> dict:
    """
    Recursively determines the moment cost of distillation using a base circuit cost and mutiplying the cost by the rate of repetition
    """
    def load_cost():
        str_cost = saved_costs[arc.__name__][arc.d]['moment_cost']
        return Counter({res.stim_functions.STR2GATE[key]: val for key, val in str_cost.items()})
    cost = load_cost()
    if depth > 0:
        cost += Counter({cirq.CZ: 1})  # Teleportation (but missing any movements)
        cost += distil_moments(arc=arc, reps=reps, depth=depth-1)
    else:
        cost += arc.syndrome_extract_cost(
            res.lattice_surgery_primitives.SyndromeExtract(1, arc.d).on(cirq.LineQubit(0))
        )['moment_cost']
        cost += Counter({cirq.PhasedXZGate: 1})
    return Counter({k: v*reps for k, v in cost.items()})

for depth in range(4):
    moments = distil_moments(
        res.architecture.DefaultMovement(d=11),
        reps=2,
        depth=depth,
    )
    pprint(moments)