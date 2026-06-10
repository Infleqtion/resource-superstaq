# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from typing import Literal
import cirq
# from compile_ftqc import ft_compile
# from architecture import Architecture
# from layout import MovementLayout
# from estimate import ResourceEstimator
import resource_estimation as res
from resource_estimation import architecture
from . import magic_state_distillation as msd
import cirq_superstaq as css
from functools import partial
# import magic_state_distillation as msd

def distil(
        arch,
) -> dict[Literal['op_time', 'moment_cost', 'gate_cost'],
          Counter[cirq.Gate, int]]:
    """
    Generates the logical qubit resources required for 15-to-1 magic
    state distilation.
    """
    circuit = msd.msd_15_to_1()
    movement = arch.movement
    if movement:
        layout = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=15)
    else:
        layout = res.layout.FactorySandwich(input_circuit=circuit, num_t_factories=15, num_s_factories=1)
    primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=False)
    estimator = res.estimate.ResourceEstimator(arc=arch)
    parallel_gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=False)
    serial_gate_cost = estimator.serial_circuit_cost(primitive_circuit, pretty=False)
    circuit_time = estimator.parallel_circuit_time(primitive_circuit)
    resources = {'op_time': circuit_time, 'moment_cost': Counter(parallel_gate_cost), 'gate_cost': Counter(serial_gate_cost)}
    return resources

# def distil2(arch: res.architecture.Architecture, reps: int, depth: int) -> dict[Literal['op_time', 'moment_cost', 'gate_cost'], Counter[cirq.Gate, int]]:
#     # Folowing Fig 33 in https://arxiv.org/pdf/1208.0928
#     # We include a Logical Reset and Hadamard moment to prepare the logical qubits
#     # Next is a layer of 16 Logical CNOTs
#     # Next is a layer of T gates
#     #     One way to handle is to do physical T and grow
#     #     Another way to handle is to perform a layer of CNOTs on previously distilled qubits, possibly with corrections
#     #     Another way is to perform a layer of CNOTs on previously cultivated qubits, possibly with corrections (But this is a difficult problem so I'll approach it later)
#     # Next is a layer of Logical Hadamard gates
#     # Finally a layer of logical Measurement gates is performed
#     # Based on the results of the measurement, the process is repeated or deemed complete
#     if isinstance(arch, res.architecture.DefaultLattice):
#         raise NotImplementedError("Complete model for distillation on lattice surgery architectures is not yet supported")
#     # Map cirq gate types to cost methods in the Architecture class
#     gate_to_cost = {
#         cirq.H: '_h_cost',
#         cirq.CNOT: '_cnot_cost',
#         cirq.MeasurementGate: '_measure_cost',
#         cirq.ResetChannel: '_reset_cost',
#         cirq.S: '_s_cost',
#     }
#     logical_moment_cost = distil_moments(reps=reps, depth=depth)
#     logical_gate_cost = distil_gates(reps=reps, depth=depth)
#     physical_moment_cost = Counter()
#     for cirq_gate, num_occurrences in logical_moment_cost.items():
#         if cirq_gate is cirq.H:
#             physical_moment_cost += arch._h_cost['moment_cost']
#         elif cirq_gate is cirq.S:
#             physical_moment_cost += arch._s_cost['moment_cost']
#         elif cirq_gate is cirq.MeasurementGate:
#             physical_moment_cost += arch._measure_cost['moment_cost']
#         if cirq_gate in arch.primitives:
#             physical_moment_cost += getattr(arch, gate_to_cost.get(cirq_gate))['moment_cost']
#             # What breaks this?
#             # CNOT might need move. Lattice architectures don't have CNOT or S. T is it's own thing
#         else:
#             physical_moment_cost += do_something_else(cirq_gate)
    
    
#     for cirq_gate, num_occurrences in logical_gate_cost.items():
#         if has_cost(cirq_gate):
#             physical_gate_cost += arch.op_cost(cirq_gate)
#         else:
#             physical_gate_cost += do_something_else(cirq_gate)
#     op_time = arch.total_time(physical_moment_cost)
#     return {'op_time': op_time, 'moment_cost': physical_moment_cost, 'gate_cost': physical_gate_cost}

# def distil_moments(reps: int, depth: int, movement: bool):
#     if depth == 0:
#         cost = Counter({
#             cirq.ResetChannel: 1 * reps,
#             cirq.H: 2 * reps,
#             cirq.CNOT: 16 * reps,
#             cirq.T: 1 * reps,
#             cirq.MeasurmentGate: 1 * reps
#         })
#         return cost
#     cost = Counter({
#         cirq.CNOT: 1 * reps,
#         cirq.MeasurementGate: 1 * reps,
#         cirq.S: 1 * reps,  # Conditional on measurement outcome 
#     })
#     cost += distil_moments(reps*reps, depth-1)
#     return cost

# def distil_gates(reps: int, depth: int):
#     if depth == 0:
#         cost = Counter({
#             cirq.ResetChannel: 16 * reps,
#             cirq.H: 20 * reps,
#             cirq.CNOT: 35 * reps,
#             cirq.T: 15 * reps,
#             cirq.MeasurementGate: 15 * reps,
#         })
#         return cost
#     else:
#         cost = Counter({
#             cirq.ResetChannel: 16 * reps,
#             cirq.H: 20 * reps,
#             cirq.CNOT: 35 * reps,
#             cirq.MeasurementGate: 15 * reps,
#         })
#         cost += distil_gates(reps*reps, depth-1)

# def distil_moments(arc: res.architecture.Architecture, reps: int, depth: int) -> dict:
#     """
#     Recursively determines the moment cost of distillation using a base circuit cost and mutiplying the cost by the rate of repetition
#     """
#     cost = Counter(load_cost(arc.__name__)['moment_cost'])
#     if depth > 0:
#         cost += Counter({cirq.CZ: 1})  # Teleportation (but missing any movements)
#         cost += distil_moments(reps, depth-1)
#     else:
#         cost += arc.syndrome_extract_cost(res.lattice_surgery_primitives.SyndromeExtract(1, arc.d))['moment_cost']
#         cost += Counter({cirq.Ph9asedXZGate: 1})
#     return Counter({k: v*reps for k, v in cost.items()})

# def distilled_costs(save: bool) -> dict:
#     available_arcs = [
#         partial(res.architecture.DualSpeciesMovement, post_op_correction=False),
#         partial(res.architecture.MeasureZonesOnly, post_op_correction=False),
#         partial(res.architecture.DefaultMovement, post_op_correction=False),
#     ]
#     qubits = cirq.LineQubit.range(16)

#     circuit = cirq.Circuit([
#         cirq.ResetChannel().on_each(*qubits),
#         # css.Barrier(16).on(*qubits),
#         cirq.H(qubits[0]),
#         cirq.H(qubits[1]),
#         cirq.H(qubits[3]),
#         cirq.H(qubits[7]),
#         cirq.H(qubits[15]),
#         # css.Barrier(16).on(*qubits),
#         cirq.CNOT(qubits[15], qubits[14]),
#         cirq.CNOT(qubits[7], qubits[8]),
#         cirq.CNOT(qubits[7], qubits[9]),
#         cirq.CNOT(qubits[7], qubits[10]),
#         cirq.CNOT(qubits[7], qubits[11]),
#         cirq.CNOT(qubits[7], qubits[12]),
#         cirq.CNOT(qubits[7], qubits[13]),
#         cirq.CNOT(qubits[7], qubits[14]),
#         cirq.CNOT(qubits[3], qubits[4]),
#         cirq.CNOT(qubits[3], qubits[5]),
#         cirq.CNOT(qubits[3], qubits[6]),
#         cirq.CNOT(qubits[3], qubits[11]),
#         cirq.CNOT(qubits[3], qubits[12]),
#         cirq.CNOT(qubits[3], qubits[13]),
#         cirq.CNOT(qubits[3], qubits[14]),
#         cirq.CNOT(qubits[1], qubits[2]),
#         cirq.CNOT(qubits[1], qubits[5]),
#         cirq.CNOT(qubits[1], qubits[6]),
#         cirq.CNOT(qubits[1], qubits[9]),
#         cirq.CNOT(qubits[1], qubits[10]),
#         cirq.CNOT(qubits[1], qubits[13]),
#         cirq.CNOT(qubits[1], qubits[14]),
#         cirq.CNOT(qubits[0], qubits[2]),
#         cirq.CNOT(qubits[0], qubits[4]),
#         cirq.CNOT(qubits[0], qubits[6]),
#         cirq.CNOT(qubits[0], qubits[8]),
#         cirq.CNOT(qubits[0], qubits[10]),
#         cirq.CNOT(qubits[0], qubits[12]),
#         cirq.CNOT(qubits[0], qubits[14]),
#         cirq.CNOT(qubits[14], qubits[2]),
#         cirq.CNOT(qubits[14], qubits[4]),
#         cirq.CNOT(qubits[14], qubits[5]),
#         cirq.CNOT(qubits[14], qubits[8]),
#         cirq.CNOT(qubits[14], qubits[9]),
#         cirq.CNOT(qubits[14], qubits[11]),
#         css.Barrier(16).on(*qubits),
#         cirq.T.on_each(*qubits[:-1]),
#         css.Barrier(16).on(*qubits),
#         cirq.H.on_each(*qubits[:-1]),
#         cirq.MeasurementGate(1).on_each(*qubits[:-1]),
#     ])
#     circuit = cirq.align_left(circuit)
#     sub_circuit = circuit[:18] + circuit[21:]  # Remove the t gate moment
#     layout = res.layout.MovementLayout(input_circuit=sub_circuit, num_t_factories=0)
#     for arc in available_arcs:
#         print(arc.__name__)
#         primitive_circuit = cirq.synchronize_terminal_measurements(
#             cirq.align_left(
#                 res.compile_ftqc.ft_compile(layout=layout, arc=res.architecture.MeasureZonesOnly(post_op_correction=False,))
#             )
#         )[1:]
#         estimator = res.estimate.ResourceEstimator(architecture)
#         _time, _moments, _gates = estimator.parallel_circuit_time(primitive_circuit), estimator.parallel_circuit_cost(primitive_circuit), estimator.serial_circuit_cost(primitive_circuit)
#         print(_time)
#         print(_moments)
#         print(_gates)

# distilled_costs()