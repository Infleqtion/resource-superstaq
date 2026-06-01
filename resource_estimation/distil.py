import magic_state_distillation as msd
from collections import Counter
from typing import Literal
import cirq
# from compile_ftqc import ft_compile
# from architecture import Architecture
# from layout import MovementLayout
# from estimate import ResourceEstimator
import resource_estimation as res
from resource_estimation import architecture

def distil(
        arch,
        state_type: str='H',
) -> dict[Literal["serial", "parallel"], Counter[cirq.Gate, int]]:
    """
    Generates the logical qubit resources required for 15-to-1 magic
    state distilation.
    """
    if state_type == 'H':       # Uses BK notation, prepares T states
        circuit = msd.msd_15_to_1()
        # ssm = arch(
        #     d=11,  # Rotated Surface Code code distance
        #     idling=False,  # Include Syndrome Extraction on idling qubits in compiled circuit
        #     post_op_correction=True,  # Turn on or off Syndrome Extraction after transversal operations
        #     syndrome_rounds=1,  # Rounds of Syndrome Extraction after transversal operations
        #     cultivation_repetition=5,  # Expected repetitions of the cultivation circuit to get a successful T state
        # )
        movement = arch.movement
        if movement:
            layout = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=5)
        else:
            layout = res.layout.FactorySandwich(input_circuit=circuit, num_t_factories=5, num_s_factories=5)
        primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=False)
        estimator = res.estimate.ResourceEstimator(arc=arch)
        parallel_gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=True)
        serial_gate_cost = estimator.serial_circuit_cost(primitive_circuit, pretty=True)
        circuit_time = estimator.parallel_circuit_time(primitive_circuit)
        resources = {'op_time': circuit_time, 'moment_cost': Counter(parallel_gate_cost), 'gate_cost': Counter(serial_gate_cost)}
    elif state_type == 'Y':
        circuit = msd.msd_7_to_1()
        movement = arch.movement
        if movement:
            layout = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=5)
        else:
            layout = res.layout.FactorySandwich(input_circuit=circuit, num_t_factories=5, num_s_factories=5)
        primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=False)
        estimator = res.estimate.ResourceEstimator(arc=arch)
        parallel_gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=True)
        serial_gate_cost = estimator.serial_circuit_cost(primitive_circuit, pretty=True)
        circuit_time = estimator.parallel_circuit_time(primitive_circuit)
        resources = {'op_time': circuit_time, 'moment_cost': Counter(parallel_gate_cost), 'gate_cost': Counter(serial_gate_cost)}
    elif state_type == 'T':
        circuit = msd.msd_5_to_1()
        movement = arch.movement
        if movement:
            layout = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=5)
        else:
            layout = res.layout.FactorySandwich(input_circuit=circuit, num_t_factories=5, num_s_factories=5)
        primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=arch, verbose=False)
        estimator = res.estimate.ResourceEstimator(arc=arch)
        parallel_gate_cost = estimator.parallel_circuit_cost(primitive_circuit, pretty=True)
        serial_gate_cost = estimator.serial_circuit_cost(primitive_circuit, pretty=True)
        circuit_time = estimator.parallel_circuit_time(primitive_circuit)
        resources = {'op_time': circuit_time, 'moment_cost': Counter(parallel_gate_cost), 'gate_cost': Counter(serial_gate_cost)}
    else:
        raise ValueError("state type must be either H, T, or Y type.")
    return resources


if __name__ == '__main__':
    # x = msd.msd_15_to_1()
    # ssm = res.architecture.DefaultMovement(
    #     d=11,  # Rotated Surface Code code distance
    #     idling=False,  # Include Syndrome Extraction on idling qubits in compiled circuit
    #     post_op_correction=True,  # Turn on or off Syndrome Extraction after transversal operations
    #     syndrome_rounds=1,  # Rounds of Syndrome Extraction after transversal operations
    #     cultivation_repetition=5,  # Expected repetitions of the cultivation circuit to get a successful T state
    # )
    ssm = res.architecture.DefaultLattice()
    x = distil(ssm, state_type='H')
    print(x)

