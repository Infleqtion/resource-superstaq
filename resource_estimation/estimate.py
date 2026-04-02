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
import warnings
from collections import Counter
import cirq
from resource_estimation.cliff_rz import compile_cliff_rz
from resource_estimation.architecture import Architecture
from tqdm import tqdm
from functools import cached_property
from random import choice
from resource_estimation.clifford_t import approx_rz
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ResourceEstimator:
    """
    Class for resource estimator objects defined by the given architecture
    """

    def __init__(self, arc: Architecture):
        self.arc = arc

    def validate_circuit_ops(self, circuit: cirq.Circuit) -> None:
        """
        Checks that the input circuit contains only valid operations and warns of operations still in progress
        """
        unrecognized = [
            op
            for op in dict(Counter([op_.gate for op_ in circuit.all_operations()])).keys()
            if op not in self.arc.primitives
        ]
        if unrecognized:
            error_message = f"""This circuit has gates that are incompatible with the input architecture parameters.\nThe following gates in this circuit are not recognized:"""
            for op in unrecognized:
                error_message += f"\n{str(op)}"
            raise ValueError(error_message)

    def serial_circuit_cost(
        self, circuit: cirq.Circuit, verbose: int = 0, pretty: bool = False
    ) -> dict[cirq.Gate | str, int]:
        """
        Counts up the total physical gates from all logical primitives in the input circuit
        """
        self.validate_circuit_ops(circuit=circuit)
        cost = Counter()
        for op in tqdm(
            circuit.all_operations(),
            total=len(list(circuit.all_operations())),
            colour="cyan",
            disable=not bool(verbose),
        ):
            cost += Counter(self.arc.gate_cost(op))
        if pretty:
            return {
                obj.__name__ if hasattr(obj, "__name__") else str(obj): val
                for obj, val in cost.items()
            }
        else:
            return {op: val for op, val in cost.items()}

    def serial_circuit_time(self, circuit: cirq.Circuit) -> float:
        """
        Adds up the total physical time from all logical primitives in the input circuit
        """
        self.validate_circuit_ops(circuit=circuit)
        return sum(
            map(lambda x: self.arc.total_time(self.arc.gate_cost(x)), circuit.all_operations())
        )

    def parallel_circuit_time(self, circuit: cirq.Circuit, verbose: int = 0) -> float:
        """
        Estimation of the critical path in the input circuit according to the most expensive operation per moment
        """
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            big_time = max(qubit_times[q] for q in op.qubits)
            big_time += self.arc.op_time(op)
            for qubit in op.qubits:
                qubit_times[qubit] = big_time
        return max(qubit_times.values())

    def critical_path(self, circuit: cirq.Circuit, verbose: int = 0) -> list[cirq.Operation]:
        """
        Returns the circuit's critical path in terms of the logical primitive operations
        Is very slow and expensive
        """
        warnings.warn(
            "This function can be very expensive.\nIf you just want the physical operations or circuit time, use `critical_path_ops` or `parallel_circuit_time` instead."
        )
        qubit_paths = {qubit: [] for qubit in circuit.all_qubits()}
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(),
            disable=not verbose,
            total=total_ops,
            colour="cyan",
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.get)
            big_path = qubit_paths[big_qubit]
            big_time = qubit_times[big_qubit]
            big_path.append(op)
            big_time += self.arc.op_time(op)
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path.copy()
                qubit_times[qubit] = big_time
        critical_qubit = max(qubit_times, key=qubit_times.get)
        critical_path = qubit_paths[critical_qubit]
        return critical_path

    def parallel_circuit_cost(
        self, circuit: cirq.Circuit, verbose: int = 0, pretty: bool = False
    ) -> dict[cirq.Gate | str, int]:
        """
        Estimation of the physical operations in critical path of the input circuit according to the most expensive operation per moment
        """
        qubit_paths = {qubit: Counter() for qubit in circuit.all_qubits()}
        qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.get)
            big_time = qubit_times[big_qubit] + self.arc.op_time(op)
            big_path = qubit_paths[big_qubit] + Counter(self.arc.moment_cost(op))
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path
                qubit_times[qubit] = big_time

        big_qubit = max(op_qubits, key=qubit_times.get)
        big_time = qubit_times[big_qubit]
        big_path = qubit_paths[big_qubit]

        if pretty:
            big_path = {
                obj.__name__ if hasattr(obj, "__name__") else str(obj): val
                for obj, val in big_path.items()
            }
        return big_path

    def physical_qubits(self, circuit: cirq.Circuit) -> int:
        """
        Calculates the physical qubit cost of the requested circuit
        """
        return cirq.num_qubits(circuit) * self.arc.patch.num_physical_qubits


class SimplifiedEstimator:  # pragma: no cover
    """
    Simplified resource estimator that doesn't do all the hard parts
    """

    def __init__(self, circuit: cirq.Circuit, atol=1e-8, eps=1e-6):
        self.circuit = circuit
        self.atol = (atol,)
        self.eps = eps
        warnings.warn("This estimator is untested!\nDon't use it unless you are developing it!")

    @cached_property
    def compiled_circuit(self) -> cirq.Circuit:
        return compile_cliff_rz(circuit)

    @cached_property
    def total_gates(self) -> dict:
        return Counter(type(op.gate).__name__ for op in self.compiled_circuit.all_operations())

    @cached_property
    def critical_path(self) -> list[cirq.Operation]:
        all_paths = {qubit: [] for qubit in self.compiled_circuit.all_qubits()}
        for moment in tqdm(self.compiled_circuit.moments):
            for op in moment:
                if len(op.qubits) == 1:
                    qubit = op.qubits[0]
                    all_paths[qubit].append(op)
                elif len(op.qubits) == 2:
                    control, target = op.qubits
                    # Paths combine here so we take the longer one
                    bigger_path = max(all_paths[control], all_paths[target], key=len)
                    bigger_path.append(op)
                    all_paths[control] = bigger_path
                    all_paths[target] = bigger_path
                else:
                    raise ValueError("This should not happen")
        longest_path = max(path for path in all_paths.values())
        return longest_path

    def critical_gates(self) -> Counter:
        def f(x):
            if x in cirq.GateFamily(cirq.S):
                return cirq.S
            elif x in cirq.GateFamily(cirq.Z):
                return cirq.Z
            elif x in cirq.GateFamily(cirq.Rz):
                return cirq.Rz
            elif x in cirq.GateFamily(cirq.H):
                return cirq.H
            elif x in cirq.GateFamily(cirq.X):
                return cirq.X
            elif x in cirq.GateFamily(cirq.MeasurementGate):
                return cirq.MeasurementGate
            elif x in cirq.GateFamily(cirq.CNOT):
                return cirq.CNOT
            else:
                raise ValueError("Unrecognized operation type")

        return Counter(map(f, self.critical_path))

    @cached_property
    def rz_op_cost(self):
        """
        Gets the cost of the median Rz (not the same as the median Rz cost)
        """
        sampled_rz_gate = choice(
            [op for op in self.compiled_circuit.all_operations() if op in cirq.GateFamily(cirq.Rz)]
        )
        theta = sampled_rz_gate.gate.exponent
        eps = self.eps
        approximated_rotation_gates = approx_rz(theta, eps)
        str_to_gate = {
            "T": cirq.T,
            "H": cirq.H,
            "W": cirq.I,
            "S": cirq.S,
            "X": cirq.X,
            "I": cirq.I,
        }
        collected_ops = Counter(map(str_to_gate.get, approximated_rotation_gates))
        # Drop Is and Ws
        collected_ops = {key: val for key, val in collected_ops.items() if key != cirq.I}
        return collected_ops

    def t_cost(self):
        base_gates = self.critical_gates()
        num_rz = base_gates.get(cirq.Rz, 0)
        rz_cost = Counter({key: val * num_rz for key, val in self.rz_op_cost.items()})
        del base_gates[cirq.Rz]
        return base_gates + rz_cost

    def idk(self, architecture: Architecture):
        # Assumes movement architecure
        gates = self.t_cost()
        ops = Counter({})
        time = 0
        for gate_type, occurrences in gates.items():
            if gate_type in cirq.GateFamily(cirq.H):
                sub_total = architecture._h_cost
            elif gate_type in cirq.GateFamily(cirq.S):
                sub_total = architecture._s_cost
            elif gate_type in cirq.GateFamily(cirq.T):
                sub_total = architecture._cultivate_cost
            else:
                # Here we elect to ignore the inexpensive gates
                continue
            time += sub_total["op_time"]
            sub_total = {key: val * occurrences for key, val in sub_total["moment_cost"].items()}
            ops += sub_total
        return ops, time
