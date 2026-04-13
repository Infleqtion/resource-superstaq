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
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import partial
from tqdm import tqdm
from resource_estimation.architecture import (
    DefaultMovement,
    DefaultLattice,
    DualSpeciesMovement,
    MeasureZonesOnly,
    Superconductor,
)
from resource_estimation.visualizations import C, boxed_header
import cirq
from cirq_superstaq import Barrier
from collections import Counter
import numpy as np
import warnings


"""
This file should contain the important functions and classes for doing full-stack resource analysis.
Some things that should be here include
- the function that determines code distance and cultivation repetition
- the function that gets the approximation epsilon
- the class that aggregates and stores the information generated during resource estimation
- Maybe others?
"""


STR2ARCH = {
    "ssm": partial(DefaultMovement, idling=False, post_op_correction=True),
    "dsnm": partial(DefaultLattice, idling=False, post_op_correction=True),
    "dsm": partial(DualSpeciesMovement, idling=False, post_op_correction=True),
    "mzo": partial(MeasureZonesOnly, idling=False, post_op_correction=True),
    "sc": partial(Superconductor, idling=False, post_op_correction=True),
}

try:
    WIDTH, _ = shutil.get_terminal_size()
except OSError:  # pragma: no cover
    WIDTH = 80


def get_eps(
    cliff_rz_circuit: cirq.Circuit, approximation_fidelity: float
) -> tuple[float, int, int]:
    """
    Gets the per angle rotation approximation parameter epsilon such that the total product of all Rz gate fidelities is under the input approximation fidelity.
    Returns maximum allowable approximation error for T gate synthesis
    """
    total_ops = len(list(cliff_rz_circuit.all_operations()))
    rz_gates = len([op for op in cliff_rz_circuit.all_operations() if isinstance(op.gate, cirq.Rz)])
    other_gates = total_ops - rz_gates
    if rz_gates == 0:
        return 0.0, rz_gates, other_gates
    min_fidelity = approximation_fidelity ** (1 / rz_gates)
    max_error = 1 - min_fidelity
    return max_error, rz_gates, other_gates


def surface_code_fidelity(d, A=0.03, pth=0.0057, p=0.001) -> float:
    """
    Fidelity of surface code operations according to the Fowler paper (Eq 11 of https://web.physics.ucsb.edu/~martinisgroup/papers/Fowler2012.pdf)
    """
    return 1 - A * (p / pth) ** ((d + 1) // 2)


def get_t_path(circuit: cirq.Circuit, verbose=True):
    """
    Get the T Path of a logical circuit
    Good for comparing with cost model resource estimations
    """
    qubit_paths = {qubit: [] for qubit in circuit.all_qubits()}
    qubit_times = {qubit: 0 for qubit in circuit.all_qubits()}
    for op in tqdm(list(circuit.all_operations()), disable=not verbose, colour="cyan"):
        op_qubits = op.qubits
        big_qubit = max(op_qubits, key=lambda qubit: qubit_times[qubit])
        big_path = qubit_paths[big_qubit]
        big_time = qubit_times[big_qubit]
        big_path.append(op)
        if op in cirq.GateFamily(cirq.T):
            big_time += 2
        else:
            big_time += 1
        for qubit in op_qubits:
            qubit_paths[qubit] = big_path.copy()
            qubit_times[qubit] = big_time
    critical_qubit = max(qubit_times, key=qubit_times.get)
    critical_path = qubit_paths[critical_qubit]
    return critical_path


def get_important_information(
    clifford_t_circuit: cirq.Circuit,
    pfid=0.99,
) -> tuple[int, int, Counter, float]:
    """
    Gets important information for setting sertain assumptions for error correction.
    Given a Clifford + T circuit and a desired program fidelity, which levels of T gate will be needed to stay under budget.
    If neither of the two fidelities is sufficient, warn the user and use the more stringent option.
    With the T fidelity set, the program then calculates the minimal surface code distance needed to stay under the error budget.
    If the error budget is already maxed out, it chooses the distance that contributes less than half of the total error.
    The distances can be overridden with `manual_distance` and `manual_cult_rep`.
    This are admittedly shaky assumptions to get the numbers, but they are at least done in good faith, and I can explain why they were chosen.
    # TODO: Explain why they were chosen
    """
    gates = Counter(op.gate for op in clifford_t_circuit.all_operations())
    t_gates = gates.get(cirq.T, 0)
    other_gates = sum(gates.values()) - t_gates

    # What do I want?
    # (f_t)^(t_gates) * (f_op)^(other_gates) > f
    # (t_gates)log(f_t) + (other_gates)log(f_op) > log(f)
    log_pfid = np.log(pfid)
    weak_t_fidelity = t_gates * np.log(1 - 3 * 10**-6)
    strong_t_fidelity = t_gates * np.log(1 - 2 * 10**-9)
    if weak_t_fidelity > log_pfid:
        t_fidelity = weak_t_fidelity
        cultivation_repetition = 1 / (1 - 0.8)  # 80% discard rate
        over_budget = False
    elif strong_t_fidelity > log_pfid:
        t_fidelity = strong_t_fidelity
        cultivation_repetition = 1 / (1 - 0.99)  # 99% discard rate
        over_budget = False
    else:
        t_fidelity = strong_t_fidelity
        cultivation_repetition = 1 / (1 - 0.99)
        warnings.warn(
            f"Cultivation Error Options of 1e-6 and 1e-9 are not sufficient for desired program fidelity of {pfid}.\nUsing 1e-9 numbers."
        )
        over_budget = True
    if over_budget:
        new_log_pfid = np.log(1 / 2 + pfid / 2)
    else:
        new_log_pfid = log_pfid - t_fidelity

    # Now we need d such that (other_gates)*log(F(d)) + t_fidelity > log(pfid) (if possible)
    for distance in range(1, 33, 2):
        if other_gates * np.log(surface_code_fidelity(distance)) > new_log_pfid:
            break
    if distance == 31:
        warnings.warn("Max code distance 31 reached")

    expected_fidelity = np.exp(t_fidelity + other_gates * np.log(surface_code_fidelity(distance)))
    return int(cultivation_repetition), distance, gates, expected_fidelity


def break_up_ops(cliff_rz_circuit: cirq.Circuit) -> tuple[int, int]:
    """
    Counts operations in Clifford + Rz circuit according to Rz Gates (continuous angle rotations) and Cliffords
    """
    total_ops = len(list(cliff_rz_circuit.all_operations()))
    num_rz_gates = len([op for op in cliff_rz_circuit.all_operations() if type(op.gate) is cirq.Rz])
    num_clifford = total_ops - num_rz_gates
    return num_rz_gates, num_clifford


def error_estimate(
    code_distance: int,
    error_per_rz: float,
    error_per_cult: float,
    num_rz_gates: int,
    num_clifford: int,
    transversal_s_gate: bool = True,
    t_fit_param: float = 4.8,  # fit parameter from synthesis plot for T
    c_fit_param: float = 7.8,  # fit parameter from synthesis plot for H, S
    hw_noise: float = 0.001,
) -> float:
    # Recast for vectorized operations
    code_distance = np.asarray(code_distance)
    error_per_rz = np.asarray(error_per_rz)
    error_per_cult = np.asarray(error_per_cult)

    # The fidelity just from approximating each Rz gate up to the given error
    synthesis_fidelity = (1 - error_per_rz) ** (num_rz_gates)
    # The number of T gates based on the Solivay-Kitaev algorithm performance
    num_t_gates = abs(t_fit_param * np.log(error_per_rz)) * num_rz_gates
    num_c_gates = abs(c_fit_param * np.log(error_per_rz)) * num_rz_gates
    # The number of extra operations from Teleportation
    code_ops_from_teleportation = num_t_gates * 2.5  # CNOT, Measure, and S (50% of the time)
    # Potential extra gates from S gate Teleportation
    if not transversal_s_gate:
        # 50% of the time we do CNOT and Measure (since S-Cult is counted above)
        # 50% times 2 operations is 1 operation per T gate
        code_ops_from_teleportation += num_t_gates
    total_code_ops = num_clifford + num_c_gates + code_ops_from_teleportation
    logical_op_fidelity = surface_code_fidelity(code_distance, p=hw_noise) ** total_code_ops

    cultivation_fidelity = (1 - error_per_cult) ** num_t_gates

    final_fidelity = synthesis_fidelity * logical_op_fidelity * cultivation_fidelity
    return 1 - final_fidelity


@dataclass
class Report:
    """
    Class for containing information about a resource estimate to be saved and reviewed later.
    This object
    """

    ## Inputs
    filename: str
    program_fidelity: float
    num_factories: int
    arch_name: str
    fold_cultiv: bool = None
    load_time: float = None

    ## Outputs
    # Clifford + RZ
    rz_width: int = None
    rz_depth: int = None
    rz_gates: int = None
    non_rz_gates: int = None
    rz_time: float = None

    # Clifford + T
    eps: float = None
    t_gates: int = None
    non_t_gates: int = None
    cliff_t_width: int = None
    cliff_t_depth: int = None
    cliff_t_time: float = None

    # QEC Params
    cultivation_repetition: int | float = None
    distance: int = None
    expected_fidelity: float = None
    qec_time: float = None

    # FT Compiled Circuit
    primitive_width: int = None
    primitive_depth: int = None
    # primitive_gates: dict = None  # This might end up being expensive
    compile_time: float = None

    # Final Resource Estimates
    time_serial: float = None
    time_parallel: float = None
    gates_serial: dict = None
    gates_parallel: dict = None
    physical_qubits: int = None
    volume: float = None
    resource_time: float = None

    total_time: float = None

    def __post_init__(self):
        # This dictionary will be usefull for generating organized reports about the data
        self.info_dict = {
            "Inputs": {
                "Filename": self.filename,
                "Program Fidelity": self.program_fidelity,
                "Archname": self.arch_name,
                "Factories": self.num_factories,
                "Folded": self.fold_cultiv,
                "Time": self.load_time,
            },
            "Clifford + RZ": {
                "Clifford + RZ Width": self.rz_width,
                "Clifford + RZ Depth": self.rz_depth,
                "Total RZ Gates": self.rz_gates,
                "Total non-RZ Gates": self.non_rz_gates,
                "Time": self.rz_time,
            },
            "Clifford + T": {
                "Determined Epsilon": self.eps,
                "Total T Gates": self.t_gates,
                "Total non-T Gates": self.non_t_gates,
                "Clifford + T Width": self.cliff_t_width,
                "Clifford + T Depth": self.cliff_t_depth,
                "Time": self.cliff_t_time,
            },
            "QEC Parameters": {
                "Cultivation Repetition": self.cultivation_repetition,
                "Code Distance": self.distance,
                "Expected Fidelity": self.expected_fidelity,  # TODO: Printing this with .2e formatting usually results in 100%
                "Time": self.qec_time,
            },
            "FT Compiled Circuit": {
                "Compiled Circuit Width": self.primitive_width,
                "Compiled Circuit Depth": self.primitive_depth,
                "Time": self.compile_time,
            },
            "Resource Estimation": {
                "Serial Gate Costs": self.gates_serial,
                "Parallel Gate Costs": self.gates_parallel,
                "Total Time Serial (μs)": self.time_serial,
                "Total Time Parallel (μs)": self.time_parallel,
                "Physical Qubits": self.physical_qubits,
                "Circuit Volume (qubits*μs)": self.volume,
                "Time": self.resource_time,
            },
            "Script Results": {"Time": self.total_time},
        }

    @property
    def arch(self):
        if self.fold_cultiv:
            return STR2ARCH[self.arch_name](
                d=self.distance,
                cultivation_repetition=self.cultivation_repetition,
                fold_cultiv=self.fold_cultiv,
            )
        return STR2ARCH[self.arch_name](
            d=self.distance, cultivation_repetition=self.cultivation_repetition
        )

    def save(self, savedir=Path("")) -> Path:
        stripped_filename = self.filename.split("/")[-1].replace(
            ".json", ""
        )  # Removes directories and .json
        stripped_fidelity = str(self.program_fidelity)[2:]  # Removes the . in .99
        base = f"re_{stripped_filename}-{stripped_fidelity}-{self.arch_name}-{self.num_factories}-{int(self.fold_cultiv)}"
        ext = "json"
        iteration = 0
        filepath = savedir / f"{base}_{iteration}.{ext}"
        while filepath.exists():
            iteration += 1
            filepath = savedir / f"{base}_{iteration}.{ext}"
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"{C.OKGREEN}Saved Report to {C.END}{C.OKCYAN}{str(filepath)}{C.END}")
        return filepath

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            configs = json.load(f)
        return cls.from_dict(configs)

    def header_line(self, title: str) -> str:
        return f"\n{C.BOLD}{C.OKCYAN}{boxed_header(title=title, width=WIDTH)}{C.END}"

    def time_line(self, name: str, seconds: float) -> str:
        return f"{C.OKGREEN}Generated {name} in {C.END}{C.YELLOW}{seconds:.3e}{C.END}{C.OKGREEN} seconds{C.END}"

    def line(self, name: str, value: float | int | str | bool, sep=29) -> str:
        if isinstance(value, bool):
            c, v = "", str(value)
        elif isinstance(value, int):
            c, v = C.MAGENTA, f"{value:.2e}"
        elif isinstance(value, float):
            c, v = C.YELLOW, f"{value:.2e}"
        else:
            c, v = "", str(value)
        return f"{name}:{' ' * (sep - len(name))}{c}{v}{C.END}"

    def line_dict(self, name: str, info_dict: dict, sep=29) -> str:
        sub_str = f"""{name}{" " * (sep - len(name) + 1)}Count     Time (μs)\n"""
        for key, (count, time_us) in info_dict.items():
            count_str = f"{C.MAGENTA}{count:.2e}{C.END}"
            time_str = f"{C.YELLOW}{time_us:.2e}{C.END}"
            sub_str += f" - {key}:{' ' * (sep - 3 - len(key))}{count_str}  {time_str}\n"
        return sub_str

    def report(self) -> str:
        self.__post_init__()
        report_string = """"""
        for header in self.info_dict:
            report_string += self.sub_report(header=header)
        return report_string

    def sub_report(self, header: str) -> str:
        self.__post_init__()
        info = self.info_dict[header].copy()
        report_string = """"""
        report_string += self.header_line(title=header) + "\n"
        report_string += self.time_line(name=header, seconds=info.pop("Time")) + "\n"
        for name, value in info.items():
            if not isinstance(value, dict):
                report_string += self.line(name=name, value=value) + "\n"
            else:
                report_string += self.line_dict(name=name, info_dict=value) + "\n"
        return report_string


class ReportToffoli(Report):  # pragma: no cover
    def __init__(self, filename, program_fidelity, num_factories, arch_name):
        super().__init__(filename, program_fidelity, num_factories, arch_name)
        del self.rz_width
        del self.rz_depth
        del self.rz_gates
        del self.non_rz_gates
        del self.rz_time

        self.toffoli_width = None
        self.toffoli_depth = None
        self.toffoli_gates = None
        self.non_toffoli_gates = None
        self.toffoli_time = None

    def __post_init__(self):
        super().__post_init__()
        del self.info_dict["Clifford + RZ"]
        self.info_dict["Clifford + Toffoli"] = {
            "Clifford + Toffoli Width": self.toffoli_width,
            "Clifford + Toffoli Depth": self.toffoli_depth,
            "Total Toffoli Gates": self.toffoli_gates,
            "Total non-Toffoli Gates": self.non_toffoli_gates,
            "Time": self.toffoli_time,
        }
