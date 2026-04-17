#!/usr/bin/env python3
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

import cirq
import cirq_superstaq as css
from time import time
import textwrap

import argparse
import resource_estimation as res
from resource_estimation.visualizations import C, make_pretty
from resource_estimation.analysis import STR2ARCH


def parse_args():
    parser = argparse.ArgumentParser(description="Resource Estimation Experiment")

    parser.add_argument("file", type=str, help="File in .json format to read as cirq circuit")
    parser.add_argument("--fid", default=0.99, type=float, help="Desired total program fidelity")
    parser.add_argument("--facts", "-f", type=int, default=20, help="Number of T Factories")
    parser.add_argument(
        "--t-path",
        "-t",
        action="store_true",
        help="Flag to generate the logical T path (can be slow)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Turns on verbosity for sub-functions"
    )
    parser.add_argument("--arch", type=str, default="ssm", help="String for architecture to load.")
    parser.add_argument(
        "--fold",
        action="store_true",
        help="Use folded cultivation (Errors for non-movement architectures)",
    )
    parser.add_argument("--nosave", action="store_true", help="Turn off automatic saving")
    parser.add_argument(
        "--code-distance",
        "-d",
        type=int,
        default=0,
        help="Code distance override. Must also override `cultivation-repetition` and `error-per-rz`",
    )
    parser.add_argument(
        "--cultivation-repetition",
        "-cr",
        type=int,
        default=0,
        help="Cultivation repetition override. Must also override `code-distance` and `error-per-rz`",
    )
    parser.add_argument(
        "--error-per-rz",
        "-erz",
        type=float,
        default=0.0,
        help="Synthesis error override. Must also override `code-distance` and `cultivation-repetition`",
    )
    parser.add_argument(
        "--error-per-cult",
        "-epc",
        type=float,
        default=0.0,
        help="Approximate error per cultivation derived elsewhere. Must also override other parameters",
    )
    return parser.parse_args()


def main(args=None) -> int:
    args = args or parse_args()
    file, fid, facts, verbose, arch_name, fold_cultiv = (
        args.file,
        args.fid,
        args.facts,
        args.verbose,
        args.arch,
        args.fold,
    )
    if arch_name in ["sc", "dsnm"] and fold_cultiv:
        raise ValueError("Can't fold with nearest neighbor architectures")
    if sum(
        [
            args.code_distance > 0,
            args.cultivation_repetition > 0,
            args.error_per_rz > 0,
            args.error_per_cult > 0,
        ]
    ) not in [0, 4]:
        raise ValueError(
            "If any of --code-distance, --cultivation-repetition, --error-per-rz, "
            "or --error-per-cult is overridden, all must be overridden"
        )

    # Flag to note when we are going to want to overwrite the error pipeline
    overwrite_error_params = args.code_distance > 0

    report = res.analysis.Report(
        filename=file,
        program_fidelity=fid,
        num_factories=facts,
        arch_name=arch_name,
        fold_cultiv=fold_cultiv,
    )
    t0 = time()
    if overwrite_error_params:
        pass
    else:
        total_allowed_error = 1 - fid
        synthesis_error = total_allowed_error / 2
        gate_error = total_allowed_error / 2

    input_circuit = cirq.read_json(file)
    report.load_time = time() - t0
    print(report.sub_report("Inputs"))

    t1 = time()
    barriers = [
        (idx, op)
        for idx, moment in enumerate(input_circuit)
        for op in moment
        if isinstance(op.gate, css.Barrier)
    ]
    input_circuit.batch_remove(barriers)
    rz_circuit = res.cliff_rz.compile_cliff_rz(circuit=input_circuit, atol=1e-8)
    rz_circuit_width = cirq.num_qubits(rz_circuit)
    rz_circuit_depth = len(rz_circuit)
    t2 = time()

    report.rz_width = rz_circuit_width
    report.rz_depth = rz_circuit_depth
    report.rz_gates = len([op for op in rz_circuit.all_operations() if type(op.gate) is cirq.Rz])
    report.non_rz_gates = len(list(rz_circuit.all_operations())) - report.rz_gates
    report.rz_time = t2 - t1
    print(report.sub_report("Clifford + RZ"))

    t1 = time()
    if overwrite_error_params:
        rz_gates, other_gates = res.analysis.break_up_ops(cliff_rz_circuit=rz_circuit)
        eps = args.error_per_rz
    else:
        eps, rz_gates, other_gates = res.analysis.get_eps(
            rz_circuit, approximation_fidelity=1 - synthesis_error
        )

    clifford_t_circuit = res.clifford_t.compile_cirq_to_clifford_t(
        rz_circuit, eps=eps, verbose=verbose
    )
    clifford_t_circuit = clifford_t_circuit.transform_qubits(
        {
            qubit: cirq.LineQubit(i)
            for i, qubit in enumerate(sorted(clifford_t_circuit.all_qubits()))
        }
    )
    t2 = time()
    if args.t_path:
        t_path = res.analysis.get_t_path(circuit=clifford_t_circuit, verbose=verbose)
        t3 = time()
    else:
        print("Skipped T Path Generation")

    report.eps = eps
    report.t_gates = len(
        [op for op in clifford_t_circuit.all_operations() if op.gate in cirq.GateFamily(cirq.T)]
    )
    report.non_t_gates = len(list(clifford_t_circuit.all_operations())) - report.t_gates
    report.cliff_t_width = cirq.num_qubits(clifford_t_circuit)
    report.cliff_t_depth = len(clifford_t_circuit)
    report.cliff_t_time = t2 - t1

    print(report.sub_report("Clifford + T"))

    if args.t_path:
        print(
            textwrap.dedent(f"""
        {C.OKGREEN}Generated T Path in {t3 - t2:.3e} seconds{C.END}
        T Path Summary:
          - Total Operations:         {C.MAGENTA}{len(t_path)}{C.END}
          - Total T Gates:            {C.MAGENTA}{sum(op.gate in cirq.GateFamily(cirq.T) for op in t_path)}{C.END}
        """).strip()
        )

    t1 = time()
    if overwrite_error_params:
        cultivation_repetition = args.cultivation_repetition
        distance = args.code_distance
        expected_fidelity = 1 - res.analysis.error_estimate(
            code_distance=distance,
            error_per_rz=eps,
            error_per_cult=args.error_per_cult,
            num_rz_gates=rz_gates,
            num_clifford=other_gates,
        )
    else:
        cultivation_repetition, distance, gates, expected_fidelity = (
            res.analysis.get_important_information(
                clifford_t_circuit=clifford_t_circuit, pfid=1 - gate_error
            )
        )
        if fold_cultiv:
            cultivation_repetition = 10  # Based on WORST case in Figure 2a of [Fold-transversal surface code cultivation](https://arxiv.org/pdf/2509.05212)
    t2 = time()

    report.cultivation_repetition = cultivation_repetition
    report.distance = distance
    report.expected_fidelity = expected_fidelity
    report.qec_time = t2 - t1
    print(report.sub_report("QEC Parameters"))

    if fold_cultiv:
        arch = STR2ARCH[arch_name](
            d=distance, cultivation_repetition=cultivation_repetition, fold_cultiv=fold_cultiv
        )
    else:
        arch = STR2ARCH[arch_name](d=distance, cultivation_repetition=cultivation_repetition)

    t1 = time()
    if isinstance(arch, res.architecture.DefaultMovement):
        layt = res.layout.MovementLayout(num_t_factories=facts, input_circuit=clifford_t_circuit)
    else:
        layt = res.layout.FactorySandwich(
            input_circuit=clifford_t_circuit, num_t_factories=facts, num_s_factories=facts
        )
    primitive_circuit = res.compile_ftqc.ft_compile(arc=arch, layout=layt, verbose=verbose)
    t2 = time()

    report.primitive_width = cirq.num_qubits(primitive_circuit)
    report.primitive_depth = len(primitive_circuit)
    report.compile_time = t2 - t1
    print(report.sub_report("FT Compiled Circuit"))

    t1 = time()
    est = res.estimate.ResourceEstimator(arc=arch)
    serial_gate_counts = est.serial_circuit_cost(primitive_circuit, pretty=False, verbose=verbose)
    serial_gate_times = {
        key: val * arch.phys_gate_times[key] for key, val in serial_gate_counts.items()
    }
    total_time_serial = sum(serial_gate_times.values())
    parallel_gate_counts = est.parallel_circuit_cost(
        primitive_circuit, pretty=False, verbose=verbose
    )
    parallel_gate_times = {
        key: val * arch.phys_gate_times[key] for key, val in parallel_gate_counts.items()
    }
    total_time_parallel = sum(parallel_gate_times.values())
    physical_qubits = est.physical_qubits(primitive_circuit)
    t2 = time()

    report.gates_serial = {
        make_pretty(gate): (serial_gate_counts[gate], gate_time)
        for gate, gate_time in serial_gate_times.items()
    }
    report.gates_parallel = {
        make_pretty(gate): (parallel_gate_counts[gate], gate_time)
        for gate, gate_time in parallel_gate_times.items()
    }
    report.time_serial = total_time_serial
    report.time_parallel = total_time_parallel
    report.physical_qubits = physical_qubits
    report.volume = total_time_parallel * physical_qubits
    report.resource_time = t2 - t1
    report.total_time = time() - t0
    print(report.sub_report("Resource Estimation"))
    print(
        f"\n{C.OKGREEN}Script Executed in {C.END}{C.YELLOW}{time() - t0:.3e}{C.END}{C.OKGREEN} seconds{C.END}\n"
    )

    print(report.report())
    if not args.nosave:
        report.save()
    return 0


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
