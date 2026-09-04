#!/usr/bin/env python3
"""Find logical-gadget fault weights with ideal terminal readout.

The SI1000 noise is applied only to the generated library. Ideal encoded Bell
preparation, terminal syndrome extraction, and final Choi measurement are
appended afterwards, so they close the experiment without adding noisy faults.

The default report uses Stim's fast graphlike-distance search.  Pass
``--sat-problem-dir`` to additionally emit an exact weighted-DIMACS MaxSAT
problem for each Choi check.  Those problems retain arbitrary detector
hyperedges. Pass ``--rc2`` to solve them in-process using PySAT's RC2 MaxSAT
solver.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile

import stim


ROOT = Path(__file__).parents[1]
CHECKER_PATH = ROOT / "tools" / "check_deq_gadget_semantics.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("choi_checker", CHECKER_PATH)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)

from surface_code_deq.perfect_readout import perfect_readout_gadgets_text


def _perfect_readout_program_text(
    gate: checker.ChoiGate, stabilizer: checker.ChoiStabilizer
) -> str:
    """Run one Choi check with ideal terminal error correction."""
    reference_ports = range(gate.port_count)
    system_ports = range(gate.port_count, 2 * gate.port_count)
    return "\n".join(
        (
            f"PROGRAM PerfectReadout{stabilizer.name} {{",
            *(
                f"    PrepareEncodedBellPair {reference} {system}"
                for reference, system in zip(reference_ports, system_ports)
            ),
            f"    {gate.operation} " + " ".join(map(str, system_ports)),
            *(f"    PerfectSyndromeExtraction {port}" for port in system_ports),
            f"    Check{stabilizer.name} "
            + " ".join(map(str, range(2 * gate.port_count))),
            "    ASSERT_EQ rec[-1] 0",
            "}",
        )
    )


def _source_text(
    gates: tuple[checker.ChoiGate, ...],
    *,
    distance: int,
    prepare_y_boundary_rounds: int | None = None,
) -> tuple[str, tuple[tuple[checker.ChoiGate, checker.ChoiStabilizer], ...]]:
    """Return a noisy-gadget, ideal-readout source and its Choi observables."""
    checks = tuple(
        (gate, stabilizer)
        for gate in gates
        for stabilizer in checker._choi_stabilizers(gate)
    )
    harness = "\n\n".join(
        (
            checker._encoded_bell_preparation_text(distance),
            perfect_readout_gadgets_text(distance),
            *(
                checker._choi_measurement_text(stabilizer, distance=distance)
                for _, stabilizer in checks
            ),
            *(
                _perfect_readout_program_text(gate, stabilizer)
                for gate, stabilizer in checks
            ),
        )
    )
    noisy_library = checker.generator.inject_si1000_noise(
        checker.generator.render_surface_code_library(
            distance, boundary_rounds=prepare_y_boundary_rounds
        ),
        0.001,
    )
    return noisy_library + "\n" + harness, checks


def _gates_at_distance(distance: int) -> tuple[checker.ChoiGate, ...]:
    """Return the Choi specifications with the selected DEQ operation names."""
    operation_names = {
        "LogicalS": f"LogicalSD{distance}",
        "LogicalH": f"LogicalHadamardD{distance}",
        "CNOT": f"FaultTolerantCNOTD{distance}",
    }
    return tuple(
        replace(gate, operation=operation_names[gate.name])
        for gate in checker.CHOI_GATES
    )


def _shortest_graphlike_error(
    circuit: stim.Circuit,
) -> tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
    """Return the full detector model and one shortest graphlike error."""
    error_model = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=1,
    )
    return error_model, error_model.shortest_graphlike_error()


def _shortest_error_sat_problem(circuit: stim.Circuit) -> str:
    """Encode the exact circuit fault-distance problem as WDIMACS MaxSAT.

    Unlike ``_shortest_graphlike_error``, this retains non-graphlike detector
    hyperedges and mutually exclusive outcomes from individual circuit noise
    mechanisms.  The returned string must be solved by a MaxSAT solver; its
    optimal cost is the minimum number of physical fault mechanisms causing an
    undetected logical-observable flip.
    """
    return circuit.shortest_error_sat_problem(format="WDIMACS")


def _shortest_error_rc2_cost(circuit: stim.Circuit) -> int:
    """Return the exact circuit fault distance using PySAT's RC2 solver."""
    try:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
    except ImportError as error:
        raise RuntimeError(
            "exact MaxSAT solving requires the optional 'sat' dependency; "
            "install it with `pip install -e '.[sat]'`"
        ) from error

    problem = WCNF(from_string=_shortest_error_sat_problem(circuit))
    with RC2(problem) as solver:
        if solver.compute() is None:
            raise RuntimeError("Stim's exact fault-distance problem was unsatisfiable")
        return solver.cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        action="append",
        choices=[gate.name for gate in checker.CHOI_GATES],
        help="test only this logical gate (may be repeated)",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=3,
        help="odd rotated-code distance to test (default: 3)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print one physical-fault representative for each shortest error",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="preserve the generated detector-annotated DEQ and Stim files here",
    )
    parser.add_argument(
        "--sat-problem-dir",
        type=Path,
        help=(
            "write one exact non-graphlike WDIMACS MaxSAT problem per Choi "
            "check; solve these files with an external MaxSAT solver"
        ),
    )
    parser.add_argument(
        "--rc2",
        action="store_true",
        help="solve each exact non-graphlike fault-distance problem with PySAT RC2",
    )
    parser.add_argument(
        "--prepare-y-boundary-rounds",
        type=int,
        help="override the number of repeated XXZZ-boundary rounds in PrepareY",
    )
    args = parser.parse_args()
    if args.distance < 3 or args.distance % 2 == 0:
        parser.error("--distance must be odd and at least 3")
    gates = tuple(
        gate
        for gate in _gates_at_distance(args.distance)
        if args.gate is None or gate.name in args.gate
    )
    source, checks = _source_text(
        gates,
        distance=args.distance,
        prepare_y_boundary_rounds=args.prepare_y_boundary_rounds,
    )
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.sat_problem_dir is not None:
        args.sat_problem_dir.mkdir(parents=True, exist_ok=True)
    directory_context = (
        tempfile.TemporaryDirectory(prefix="surface-code-deq-cnot-distance-")
        if args.work_dir is None
        else nullcontext(str(args.work_dir))
    )
    with directory_context as directory:
        directory_path = Path(directory)
        source_path = directory_path / "cnot_perfect_readout.deq"
        source_path.write_text(source)
        for _, stabilizer in checks:
            program = f"PerfectReadout{stabilizer.name}"
            jit_path = directory_path / f"{program}.deq.jit"
            checker.run_deq(
                "transpile",
                str(source_path),
                "--program",
                program,
                "--detectors",
                "--out",
                str(jit_path),
                "--jobs",
                "1",
            )
            stim_path = jit_path.with_suffix("").with_suffix(".stim")
            circuit = stim.Circuit.from_file(stim_path)
            if args.sat_problem_dir is not None:
                sat_path = args.sat_problem_dir / f"{stabilizer.name}.wcnf"
                sat_path.write_text(_shortest_error_sat_problem(circuit))
                print(f"{stabilizer.name}: wrote exact MaxSAT problem {sat_path}")
            if args.rc2:
                exact_cost = _shortest_error_rc2_cost(circuit)
                print(f"{stabilizer.name}: exact fault weight {exact_cost}")
            _, shortest_error = _shortest_graphlike_error(circuit)
            print(f"{stabilizer.name}: graphlike fault weight {len(shortest_error)}")
            if args.explain:
                print(shortest_error)
                for explanation in circuit.explain_detector_error_model_errors(
                    dem_filter=shortest_error,
                    reduce_to_one_representative_error=True,
                ):
                    print(explanation)


if __name__ == "__main__":
    main()
