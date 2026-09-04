#!/usr/bin/env python3
"""Run a window-decoded SI1000 LER experiment for a logical Clifford circuit.

The circuit input is a plain gate list, one gate per line:

    H 0
    S 1
    CX 0 1

By default, the experiment prepares logical |0...0>, runs the supplied circuit
followed by its logical inverse, and asserts an all-zero logical-Z readout.
Use ``--no-inverse`` for a known identity circuit, such as ten CNOTs, to run
the supplied gates exactly once. The zero-noise preflight refuses to report a
noisy LER unless the final readout is deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_rotated_surface_code_deq.py"
SPEC = importlib.util.spec_from_file_location("generator", GENERATOR_PATH)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@dataclass(frozen=True)
class LogicalGate:
    name: str
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentFiles:
    """The generated inputs and compiled libraries for one LER experiment."""

    noisy_library: Path
    noisy_jit: Path
    ideal_library: Path
    ideal_jit: Path
    program: Path


def experiment_files(work_directory: Path) -> ExperimentFiles:
    """Return the conventional file layout under ``work_directory``."""
    return ExperimentFiles(
        noisy_library=work_directory / "library_si1000.deq",
        noisy_jit=work_directory / "library_si1000.deq.jit",
        ideal_library=work_directory / "library_ideal.deq",
        ideal_jit=work_directory / "library_ideal.deq.jit",
        program=work_directory / "logical_clifford_cycle.deq",
    )


def parse_gate_list(path: Path, *, num_logical_qubits: int) -> tuple[LogicalGate, ...]:
    """Parse H, S, and CX gates while validating their logical-qubit indices."""
    gates: list[LogicalGate] = []
    arities = {"H": 1, "S": 1, "CX": 2}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        fields = raw_line.partition("#")[0].split()
        if not fields:
            continue
        name = fields[0].upper()
        if name not in arities:
            raise ValueError(f"{path}:{line_number}: unsupported gate {name!r}")
        if len(fields) != arities[name] + 1:
            raise ValueError(
                f"{path}:{line_number}: {name} requires {arities[name]} qubit index(es)"
            )
        try:
            qubits = tuple(int(field) for field in fields[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: qubit indices must be integers") from error
        if any(qubit < 0 or qubit >= num_logical_qubits for qubit in qubits):
            raise ValueError(
                f"{path}:{line_number}: qubit index outside 0..{num_logical_qubits - 1}"
            )
        if name == "CX" and qubits[0] == qubits[1]:
            raise ValueError(f"{path}:{line_number}: CX control and target must differ")
        gates.append(LogicalGate(name, qubits))
    return tuple(gates)


def inverse_circuit(gates: tuple[LogicalGate, ...]) -> tuple[LogicalGate, ...]:
    """Return the exact inverse, with S† represented by three S gadgets."""
    inverse: list[LogicalGate] = []
    for gate in reversed(gates):
        if gate.name in {"H", "CX"}:
            inverse.append(gate)
        elif gate.name == "S":
            inverse.extend((LogicalGate("S", gate.qubits),) * 3)
        else:
            raise AssertionError(f"unsupported gate {gate.name!r}")
    return tuple(inverse)


def _render_gate(gate: LogicalGate, *, distance: int) -> str:
    if gate.name == "H":
        return f"    LogicalHadamardD{distance} {gate.qubits[0]}"
    if gate.name == "S":
        return f"    LogicalSD{distance} {gate.qubits[0]}"
    if gate.name == "CX":
        return f"    LogicalCNOTD{distance} {gate.qubits[0]} {gate.qubits[1]}"
    raise AssertionError(f"unsupported gate {gate.name!r}")


def render_cycle_program(
    gates: tuple[LogicalGate, ...],
    *,
    distance: int,
    num_logical_qubits: int,
    include_inverse: bool = True,
) -> str:
    """Render a deterministic logical circuit with final Z assertions."""
    name = f"LogicalCliffordCycleD{distance}"
    cycle = (*gates, *inverse_circuit(gates)) if include_inverse else gates
    description = (
        "# The ideal circuit is U†U = I; all final logical-Z outcomes are zero."
        if include_inverse
        else "# The supplied circuit preserves |0...0>; final logical-Z outcomes are zero."
    )
    return "\n".join(
        (
            description,
            f"PROGRAM {name} {{",
            *(f"    PrepareZ {qubit}" for qubit in range(num_logical_qubits)),
            *(_render_gate(gate, distance=distance) for gate in cycle),
            *(f"    MeasureZ {qubit}" for qubit in range(num_logical_qubits)),
            *(
                f"    ASSERT_EQ rec[-{index}] 0"
                for index in range(num_logical_qubits, 0, -1)
            ),
            "}",
            "",
        )
    )


def run_deq(*arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "deq", *arguments]
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def ler_arguments(
    *,
    library_path: Path,
    program_path: Path,
    jit_path: Path,
    distance: int,
    decoder: str,
    decoder_config: str | None,
    buffer_radius: int,
    lookahead_radius: int,
    shots: int,
    errors: int,
    batch_size: int,
    jobs: int,
    seed: int | None = None,
) -> tuple[str, ...]:
    """Return the DEQ JIT/window-decoder invocation for one LER experiment."""
    return (
        "simulate",
        "ler",
        str(library_path),
        str(program_path),
        "--program",
        f"LogicalCliffordCycleD{distance}",
        "--jit",
        str(jit_path),
        "--coordinator",
        "window",
        "--coordinator-config",
        json.dumps(
            {
                "buffer_radius": buffer_radius,
                "lookahead_radius": lookahead_radius,
            }
        ),
        "--decoder",
        decoder,
        *(
            ("--decoder-config", decoder_config)
            if decoder_config is not None
            else ()
        ),
        "--shots",
        str(shots),
        "--errors",
        str(errors),
        "--batch-size",
        str(batch_size),
        "--jobs",
        str(jobs),
        *(("--seed", str(seed)) if seed is not None else ()),
    )


def logical_error_count(simulation_output: str) -> int:
    """Extract DEQ's logical-error count, rejecting an unrecognised report."""
    match = re.search(r"^  Logical errors:\s+(\d+)$", simulation_output, re.MULTILINE)
    if not match:
        raise RuntimeError("could not find DEQ's logical-error count in its output")
    return int(match.group(1))


def run_zero_noise_preflight(
    *,
    library_path: Path,
    jit_path: Path,
    program_path: Path,
    distance: int,
    decoder: str,
    decoder_config: str | None,
    buffer_radius: int,
    lookahead_radius: int,
    shots: int,
    batch_size: int,
    jobs: int,
    seed: int | None = None,
) -> None:
    """Compile and require a deterministic, window-decoded zero-noise cycle."""
    run_deq("transpile", str(library_path), "--out", str(jit_path), "--jobs", str(jobs))
    print("Running zero-noise logical-cycle preflight...", flush=True)
    result = run_deq(
        *ler_arguments(
            library_path=library_path,
            program_path=program_path,
            jit_path=jit_path,
            distance=distance,
            decoder=decoder,
            decoder_config=decoder_config,
            buffer_radius=buffer_radius,
            lookahead_radius=lookahead_radius,
            shots=shots,
            errors=1,
            batch_size=batch_size,
            jobs=jobs,
            seed=seed,
        ),
        capture_output=True,
    )
    print(result.stdout, end="")
    if logical_error_count(result.stdout):
        raise SystemExit(
            "Zero-noise preflight found logical failures; refusing to report "
            "a noisy LER. Inspect the generated cycle or pass "
            "--skip-ideal-check to override this guard."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--num-logical-qubits", type=int, required=True)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--noise-p", type=float, required=True)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--errors", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="shots submitted to each decoder worker at a time (default: 100)",
    )
    parser.add_argument("--decoder", default="black-box-relay-bp")
    parser.add_argument(
        "--decoder-config",
        help="JSON configuration passed to the selected decoder",
    )
    parser.add_argument("--buffer-radius", type=int, default=2)
    parser.add_argument("--lookahead-radius", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        help="fixed DEQ sampler seed; pass this when comparing decoders",
    )
    parser.add_argument(
        "--ideal-shots",
        type=int,
        default=32,
        help="zero-noise preflight shots (default: 32)",
    )
    parser.add_argument(
        "--skip-ideal-check",
        action="store_true",
        help="run the noisy experiment without first requiring a zero-noise pass",
    )
    parser.add_argument(
        "--no-inverse",
        action="store_true",
        help="run only the supplied circuit; it must preserve the final all-zero Z readout",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/surface-code-deq-ler"))
    args = parser.parse_args()

    if args.num_logical_qubits < 1:
        parser.error("--num-logical-qubits must be positive")
    if (
        args.shots < 1
        or args.errors < 1
        or args.batch_size < 1
        or args.jobs < 1
        or args.ideal_shots < 1
    ):
        parser.error(
            "--shots, --errors, --batch-size, --jobs, and --ideal-shots must be positive"
        )
    if args.buffer_radius < 0 or args.lookahead_radius < 0:
        parser.error("window radii must be non-negative")

    gates = parse_gate_list(
        args.circuit, num_logical_qubits=args.num_logical_qubits
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    files = experiment_files(args.work_dir)
    files.noisy_library.write_text(
        generator.inject_si1000_noise(
            generator.render_surface_code_library(args.distance), args.noise_p
        )
    )
    files.program.write_text(
        render_cycle_program(
            gates,
            distance=args.distance,
            num_logical_qubits=args.num_logical_qubits,
            include_inverse=not args.no_inverse,
        )
    )

    if not args.skip_ideal_check:
        files.ideal_library.write_text(generator.render_surface_code_library(args.distance))
        run_zero_noise_preflight(
            library_path=files.ideal_library,
            jit_path=files.ideal_jit,
            program_path=files.program,
            distance=args.distance,
            decoder=args.decoder,
            decoder_config=args.decoder_config,
            buffer_radius=args.buffer_radius,
            lookahead_radius=args.lookahead_radius,
            shots=args.ideal_shots,
            batch_size=args.batch_size,
            jobs=args.jobs,
            seed=args.seed,
        )

    run_deq(
        "transpile",
        str(files.noisy_library),
        "--out",
        str(files.noisy_jit),
        "--jobs",
        str(args.jobs),
    )
    run_deq(
        *ler_arguments(
            library_path=files.noisy_library,
            program_path=files.program,
            jit_path=files.noisy_jit,
            distance=args.distance,
            decoder=args.decoder,
            decoder_config=args.decoder_config,
            buffer_radius=args.buffer_radius,
            lookahead_radius=args.lookahead_radius,
            shots=args.shots,
            errors=args.errors,
            batch_size=args.batch_size,
            jobs=args.jobs,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
