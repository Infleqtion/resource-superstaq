"""Execute noiseless, JIT-aware encoded-Choi checks through the DEQ CLI.

Run this with the project-local DEQ environment:

    .venv-deq/bin/python tools/check_deq_gadget_semantics.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import stim

ROOT = Path(__file__).parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_rotated_surface_code_deq.py"
SUPPORTED_DISTANCES = (3, 5, 7)
WINDOW_COORDINATOR_CONFIG = '{"buffer_radius": 2, "lookahead_radius": 0}'
SPEC = importlib.util.spec_from_file_location("generator", GENERATOR_PATH)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@dataclass(frozen=True)
class LogicalFactor:
    """One logical Pauli acting on a port."""

    pauli: str
    port: int


@dataclass(frozen=True)
class PauliImage:
    """The image of one input logical Pauli under a logical Clifford gate."""

    input: LogicalFactor
    output: tuple[LogicalFactor, ...]


@dataclass(frozen=True)
class ChoiGate:
    """A Clifford gate specified by its independent logical-Pauli images."""

    name: str
    operation: str
    pauli_images: tuple[PauliImage, ...]

    @property
    def port_count(self) -> int:
        return 1 + max(
            factor.port
            for image in self.pauli_images
            for factor in (image.input, *image.output)
        )


@dataclass(frozen=True)
class ChoiStabilizer:
    """One Choi stabilizer derived from a gate's Pauli-image specification."""

    name: str
    observable: tuple[LogicalFactor, ...]
    port_count: int


CHOI_GATES = (
    ChoiGate(
        name="LogicalS",
        operation="LogicalSD3",
        pauli_images=(
            PauliImage(LogicalFactor("X", 0), (LogicalFactor("Y", 0),)),
            PauliImage(LogicalFactor("Z", 0), (LogicalFactor("Z", 0),)),
        ),
    ),
    ChoiGate(
        name="LogicalH",
        operation="LogicalHadamardD3",
        pauli_images=(
            PauliImage(LogicalFactor("X", 0), (LogicalFactor("Z", 0),)),
            PauliImage(LogicalFactor("Z", 0), (LogicalFactor("X", 0),)),
        ),
    ),
    ChoiGate(
        name="CNOT",
        operation="LogicalCNOTD3",
        pauli_images=(
            PauliImage(
                LogicalFactor("X", 0),
                (LogicalFactor("X", 0), LogicalFactor("X", 1)),
            ),
            PauliImage(LogicalFactor("Z", 0), (LogicalFactor("Z", 0),)),
            PauliImage(LogicalFactor("X", 1), (LogicalFactor("X", 1),)),
            PauliImage(
                LogicalFactor("Z", 1),
                (LogicalFactor("Z", 0), LogicalFactor("Z", 1)),
            ),
        ),
    ),
)


PREPARE_Y_STATE = ChoiStabilizer(
    name="PrepareYState",
    observable=(LogicalFactor("Y", 0),),
    port_count=1,
)


def _choi_stabilizers(gate: ChoiGate) -> tuple[ChoiStabilizer, ...]:
    """Return the independent Choi stabilizers for ``gate``.

    Port numbers in ``PauliImage`` refer to the tested gate.  The generated
    Choi program places its reference patches first and the gate inputs second.
    """
    return tuple(
        ChoiStabilizer(
            name=f"{gate.name}_{image.input.pauli}{image.input.port}",
            observable=(
                image.input,
                *(
                    LogicalFactor(factor.pauli, gate.port_count + factor.port)
                    for factor in image.output
                ),
            ),
            port_count=2 * gate.port_count,
        )
        for image in gate.pauli_images
    )


def _logical_pauli(label: str, *, offset: int, distance: int) -> dict[int, str]:
    """Return one concrete, positive logical-Pauli representative."""
    if label == "X":
        return {offset + distance * row: "X" for row in range(distance)}
    if label == "Z":
        return {offset + column: "Z" for column in range(distance)}
    if label == "Y":
        return {
            offset: "Y",
            **{offset + column: "Z" for column in range(1, distance)},
            **{offset + distance * row: "X" for row in range(1, distance)},
        }
    raise ValueError(f"unsupported logical Pauli {label!r}")


def _stim_pauli(
    num_qubits: int, factors: tuple[LogicalFactor, ...], distance: int
) -> stim.PauliString:
    result = stim.PauliString(num_qubits)
    for factor in factors:
        for qubit, pauli in _logical_pauli(
            factor.pauli, offset=factor.port * distance**2, distance=distance
        ).items():
            if result[qubit] != 0:
                raise ValueError("Choi observable has overlapping representatives")
            result[qubit] = pauli
    return result


def _encoded_bell_preparation_text(distance: int) -> str:
    """Prepare an ideal encoded Bell pair without using a tested gadget."""
    patch = generator.RotatedSurfaceCode(distance, distance)
    size = patch.num_data_qubits
    stabilizers: list[stim.PauliString] = []
    for offset in (0, size):
        for pauli, support in patch.stabilizers():
            stabilizer = stim.PauliString(2 * size)
            for qubit in support:
                stabilizer[offset + qubit] = pauli
            stabilizers.append(stabilizer)
    stabilizers.extend(
        (
            _stim_pauli(
                2 * size,
                (LogicalFactor("X", 0), LogicalFactor("X", 1)),
                distance,
            ),
            _stim_pauli(
                2 * size,
                (LogicalFactor("Z", 0), LogicalFactor("Z", 1)),
                distance,
            ),
        )
    )
    circuit = stim.Tableau.from_stabilizers(
        stabilizers, allow_underconstrained=False
    ).to_circuit()
    allowed_instructions = {"CX", "H"}
    if {instruction.name for instruction in circuit} - allowed_instructions:
        raise AssertionError("encoded Bell preparation used an unsupported instruction")
    return "\n".join(
        (
            "# An ideal encoded Bell pair used only by the Choi test harness.",
            "GADGET PrepareEncodedBellPair {",
            "    R " + " ".join(map(str, range(2 * size))),
            *("    " + line for line in str(circuit).splitlines()),
            f"    OUTPUT {patch.type_name} " + " ".join(map(str, range(size))),
            f"    OUTPUT {patch.type_name} "
            + " ".join(map(str, range(size, 2 * size))),
            "}",
        )
    )


def _choi_measurement_text(stabilizer: ChoiStabilizer, *, distance: int) -> str:
    """Destructively read one Choi stabilizer as a single DEQ readout bit."""
    patch = generator.RotatedSurfaceCode(distance, distance)
    size = patch.num_data_qubits
    observable = _stim_pauli(
        stabilizer.port_count * size, stabilizer.observable, distance
    )
    measured_by_basis: dict[str, list[int]] = {"X": [], "Y": [], "Z": []}
    pauli_names = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    for qubit, pauli in enumerate(observable):
        if pauli:
            measured_by_basis[pauli_names[pauli]].append(qubit)
    operations = {"X": "MX", "Y": "MY", "Z": "M"}
    measurement_count = sum(map(len, measured_by_basis.values()))
    return "\n".join(
        (
            f"GADGET Check{stabilizer.name} {{",
            *(
                f"    INPUT {patch.type_name} "
                + " ".join(map(str, range(port * size, (port + 1) * size)))
                for port in range(stabilizer.port_count)
            ),
            *(
                f"    {operations[basis]} " + " ".join(map(str, targets))
                for basis, targets in measured_by_basis.items()
                if targets
            ),
            "    READOUT "
            + " ".join(f"rec[-{index}]" for index in range(measurement_count, 0, -1)),
            "}",
        )
    )


def _choi_program_text(gate: ChoiGate, stabilizer: ChoiStabilizer) -> str:
    """Test one Choi stabilizer using a fresh encoded Bell state."""
    return "\n".join(
        (
            f"PROGRAM {stabilizer.name} {{",
            *(
                f"    PrepareEncodedBellPair {reference} {system}"
                for reference, system in zip(
                    range(gate.port_count), range(gate.port_count, 2 * gate.port_count)
                )
            ),
            f"    {gate.operation} "
            + " ".join(map(str, range(gate.port_count, 2 * gate.port_count))),
            f"    Check{stabilizer.name} "
            + " ".join(map(str, range(stabilizer.port_count))),
            "    ASSERT_EQ rec[-1] 0",
            "}",
        )
    )


def _logical_effect_source(gates: tuple[ChoiGate, ...] = CHOI_GATES) -> str:
    """Return direct Choi-isomorphism tests for the d=3 Clifford gadgets."""
    distance = 3
    checks = tuple(
        (gate, stabilizer)
        for gate in gates
        for stabilizer in _choi_stabilizers(gate)
    )
    parts = [_encoded_bell_preparation_text(distance)]
    parts.extend(
        _choi_measurement_text(stabilizer, distance=distance)
        for _, stabilizer in checks
    )
    parts.extend(
        _choi_program_text(gate, stabilizer) for gate, stabilizer in checks
    )
    return "\n\n".join(parts)


def _prepare_y_state_source() -> str:
    """Return a direct logical-``+Y`` assertion for the preparation gadget."""
    return "\n\n".join(
        (
            _choi_measurement_text(PREPARE_Y_STATE, distance=3),
            "\n".join(
                (
                    "PROGRAM PrepareYState {",
                    "    PrepareY 0",
                    "    CheckPrepareYState 0",
                    "    ASSERT_EQ rec[-1] 0",
                    "}",
                )
            ),
        )
    )


def run_deq(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "deq", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            "DEQ command failed:\n"
            + " ".join(result.args)
            + "\n\nstdout:\n"
            + result.stdout
            + "\n\nstderr:\n"
            + result.stderr
        )
    if arguments[0] == "transpile" and result.stdout:
        print(result.stdout, end="")
    return result.stdout


def transpile_library(source_path: Path, jit_path: Path) -> None:
    """Compile one library and report DEQ's normal transpilation output."""
    run_deq("transpile", str(source_path), "--out", str(jit_path), "--jobs", "1")


def choi_ler(
    source_path: Path,
    jit_path: Path,
    stabilizer: ChoiStabilizer,
    *,
    shots: int,
    errors: int,
) -> tuple[int, int]:
    """Return logical failures and samples for one frame-aware Choi check.

    Exported Stim alone cannot represent decoder-level ``CONDITIONAL`` or
    ``VIRTUAL`` Pauli frames. ``simulate ler`` evaluates the compiled JIT
    correction matrices, so this includes all such frame updates.
    """
    output = run_deq(
        "simulate",
        "ler",
        str(source_path),
        "--program",
        stabilizer.name,
        "--jit",
        str(jit_path),
        "--coordinator",
        "window",
        "--coordinator-config",
        WINDOW_COORDINATOR_CONFIG,
        "--shots",
        str(shots),
        "--errors",
        str(errors),
        "--jobs",
        "1",
    )
    match = re.search(r"^  Logical errors:\s+(\d+)$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"{stabilizer.name}: DEQ did not report a logical-error count"
        )
    shot_match = re.search(r"^  Shots:\s+(\d+)$", output, re.MULTILINE)
    if shot_match is None:
        raise RuntimeError(f"{stabilizer.name}: DEQ did not report a shot count")
    return int(match.group(1)), int(shot_match.group(1))


def check_choi_stabilizer(
    source_path: Path, jit_path: Path, stabilizer: ChoiStabilizer
) -> None:
    """Require a positive Choi stabilizer through DEQ's frame-aware runtime."""
    logical_errors, _ = choi_ler(
        source_path, jit_path, stabilizer, shots=64, errors=1
    )
    if logical_errors:
        raise AssertionError(
            f"{stabilizer.name}: Choi stabilizer is not deterministically +1"
        )


def run_noisy_choi_ler(
    *,
    gates: tuple[ChoiGate, ...],
    stabilizers: tuple[ChoiStabilizer, ...],
    physical_error_rate: float,
    shots: int,
    errors: int,
    preflight_shots: int,
    work_directory: Path,
) -> None:
    """Measure isolated gadget Choi-stabilizer LERs under SI1000 noise.

    The Bell-pair preparation and final Pauli measurement are appended after
    noise injection. Therefore the reported failures originate in the tested
    logical gadget, rather than in the Choi harness.
    """
    work_directory.mkdir(parents=True, exist_ok=True)
    harness = _logical_effect_source(gates)

    ideal_source_path = work_directory / "library_ideal.deq"
    ideal_jit_path = work_directory / "library_ideal.deq.jit"
    ideal_source_path.write_text(generator.render_surface_code_library(3) + "\n" + harness)
    transpile_library(ideal_source_path, ideal_jit_path)
    for stabilizer in stabilizers:
        logical_errors, _ = choi_ler(
            ideal_source_path,
            ideal_jit_path,
            stabilizer,
            shots=preflight_shots,
            errors=1,
        )
        if logical_errors:
            raise AssertionError(
                f"{stabilizer.name}: zero-noise Choi preflight failed"
            )

    noisy_source_path = work_directory / "library_si1000.deq"
    noisy_jit_path = work_directory / "library_si1000.deq.jit"
    noisy_source_path.write_text(
        generator.inject_si1000_noise(
            generator.render_surface_code_library(3), physical_error_rate
        )
        + "\n"
        + harness
    )
    transpile_library(noisy_source_path, noisy_jit_path)
    print(f"SI1000 p={physical_error_rate:g}; ideal Choi harness; d=3")
    for stabilizer in stabilizers:
        logical_errors, completed_shots = choi_ler(
            noisy_source_path,
            noisy_jit_path,
            stabilizer,
            shots=shots,
            errors=errors,
        )
        print(
            f"  {stabilizer.name}: {logical_errors}/{completed_shots} "
            f"= {logical_errors / completed_shots:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        action="append",
        choices=[gate.name for gate in CHOI_GATES],
        help="run only this logical Clifford gate's Choi checks (may be repeated)",
    )
    parser.add_argument(
        "--noise-p",
        type=float,
        help="run isolated d=3 gadget Choi LERs with SI1000 noise at this rate",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100_000,
        help="maximum noisy shots per Choi stabilizer (default: 100000)",
    )
    parser.add_argument(
        "--errors",
        type=int,
        default=100,
        help="stop each noisy Choi stabilizer after this many failures (default: 100)",
    )
    parser.add_argument(
        "--preflight-shots",
        type=int,
        default=32,
        help="zero-noise shots per Choi stabilizer before a noisy run (default: 32)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="directory for generated noisy and ideal Choi libraries",
    )
    args = parser.parse_args()
    selected_gates = tuple(
        gate for gate in CHOI_GATES if args.gate is None or gate.name in args.gate
    )
    selected_stabilizers = tuple(
        stabilizer
        for gate in selected_gates
        for stabilizer in _choi_stabilizers(gate)
    )

    if args.noise_p is not None:
        if args.shots < 1 or args.errors < 1 or args.preflight_shots < 1:
            parser.error("--shots, --errors, and --preflight-shots must be positive")
        if args.work_dir is not None:
            run_noisy_choi_ler(
                gates=selected_gates,
                stabilizers=selected_stabilizers,
                physical_error_rate=args.noise_p,
                shots=args.shots,
                errors=args.errors,
                preflight_shots=args.preflight_shots,
                work_directory=args.work_dir,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="surface-code-deq-choi-ler-") as directory:
                run_noisy_choi_ler(
                    gates=selected_gates,
                    stabilizers=selected_stabilizers,
                    physical_error_rate=args.noise_p,
                    shots=args.shots,
                    errors=args.errors,
                    preflight_shots=args.preflight_shots,
                    work_directory=Path(directory),
                )
        return

    with tempfile.TemporaryDirectory(prefix="surface-code-deq-") as directory:
        temporary_directory = Path(directory)

        # Transpilation type-checks and validates every gadget in each library.
        for distance in SUPPORTED_DISTANCES:
            source = generator.render_surface_code_library(distance)
            source_path = temporary_directory / f"rotated_surface_code_d{distance}.deq"
            jit_path = temporary_directory / f"rotated_surface_code_d{distance}.jit"
            source_path.write_text(source)
            transpile_library(source_path, jit_path)

        # These encoded Choi programs directly check the signed logical-Pauli
        # maps of the composed d=3 Clifford operations, including entangled
        # reference patches that never pass through the tested gadget.
        source_path = temporary_directory / "logical_effects_d3.deq"
        source_path.write_text(
            generator.render_surface_code_library(3)
            + "\n"
            + _logical_effect_source(selected_gates)
            + "\n"
            + _prepare_y_state_source()
        )
        jit_path = temporary_directory / "logical_effects_d3.jit"
        transpile_library(source_path, jit_path)
        failures: list[str] = []
        for stabilizer in selected_stabilizers:
            try:
                check_choi_stabilizer(source_path, jit_path, stabilizer)
            except AssertionError as error:
                failures.append(str(error))
        try:
            check_choi_stabilizer(source_path, jit_path, PREPARE_Y_STATE)
        except AssertionError as error:
            failures.append(str(error))
        if failures:
            raise AssertionError("\n".join(failures))

    print("DEQ transpilation and logical-effect checks passed.")


if __name__ == "__main__":
    main()
