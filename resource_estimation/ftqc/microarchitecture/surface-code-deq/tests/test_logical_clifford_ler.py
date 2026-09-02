import importlib.util
from pathlib import Path
import sys

import pytest


_runner_path = Path(__file__).parents[1] / "tools" / "run_logical_clifford_ler.py"
_spec = importlib.util.spec_from_file_location("logical_clifford_ler", _runner_path)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)


def test_gate_list_parser_and_inverse_circuit(tmp_path: Path) -> None:
    circuit_path = tmp_path / "random_clifford.txt"
    circuit_path.write_text("# A two-qubit logical Clifford\nH 0\nS 1\nCX 0 1\n")

    gates = runner.parse_gate_list(circuit_path, num_logical_qubits=2)
    assert gates == (
        runner.LogicalGate("H", (0,)),
        runner.LogicalGate("S", (1,)),
        runner.LogicalGate("CX", (0, 1)),
    )
    assert runner.inverse_circuit(gates) == (
        runner.LogicalGate("CX", (0, 1)),
        runner.LogicalGate("S", (1,)),
        runner.LogicalGate("S", (1,)),
        runner.LogicalGate("S", (1,)),
        runner.LogicalGate("H", (0,)),
    )


def test_gate_list_parser_rejects_invalid_gates(tmp_path: Path) -> None:
    circuit_path = tmp_path / "invalid.txt"
    circuit_path.write_text("CX 0 0\n")
    with pytest.raises(ValueError, match="must differ"):
        runner.parse_gate_list(circuit_path, num_logical_qubits=2)


def test_cycle_program_has_a_deterministic_logical_z_readout() -> None:
    program = runner.render_cycle_program(
        (runner.LogicalGate("H", (0,)), runner.LogicalGate("S", (0,))),
        distance=3,
        num_logical_qubits=1,
    )
    assert "PrepareZ 0" in program
    assert program.count("LogicalSD3 0") == 4
    assert program.count("LogicalHadamardD3 0") == 2
    assert "MeasureZ 0" in program
    assert "ASSERT_EQ rec[-1] 0" in program


def test_cycle_program_can_run_a_known_identity_without_an_inverse() -> None:
    program = runner.render_cycle_program(
        (runner.LogicalGate("CX", (0, 1)),) * 10,
        distance=3,
        num_logical_qubits=2,
        include_inverse=False,
    )
    assert program.count("FaultTolerantCNOTD3 0 1") == 10
    assert "U†U" not in program
    assert program.count("ASSERT_EQ") == 2


def test_ler_arguments_request_jit_window_decoding() -> None:
    arguments = runner.ler_arguments(
        library_path=Path("library.deq"),
        program_path=Path("program.deq"),
        jit_path=Path("library.deq.jit"),
        distance=3,
        decoder="black-box-relay-bp",
        decoder_config='{"pre_iter": 10}',
        buffer_radius=2,
        lookahead_radius=0,
        shots=100,
        errors=10,
        batch_size=5,
        jobs=1,
    )
    assert "--jit" in arguments
    assert arguments[arguments.index("--batch-size") + 1] == "5"
    assert arguments[arguments.index("--coordinator") + 1] == "window"
    assert arguments[arguments.index("--decoder-config") + 1] == '{"pre_iter": 10}'
    assert runner.logical_error_count("  Logical errors: 7\n") == 7


def test_experiment_files_use_a_small_predictable_layout(tmp_path: Path) -> None:
    files = runner.experiment_files(tmp_path)
    assert files.program == tmp_path / "logical_clifford_cycle.deq"
    assert files.ideal_library == tmp_path / "library_ideal.deq"
    assert files.noisy_jit == tmp_path / "library_si1000.deq.jit"
