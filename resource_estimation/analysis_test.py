import cirq
import math
import pytest
from cirq_superstaq import Barrier
import numpy as np

import resource_estimation.analysis as analysis
import resource_estimation.architecture as arc


@pytest.fixture
def report():
    return analysis.Report(
        filename="dummy_file.json",
        program_fidelity=0.99,
        num_factories=10,
        arch_name="ssm",
        fold_cultiv=True,
        cultivation_repetition=1,
        distance=11,
    )


@pytest.fixture
def populated_report(report):
    report.load_time = 1.0
    report.rz_width = 10
    report.rz_depth = 10
    report.rz_gates = 10
    report.non_rz_gates = 10
    report.rz_time = 1.0
    report.eps = 1.0
    report.t_gates = 10
    report.non_t_gates = 10
    report.cliff_t_width = 10
    report.cliff_t_depth = 10
    report.cliff_t_time = 1.0
    report.expected_fidelity = 1.0
    report.qec_time = 1.0
    report.primitive_width = 10
    report.primitive_depth = 10
    report.compile_time = 1.0
    report.gates_serial = {}
    report.gates_parallel = {}
    report.time_serial = 1.0
    report.time_parallel = 1.0
    report.physical_qubits = 10
    report.volume = 10
    report.resource_time = 1.0
    report.total_time = 1.0
    return report


def test_get_eps():
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit([cirq.Rz(rads=1.23).on(q)] * 5 + [cirq.H.on(q)] * 10)
    approximation_fidelity = 0.59049  # exactly 0.90**5
    max_error, rz_gates, other_gates = analysis.get_eps(circuit, approximation_fidelity)
    expected_error, expected_rz_gates, expected_other_gates = 0.10, 5, 10
    assert math.isclose(max_error, expected_error, rel_tol=1e-9, abs_tol=1e-12)
    assert rz_gates == expected_rz_gates
    assert other_gates == expected_other_gates
    empty_circuit = cirq.Circuit()
    approximation_fidelity = 0.99
    max_error, rz_gates, other_gates = analysis.get_eps(empty_circuit, approximation_fidelity)
    assert max_error == 0
    assert rz_gates == 0
    assert other_gates == 0


def test_save_and_load_round_trip(report, tmp_path):
    filepath = report.save(tmp_path)

    assert filepath.exists()
    assert filepath.parent == tmp_path

    loaded_report = analysis.Report.load(filepath)
    assert loaded_report.info_dict == report.info_dict


def test_save_increments_filename(report, tmp_path):
    filepath1 = report.save(tmp_path)
    filepath2 = report.save(tmp_path)

    assert filepath1.parent == tmp_path
    assert filepath2.parent == tmp_path
    assert filepath1.name == "re_dummy_file-99-ssm-10-1_0.json"
    assert filepath2.name == "re_dummy_file-99-ssm-10-1_1.json"


def test_arch(report):
    architecture = report.arch
    assert isinstance(architecture, arc.DefaultMovement)
    assert architecture.fold_cultiv

    report.fold_cultiv = False
    architecture = report.arch

    assert isinstance(architecture, arc.DefaultMovement)
    assert not architecture.fold_cultiv


def test_report_contains_expected_sections(populated_report):
    report_str = populated_report.report()

    assert "Inputs" in report_str
    assert "Clifford + RZ" in report_str
    assert "Clifford + T" in report_str
    assert "QEC Parameters" in report_str
    assert "Resource Estimation" in report_str
    assert "dummy_file.json" in report_str
    assert "ssm" in report_str
    assert "1.00e+01" in report_str


def test_line_dict(report):
    info_dict = {
        "key1": (10, 1.0),
        "key2": (100, 2.0),
    }
    line_dict = report.line_dict("test", info_dict)

    assert "test" in line_dict
    assert "Count" in line_dict
    assert "Time" in line_dict
    assert "key1" in line_dict
    assert "key2" in line_dict
    assert "1.00e+01" in line_dict
    assert "1.00e+02" in line_dict
    assert "1.00e+00" in line_dict
    assert "2.00e+00" in line_dict


def test_surface_code_fidelity():
    assert analysis.surface_code_fidelity(100, p=0.0057) == 0.97
    assert (
        analysis.surface_code_fidelity(7)
        < analysis.surface_code_fidelity(9)
        < analysis.surface_code_fidelity(11)
    )
    assert analysis.surface_code_fidelity(100, p=0) == 1


def test_break_up_opss():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.Rz(rads=0.1).on(q), cirq.X.on(q), cirq.H.on(q), cirq.Rz(rads=0.2).on(q)
    )

    assert analysis.break_up_ops(circuit) == (2, 2)


def test_get_important_information_t_paths():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.T.on(q),
        cirq.X.on(q),
        cirq.H.on(q),
    )
    f_weak = 0.99999
    f_strong = 0.9999999
    cultivation_repetition, distance, gates, expected_fidelity = analysis.get_important_information(
        circuit,
        pfid=f_weak,
    )

    assert cultivation_repetition == 5
    assert gates[cirq.T] == 1
    assert gates[cirq.X] == 1
    assert gates[cirq.H] == 1
    assert distance == 11
    assert f_weak < expected_fidelity <= 1
    cultivation_repetition, distance, gates, expected_fidelity = analysis.get_important_information(
        circuit,
        pfid=f_strong,
    )
    assert cultivation_repetition == 99
    assert gates[cirq.T] == 1
    assert gates[cirq.X] == 1
    assert gates[cirq.H] == 1
    assert distance == 15
    assert f_strong < expected_fidelity <= 1


def test_get_important_information_warnings():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit([cirq.T.on(q)] * 10)

    with pytest.warns(UserWarning, match="not sufficient"):
        cultivation_repetition, _, _, _ = analysis.get_important_information(
            circuit,
            pfid=1.0,
        )
    assert cultivation_repetition == 99

    circuit = cirq.Circuit([cirq.H.on(q)])
    with pytest.warns(UserWarning, match="Max code distance"):
        cultivation_repeptition, _, _, _ = analysis.get_important_information(circuit, pfid=1.0)


def test_error_estimate():
    with_transversal = analysis.error_estimate(
        code_distance=9,
        error_per_rz=1e-3,
        error_per_cult=1e-6,
        num_rz_gates=100,
        num_clifford=200,
        transversal_s_gate=True,
    )

    without_transversal = analysis.error_estimate(
        code_distance=9,
        error_per_rz=1e-3,
        error_per_cult=1e-6,
        num_rz_gates=100,
        num_clifford=200,
        transversal_s_gate=False,
    )

    assert 0 <= with_transversal <= 1
    assert 0 <= without_transversal <= 1
    assert without_transversal > with_transversal

    low_rz_error = analysis.error_estimate(
        code_distance=9,
        error_per_rz=1e-4,
        error_per_cult=1e-6,
        num_rz_gates=100,
        num_clifford=200,
    )

    high_rz_error = analysis.error_estimate(
        code_distance=9,
        error_per_rz=1e-3,
        error_per_cult=1e-6,
        num_rz_gates=100,
        num_clifford=200,
    )

    assert 0 <= low_rz_error <= 1
    assert 0 <= high_rz_error <= 1
    assert low_rz_error < high_rz_error

    vector = analysis.error_estimate(
        code_distance=np.array([9]),
        error_per_rz=np.array([1e-3]),
        error_per_cult=np.array([1e-6]),
        num_rz_gates=100,
        num_clifford=200,
    )
    assert np.allclose(vector, np.array([high_rz_error]))


def test_t_path():
    qubits = [*cirq.LineQubit.range(3)]
    circuit = cirq.Circuit(
        cirq.H.on(qubits[0]),
        cirq.T.on_each(qubits[0], qubits[1]),
        cirq.CNOT.on(qubits[0], qubits[1]),
        cirq.T.on(qubits[1]),
        cirq.CNOT.on(qubits[1], qubits[2]),
        cirq.H.on(qubits[1]),
    )
    result = analysis.get_t_path(circuit)
    expectation = [
        cirq.H.on(qubits[0]),
        cirq.T.on(qubits[0]),
        cirq.CNOT.on(qubits[0], qubits[1]),
        cirq.T.on(qubits[1]),
        cirq.CNOT.on(qubits[1], qubits[2]),
        cirq.H.on(qubits[1]),
    ]
    assert expectation == result

    circuit1 = cirq.Circuit(cirq.H.on(qubits[0]), cirq.T.on(qubits[0]), cirq.H.on(qubits[0]))
    circuit2 = cirq.Circuit(
        cirq.CNOT.on(qubits[0], qubits[1]), cirq.T.on(qubits[0]), cirq.CNOT.on(qubits[0], qubits[1])
    )
    path1 = analysis.get_t_path(circuit1)
    path2 = analysis.get_t_path(circuit2)
    assert len(path1) == len(path2)

    circuit = cirq.Circuit(
        [cirq.H.on(qubits[0])] * 8 + [cirq.CNOT.on(qubits[0], qubits[1]), cirq.T.on(qubits[2])]
    )
    expectation = [cirq.H.on(qubits[0])] * 8 + [cirq.CNOT.on(qubits[0], qubits[1])]
    result = analysis.get_t_path(circuit)
    assert expectation == result
