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
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cirq
import pytest

from scripts.analyze import main, parse_args


@pytest.fixture
def circuit_file(tmp_path: Path) -> Path:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H.on_each(q0, q1, q2),
        cirq.CNOT.on(q0, q1),
        cirq.CNOT.on(q0, q2),
        cirq.CNOT.on(q1, q2),
        cirq.T.on(q0),
        cirq.MeasurementGate(1).on_each(q0, q1, q2),
    )
    path = tmp_path / "test_circuit.json"
    with open(path, "w") as f:
        cirq.to_json(circuit, f)
    return path


def make_namespace(**kwargs: str | float | int | None) -> argparse.Namespace:
    # Uses defaults from analyze.py
    defaults = {
        "file": None,
        "fid": 0.99,
        "facts": 20,
        "t_path": False,
        "verbose": False,
        "arch": "ssm",
        "fold": False,
        "nosave": False,
        "code_distance": 0,
        "cultivation_repetition": 0,
        "error_per_rz": 0.0,
        "error_per_cult": 0.0,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["analyze.py", "circuit.json", "--fid", "0.95", "--facts", "10"],
    )
    args = parse_args()

    assert args.file == "circuit.json"
    assert args.fid == 0.95
    assert args.facts == 10
    assert args.t_path is False  # default
    assert args.arch == "ssm"  # default


def test_analyze_defaults(
    circuit_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    args = make_namespace(file=str(circuit_file))
    exit_code = main(args)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Generated Script Results" in captured.out

    expected_saved_file = "re_test_circuit-99-ssm-20-0_0.json"
    assert os.path.isfile(expected_saved_file)

    with open(expected_saved_file) as f:
        saved_data = json.load(f)

    assert saved_data["filename"] == str(circuit_file)
    assert saved_data["arch_name"] == "ssm"
    assert saved_data["num_factories"] == 20
    assert 0 < saved_data["expected_fidelity"] <= 1
    assert saved_data["physical_qubits"] > 0


def test_override_error_params(
    circuit_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code_distance, cultivation_repetition, error_per_rz, error_per_cult = 19, 7, 8e-5, 9e-6
    args = make_namespace(
        file=str(circuit_file),
        code_distance=code_distance,
        cultivation_repetition=cultivation_repetition,
        error_per_rz=error_per_rz,
        error_per_cult=error_per_cult,
    )
    exit_code = main(args)
    assert exit_code == 0
    expected_saved_file = "re_test_circuit-99-ssm-20-0_0.json"
    with open(expected_saved_file) as f:
        saved_data = json.load(f)
    assert saved_data["distance"] == code_distance
    assert saved_data["cultivation_repetition"] == cultivation_repetition
    assert saved_data["eps"] == error_per_rz


def test_bad_override(circuit_file: Path) -> None:
    code_distance, cultivation_repetition, error_per_rz = 19, 7, 8e-5
    args = make_namespace(
        file=str(circuit_file),
        code_distance=code_distance,
        cultivation_repetition=cultivation_repetition,
        error_per_rz=error_per_rz,
    )
    with pytest.raises(ValueError, match="all must be overridden"):
        _ = main(args)


@pytest.mark.parametrize("arch", ("ssm", "dsnm"))
@pytest.mark.parametrize("t_path", (True, False))
@pytest.mark.parametrize("fold", (True, False))
def test_cases(arch: str, fold: bool, t_path: bool, circuit_file: Path) -> None:
    args = make_namespace(file=str(circuit_file), arch=arch, fold=fold, t_path=t_path, nosave=True)
    if fold and arch == "dsnm":
        with pytest.raises(ValueError, match="Can't fold"):
            _ = main(args)
    else:
        exit_code = main(args)
        assert exit_code == 0
