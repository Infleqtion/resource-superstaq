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
import subprocess
import sys
from pathlib import Path

import cirq
import pytest


def test_clifford_t() -> None:
    result = subprocess.run(["python", "scripts/clifford_t.py"])
    assert result


def test_scaling() -> None:
    result = subprocess.run(["python", "scripts/scaling.py", "10", "20"])
    assert result


def test_rz_games() -> None:
    result = subprocess.run(["python", "scripts/rz_games.py", ".122441", "12", "0"])
    assert result


@pytest.mark.parametrize("architecture", ["ssm", "mzo", "dsm"])
def test_analyze_movement_layout(architecture: str, tmp_path: Path) -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.measure(a))
    circuit_path = tmp_path / "circuit.json"
    cirq.to_json(circuit, circuit_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze.py",
            str(circuit_path),
            "--arch",
            architecture,
            "--code-distance",
            "7",
            "--cultivation-repetition",
            "1",
            "--error-per-rz",
            "0.001",
            "--error-per-cult",
            "0.000001",
            "--facts",
            "0",
            "--nosave",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
