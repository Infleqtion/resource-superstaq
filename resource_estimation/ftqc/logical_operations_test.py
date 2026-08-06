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

from types import SimpleNamespace

import pytest
import stim

import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.logical_operations import (
    CSSLogicalOperations,
    validate_k1_css_patch,
)


@pytest.fixture(scope="module")
def steane_patch() -> lsp.CodePatch:
    return lsp.CodePatch("steane")


def test_profile_verifies_supplied_steane_cliffords(steane_patch: lsp.CodePatch) -> None:
    h_circuit = stim.Circuit("H 0 1 2 3 4 5 6")
    s_circuit = stim.Circuit("S_DAG 0 1 2 3 4 5 6")

    profile = CSSLogicalOperations.from_circuits(
        steane_patch, h_circuit=h_circuit, s_circuit=s_circuit
    )

    assert profile.circuit_for("H") == h_circuit
    assert profile.circuit_for("S") == s_circuit


def test_profile_rejects_incorrect_logical_action(steane_patch: lsp.CodePatch) -> None:
    with pytest.raises(ValueError, match="does not implement logical H exactly"):
        CSSLogicalOperations.from_circuits(steane_patch, h_circuit=stim.Circuit("I 0 1 2 3 4 5 6"))


def test_explicit_discovery_records_unavailable_search(
    steane_patch: lsp.CodePatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qldpc

    def unavailable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("GAP is unavailable")

    monkeypatch.setattr(qldpc.circuits, "get_transversal_circuits", unavailable)

    profile = CSSLogicalOperations.discover_from_qldpc(steane_patch)

    assert profile.circuit_for("H") is None
    assert profile.circuit_for("S") is None
    assert "qLDPC transversal search unavailable" in profile.missing_reason_for("H")
    assert "GAP is unavailable" in profile.missing_reason_for("S")


def test_explicit_discovery_records_missing_transversal_gates(
    steane_patch: lsp.CodePatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qldpc

    def no_transversal_gates(*args: object, **kwargs: object) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr(qldpc.circuits, "get_transversal_circuits", no_transversal_gates)

    profile = CSSLogicalOperations.discover_from_qldpc(steane_patch)

    assert profile.circuit_for("H") is None
    assert profile.circuit_for("S") is None
    assert profile.missing_reason_for("H") == "qLDPC found no transversal H implementation"
    assert profile.missing_reason_for("S") == "qLDPC found no transversal S implementation"


def test_profile_reports_qldpc_verification_failure(
    steane_patch: lsp.CodePatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qldpc

    def verification_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("invalid deformation")

    monkeypatch.setattr(qldpc.circuits, "get_logical_tableau", verification_failure)

    with pytest.raises(ValueError, match="does not preserve the supplied CodePatch"):
        CSSLogicalOperations.from_circuits(steane_patch, h_circuit=stim.Circuit("H 0"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"is_binary": False}, "binary CSS CodePatch"),
        ({"is_stabilizer_code": False}, "Subsystem CSS codes"),
        ({"d": None}, "known positive integer code distance"),
    ],
)
def test_validate_k1_css_patch_rejects_invalid_metadata(
    overrides: dict[str, object], message: str
) -> None:
    metadata: dict[str, object] = {
        "patch_label": "compute",
        "is_css": True,
        "is_binary": True,
        "is_stabilizer_code": True,
        "k": 1,
        "d": 3,
    }
    metadata.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_k1_css_patch(SimpleNamespace(**metadata))  # type: ignore[arg-type]


def test_validate_k1_css_patch_rejects_other_code_shapes() -> None:
    with pytest.raises(ValueError, match="require k=1"):
        validate_k1_css_patch(lsp.CodePatch("toric", d=2))

    from qldpc import codes

    with pytest.raises(ValueError, match="require a CSS"):
        validate_k1_css_patch(lsp.CodePatch(codes.FiveQubitCode))

    with pytest.raises(ValueError, match="require a compute CodePatch"):
        validate_k1_css_patch(lsp.CodePatch("steane", patch_label="memory"))
