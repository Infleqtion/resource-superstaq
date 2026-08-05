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

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import stim

import resource_estimation.ftqc.lattice_surgery_primitives as lsp

LogicalClifford = Literal["H", "S"]


def validate_k1_css_patch(patch: lsp.CodePatch) -> None:
    """Validate the codes supported by generic movement-architecture costing."""
    if patch.patch_label != "compute":
        raise ValueError(
            f"Movement architectures require a compute CodePatch, not {patch.patch_label!r}."
        )
    if not patch.is_css:
        raise ValueError("Movement architectures require a CSS CodePatch.")
    if not patch.is_binary:
        raise ValueError("Movement architectures currently require a binary CSS CodePatch.")
    if not patch.is_stabilizer_code:
        raise ValueError("Subsystem CSS codes are not currently supported.")
    if patch.k != 1:
        raise ValueError(f"Movement architectures currently require k=1, not k={patch.k}.")
    if (
        patch.d is None
        or not isinstance(patch.d, (int, float))
        or not isfinite(patch.d)
        or patch.d <= 0
        or int(patch.d) != patch.d
    ):
        raise ValueError("Movement architectures require a known positive integer code distance.")


@dataclass(frozen=True)
class CSSLogicalOperations:
    """Verified physical implementations of logical Cliffords for one CSS patch.

    Constructing a profile from circuits verifies their logical action with qLDPC.  Transversal
    discovery is deliberately an explicit factory call because qLDPC's search can require external
    algebra systems and can have exponential runtime.
    """

    patch: lsp.CodePatch
    h_circuit: stim.Circuit | None = None
    s_circuit: stim.Circuit | None = None
    h_missing_reason: str = "no H circuit was supplied"
    s_missing_reason: str = "no S circuit was supplied"

    def __post_init__(self) -> None:
        validate_k1_css_patch(self.patch)
        if self.h_circuit is not None:
            self._verify_circuit(self.patch, "H", self.h_circuit)
            object.__setattr__(self, "h_missing_reason", "")
        if self.s_circuit is not None:
            self._verify_circuit(self.patch, "S", self.s_circuit)
            object.__setattr__(self, "s_missing_reason", "")

    @classmethod
    def from_circuits(
        cls,
        patch: lsp.CodePatch,
        *,
        h_circuit: stim.Circuit | None = None,
        s_circuit: stim.Circuit | None = None,
    ) -> CSSLogicalOperations:
        """Build a profile after verifying any supplied physical circuits with qLDPC."""
        return cls(patch=patch, h_circuit=h_circuit, s_circuit=s_circuit)

    @classmethod
    def discover_from_qldpc(
        cls,
        patch: lsp.CodePatch,
        *,
        local_gates: tuple[str, ...] = ("H", "S", "SWAP"),
        with_magma: bool = False,
    ) -> CSSLogicalOperations:
        """Explicitly ask qLDPC to find physical H and S implementations.

        Search failures are recorded in the returned profile rather than raised.  They become lazy
        undercount warnings if the corresponding logical operation is actually costed.
        """
        validate_k1_css_patch(patch)
        from qldpc import circuits

        try:
            found = circuits.get_transversal_circuits(
                patch.qldpc_code,
                [stim.Circuit("H 0"), stim.Circuit("S 0")],
                local_gates=local_gates,
                deform_code=False,
                with_magma=with_magma,
            )
        except Exception as ex:
            reason = f"qLDPC transversal search unavailable ({type(ex).__name__}: {ex})"
            return cls(
                patch=patch,
                h_missing_reason=reason,
                s_missing_reason=reason,
            )

        h_circuit, s_circuit = found
        return cls(
            patch=patch,
            h_circuit=h_circuit,
            s_circuit=s_circuit,
            h_missing_reason=(
                "qLDPC found no transversal H implementation" if h_circuit is None else ""
            ),
            s_missing_reason=(
                "qLDPC found no transversal S implementation" if s_circuit is None else ""
            ),
        )

    def circuit_for(self, gate: LogicalClifford) -> stim.Circuit | None:
        return self.h_circuit if gate == "H" else self.s_circuit

    def missing_reason_for(self, gate: LogicalClifford) -> str:
        return self.h_missing_reason if gate == "H" else self.s_missing_reason

    @staticmethod
    def _verify_circuit(
        patch: lsp.CodePatch, gate: LogicalClifford, physical_circuit: stim.Circuit
    ) -> None:
        from qldpc import circuits

        try:
            actual = circuits.get_logical_tableau(
                patch.qldpc_code, physical_circuit, deform_code=False
            )
        except Exception as ex:
            raise ValueError(
                f"Physical {gate} circuit does not preserve the supplied CodePatch."
            ) from ex
        expected = stim.Circuit(f"{gate} 0").to_tableau()
        if actual != expected:
            raise ValueError(f"Physical {gate} circuit does not implement logical {gate} exactly.")
