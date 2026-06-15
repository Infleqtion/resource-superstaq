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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cirq

ReactionBasis = Literal["X", "Z"]
ReactionDepth = dict[ReactionBasis, int]
ReactionDepthState = list[ReactionDepth]
ReactionDynamic = Callable[[ReactionDepthState], ReactionDepthState]
FactoryType = str


def factory_type_for_gate(gate: cirq.Gate | None) -> FactoryType:
    """Return the layout `ftype` string corresponding to a logical gate.

    Layout graph nodes already use lowercase `ftype` strings such as `"t"` and
    `"s"`. The reaction-depth collector uses this helper so factory membership
    checks and factory-spec lookup use the same representation.
    """
    if gate is None:
        return ""
    return str(gate).lower()


@dataclass(frozen=True)
class CorrectionPolicy:
    """Correction handling metadata for reaction-depth accounting.

    Attributes:
        name: Stable name for this correction policy.
        reaction_dynamic: Callable that transforms the participating qubits'
            old reaction depths into updated reaction depths. The returned list
            must align positionally with the input list.
    """

    name: str
    """Stable name for this correction policy."""

    reaction_dynamic: ReactionDynamic
    """Reaction-depth update rule for all qubits acted on by this correction policy."""


@dataclass(frozen=True)
class FactorySpec:
    """Reusable metadata for one kind of factory placed by a layout.

    Attributes:
        name: Stable name for this factory spec.
        ftype: Factory type string used by layout graph nodes.
        produced_gate: Logical resource or gate supplied by this factory.
        correction_policy: Correction handling for the produced resource.
    """

    name: str
    """Stable name for this factory spec."""

    ftype: FactoryType
    """Factory type string used by layout graph nodes."""

    produced_gate: cirq.Gate
    """Logical resource or gate supplied by this factory."""

    correction_policy: CorrectionPolicy
    """Correction handling for the produced resource."""


def _t_auto_corrected_reaction_dynamic(old_depths: ReactionDepthState) -> ReactionDepthState:
    """Return reaction-depth updates for auto-corrected T factories.

    This assumes the auto-corrected circuit from Game of Surface Codes
    (http://arxiv.org/abs/1808.02892).

    Args:
        old_depths: Single-qubit reaction-depth state before the T correction.

    Returns:
        Single-qubit update applying `newZ = max(oldZ, oldX + 1)`.
    """
    old_depth = old_depths[0]
    return [{"Z": max(old_depth.get("X", 0) + 1, old_depth.get("Z", 0))}]


def _t_non_auto_corrected_reaction_dynamic(old_depths: ReactionDepthState) -> ReactionDepthState:
    """Return reaction-depth updates for non-auto-corrected T factories.

    Args:
        old_depths: Single-qubit reaction-depth state before the T correction.

    Returns:
        Single-qubit update applying `newX = oldX + 1` and `newZ = oldZ + 1`.
    """
    old_depth = old_depths[0]
    return [{"X": old_depth.get("X", 0) + 1, "Z": old_depth.get("Z", 0) + 1}]


def _s_reaction_dynamic(old_depths: ReactionDepthState) -> ReactionDepthState:
    """Return reaction-depth updates for standard S factories.

    Args:
        old_depths: Single-qubit reaction-depth state before the S correction.

    Returns:
        Single-qubit update applying `newX = oldX + 1` and `newZ = oldZ + 1`.
    """
    old_depth = old_depths[0]
    return [{"X": old_depth.get("X", 0) + 1, "Z": old_depth.get("Z", 0) + 1}]


def _ccz_auto_corrected_reaction_dynamic(_old_depths: ReactionDepthState) -> ReactionDepthState:
    """Return reaction-depth updates for auto-corrected CCZ factories.

    This assumes the auto-corrected circuit from How to Eat Magic States
    (https://docs.google.com/presentation/d/1b0r3pKWi3_Bu64Rc5Ojc_9eVjWyZPWRP3-UBnqNdJB0).

    Args:
        _old_depths: Three-qubit reaction-depth state ordered as control1,
            control2, target.

    Returns:
        Three positional qubit updates using the auto-corrected CCZ dynamics.
    """
    control1_old, control2_old, target_old = _old_depths
    control1_new: ReactionDepth = {
        "Z": max(control1_old.get("Z", 0), control2_old.get("X", 0) + 1, target_old.get("Z", 0) + 1)
    }
    control2_new: ReactionDepth = {
        "Z": max(control2_old.get("Z", 0), control1_old.get("X", 0) + 1, target_old.get("Z", 0) + 1)
    }
    target_new: ReactionDepth = {
        "X": max(
            target_old.get("X", 0), control1_old.get("X", 0) + 1, control2_old.get("X", 0) + 1
        ),
    }

    return [control1_new, control2_new, target_new]


def _ccz_non_auto_corrected_reaction_dynamic(_old_depths: ReactionDepthState) -> ReactionDepthState:
    """Return reaction-depth updates for non-auto-corrected CCZ factories.

    Args:
        _old_depths: Three-qubit reaction-depth state ordered as control1,
            control2, target.

    Returns:
        Three positional qubit updates applying `newX = oldX + 1` and
        `newZ = oldZ + 1` to each participating qubit.
    """
    return [{"X": depths.get("X", 0) + 1, "Z": depths.get("Z", 0) + 1} for depths in _old_depths]


T_AUTO_CORRECTED_CORRECTION_POLICY = CorrectionPolicy(
    name="t-auto-corrected",
    reaction_dynamic=_t_auto_corrected_reaction_dynamic,
)
T_NON_AUTO_CORRECTED_CORRECTION_POLICY = CorrectionPolicy(
    name="t-non-auto-corrected",
    reaction_dynamic=_t_non_auto_corrected_reaction_dynamic,
)
S_CORRECTION_POLICY = CorrectionPolicy(
    name="s",
    reaction_dynamic=_s_reaction_dynamic,
)
CCZ_AUTO_CORRECTED_CORRECTION_POLICY = CorrectionPolicy(
    name="ccz-auto-corrected",
    reaction_dynamic=_ccz_auto_corrected_reaction_dynamic,
)
CCZ_NON_AUTO_CORRECTED_CORRECTION_POLICY = CorrectionPolicy(
    name="ccz-non-auto-corrected",
    reaction_dynamic=_ccz_non_auto_corrected_reaction_dynamic,
)

T_AUTO_CORRECTED_FACTORY_SPEC = FactorySpec(
    name="t-auto-corrected",
    ftype="t",
    produced_gate=cirq.T,
    correction_policy=T_AUTO_CORRECTED_CORRECTION_POLICY,
)
T_NON_AUTO_CORRECTED_FACTORY_SPEC = FactorySpec(
    name="t-non-auto-corrected",
    ftype="t",
    produced_gate=cirq.T,
    correction_policy=T_NON_AUTO_CORRECTED_CORRECTION_POLICY,
)
S_FACTORY_SPEC = FactorySpec(
    name="s",
    ftype="s",
    produced_gate=cirq.S,
    correction_policy=S_CORRECTION_POLICY,
)
CCZ_AUTO_CORRECTED_FACTORY_SPEC = FactorySpec(
    name="ccz-auto-corrected",
    ftype="ccz",
    produced_gate=cirq.CCZ,
    correction_policy=CCZ_AUTO_CORRECTED_CORRECTION_POLICY,
)
CCZ_NON_AUTO_CORRECTED_FACTORY_SPEC = FactorySpec(
    name="ccz-non-auto-corrected",
    ftype="ccz",
    produced_gate=cirq.CCZ,
    correction_policy=CCZ_NON_AUTO_CORRECTED_CORRECTION_POLICY,
)


def default_factory_specs(
    num_t_factories: int,
    num_s_factories: int,
) -> dict[FactoryType, FactorySpec]:
    """Return default factory specs for factory types present in a layout.

    Args:
        num_t_factories: Number of T factory patches requested or generated by
            the layout.
        num_s_factories: Number of S factory patches requested or generated by
            the layout.

    Returns:
        A new dictionary keyed by factory `ftype`. The dictionary contains the
        auto-corrected T spec when `num_t_factories` is positive and the
        standard S spec when `num_s_factories` is positive.
    """
    specs: dict[FactoryType, FactorySpec] = {}
    if num_t_factories > 0:
        specs["t"] = T_AUTO_CORRECTED_FACTORY_SPEC
    if num_s_factories > 0:
        specs["s"] = S_FACTORY_SPEC
    return specs


__all__ = [
    "CCZ_AUTO_CORRECTED_CORRECTION_POLICY",
    "CCZ_AUTO_CORRECTED_FACTORY_SPEC",
    "CCZ_NON_AUTO_CORRECTED_CORRECTION_POLICY",
    "CCZ_NON_AUTO_CORRECTED_FACTORY_SPEC",
    "CorrectionPolicy",
    "FactorySpec",
    "FactoryType",
    "ReactionBasis",
    "ReactionDepth",
    "ReactionDepthState",
    "ReactionDynamic",
    "S_CORRECTION_POLICY",
    "S_FACTORY_SPEC",
    "T_AUTO_CORRECTED_CORRECTION_POLICY",
    "T_AUTO_CORRECTED_FACTORY_SPEC",
    "T_NON_AUTO_CORRECTED_CORRECTION_POLICY",
    "T_NON_AUTO_CORRECTED_FACTORY_SPEC",
    "default_factory_specs",
    "factory_type_for_gate",
]
