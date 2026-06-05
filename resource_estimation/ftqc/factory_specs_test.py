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
import cirq
import pytest

import resource_estimation.ftqc.factory_specs as factory_specs


def test_factory_spec_stores_correction_metadata():
    def reaction_dynamic(
        old_depths: factory_specs.ReactionDepthState,
    ) -> factory_specs.ReactionDepthState:
        return [dict(old_depth) for old_depth in old_depths]

    correction_policy = factory_specs.CorrectionPolicy(
        name="test-correction",
        reaction_dynamic=reaction_dynamic,
    )
    factory_spec = factory_specs.FactorySpec(
        name="test-t",
        ftype="t",
        produced_gate=cirq.T,
        correction_policy=correction_policy,
    )

    assert factory_spec.ftype == "t"
    assert factory_spec.produced_gate == cirq.T
    assert factory_spec.correction_policy is correction_policy
    assert factory_spec.correction_policy.reaction_dynamic is reaction_dynamic


def test_factory_type_for_gate_matches_layout_ftype_strings():
    assert factory_specs.factory_type_for_gate(cirq.T) == "t"
    assert factory_specs.factory_type_for_gate(cirq.S) == "s"
    assert factory_specs.factory_type_for_gate(cirq.CCZ) == "ccz"
    assert factory_specs.factory_type_for_gate(None) == ""


@pytest.mark.parametrize(
    ("auto_corrected_spec", "non_auto_corrected_spec", "ftype", "produced_gate"),
    [
        (
            factory_specs.T_AUTO_CORRECTED_FACTORY_SPEC,
            factory_specs.T_NON_AUTO_CORRECTED_FACTORY_SPEC,
            "t",
            cirq.T,
        ),
        (
            factory_specs.S_AUTO_CORRECTED_FACTORY_SPEC,
            factory_specs.S_NON_AUTO_CORRECTED_FACTORY_SPEC,
            "s",
            cirq.S,
        ),
        (
            factory_specs.CCZ_AUTO_CORRECTED_FACTORY_SPEC,
            factory_specs.CCZ_NON_AUTO_CORRECTED_FACTORY_SPEC,
            "ccz",
            cirq.CCZ,
        ),
    ],
)
def test_standard_factory_specs(auto_corrected_spec, non_auto_corrected_spec, ftype, produced_gate):
    assert auto_corrected_spec is not non_auto_corrected_spec
    assert auto_corrected_spec.correction_policy is not non_auto_corrected_spec.correction_policy

    assert auto_corrected_spec.ftype == ftype
    assert non_auto_corrected_spec.ftype == ftype
    assert auto_corrected_spec.produced_gate == produced_gate
    assert non_auto_corrected_spec.produced_gate == produced_gate

    assert auto_corrected_spec.name == f"{ftype}-auto-corrected"
    assert non_auto_corrected_spec.name == f"{ftype}-non-auto-corrected"
    assert auto_corrected_spec.correction_policy.name == f"{ftype}-auto-corrected"
    assert non_auto_corrected_spec.correction_policy.name == f"{ftype}-non-auto-corrected"


@pytest.mark.parametrize(
    "correction_policy",
    [
        factory_specs.T_NON_AUTO_CORRECTED_CORRECTION_POLICY,
        factory_specs.S_AUTO_CORRECTED_CORRECTION_POLICY,
        factory_specs.S_NON_AUTO_CORRECTED_CORRECTION_POLICY,
        factory_specs.CCZ_NON_AUTO_CORRECTED_CORRECTION_POLICY,
    ],
)
def test_unimplemented_standard_reaction_dynamics_are_skeletons(correction_policy):
    with pytest.raises(NotImplementedError, match=correction_policy.name):
        correction_policy.reaction_dynamic([{"X": 0, "Z": 0}])


@pytest.mark.parametrize(
    ("old_depth", "expected_depth"),
    [
        pytest.param({"X": 2, "Z": 1}, {"Z": 3}, id="old_x_plus_one_wins"),
        pytest.param({"X": 2, "Z": 5}, {"Z": 5}, id="old_z_wins"),
        pytest.param({"X": 2, "Z": 3}, {"Z": 3}, id="old_x_plus_one_ties_old_z"),
    ],
)
def test_t_auto_corrected_reaction_dynamic_updates_single_qubit(old_depth, expected_depth):
    assert factory_specs.T_AUTO_CORRECTED_CORRECTION_POLICY.reaction_dynamic([old_depth]) == [
        expected_depth
    ]


@pytest.mark.parametrize(
    ("old_depths", "expected_depths"),
    [
        pytest.param(
            [
                {"X": 0, "Z": 10},
                {"X": 0, "Z": 11},
                {"X": 12, "Z": 0},
            ],
            [
                {"Z": 10},
                {"Z": 11},
                {"X": 12},
            ],
            id="z1_z2_x3_existing_depths_win",
        ),
        pytest.param(
            [
                {"X": 10, "Z": 0},
                {"X": 5, "Z": 0},
                {"X": 0, "Z": 0},
            ],
            [
                {"Z": 6},
                {"Z": 11},
                {"X": 11},
            ],
            id="z1_from_x2_z2_from_x1_x3_from_x1",
        ),
        pytest.param(
            [
                {"X": 0, "Z": 0},
                {"X": 10, "Z": 0},
                {"X": 0, "Z": 20},
            ],
            [
                {"Z": 21},
                {"Z": 21},
                {"X": 11},
            ],
            id="z1_z2_from_z3_x3_from_x2",
        ),
    ],
)
def test_ccz_auto_corrected_reaction_dynamic_updates_controls_and_target(
    old_depths, expected_depths
):
    assert (
        factory_specs.CCZ_AUTO_CORRECTED_CORRECTION_POLICY.reaction_dynamic(old_depths)
        == expected_depths
    )
