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
from math import pi

import cirq
import pytest
from numpy.testing import assert_array_equal

import resource_estimation.ftqc.lattice_surgery_primitives as lsp


def test_merge() -> None:
    merge_gate = lsp.Merge(2, smooth=True)
    assert merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"

    merge_gate = lsp.Merge(2, smooth=False)
    assert not merge_gate.smooth
    assert merge_gate.num_qubits() == 2
    assert str(merge_gate) == "MERGE"


def test_split() -> None:
    partitions = [1, 2, 3, 4]
    split_gate = lsp.Split(partitions=partitions, smooth=True)
    assert split_gate.smooth
    assert split_gate.num_qubits() == 10
    assert str(split_gate) == "SPLIT"
    assert split_gate.partitions == partitions

    split_gate = lsp.Split(partitions=partitions, smooth=False)
    assert not split_gate.smooth
    assert split_gate.num_qubits() == 10
    assert str(split_gate) == "SPLIT"
    assert split_gate.partitions == partitions


def test_syndrome_extract() -> None:
    for i in [1, 2, 3, 4]:
        extraction_gate = lsp.SyndromeExtract(i, i * 2)
        assert extraction_gate.num_qubits() == i
        assert extraction_gate.rounds == i * 2
        assert str(extraction_gate) == f"SE({i * 2})"


def test_error_correct() -> None:
    error_correction_gate = lsp.ErrorCorrect(1)
    assert str(error_correction_gate) == "ERROR CORRECT"


def test_cultivate() -> None:
    theta = pi / 2
    cultivation_gate = lsp.Cultivate(theta=theta)
    assert cultivation_gate.theta == theta
    assert str(cultivation_gate) == "CULT(1.571)"


def test_move() -> None:
    a, b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    alley_move = lsp.Move(None).on(a, b)
    assert str(alley_move) == "MOVE(q(0, 0), q(0, 1))"
    interact_move = lsp.Move("interact").on(a)
    assert str(interact_move) == "MOVE_IZ(q(0, 0))"
    measure_move = lsp.Move("measure").on(b)
    assert str(measure_move) == "MOVE_MZ(q(0, 1))"


def test_code_patch_surface_metadata() -> None:
    patch = lsp.CodePatch()

    assert patch.code_type == "surface"
    assert patch.code_params == (49, 1, 7)
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.patch_label == "compute"
    assert not patch.is_qldpc_backed
    assert (
        repr(patch)
        == "lsp.CodePatch(code_type='surface', d=7, n=49, k=1, patch_label='compute')"
    )

    patch = lsp.CodePatch("surface", d=5, patch_label="cultivate")

    assert patch.code_params == (25, 1, 5)
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.patch_label == "cultivate"
    assert not patch.is_qldpc_backed
    assert (
        repr(patch)
        == "lsp.CodePatch(code_type='surface', d=5, n=25, k=1, patch_label='cultivate')"
    )


def test_code_patch_explicit_metadata() -> None:
    patch = lsp.CodePatch(
        "custom",
        d=3,
        n=15,
        k=1,
        num_measure_qubits=10,
        patch_label="memory",
    )

    assert patch.code_params == (15, 1, 3)
    assert patch.n == 15
    assert patch.patch_label == "memory"

    with pytest.raises(ValueError, match="requires a qLDPC code"):
        lsp.CodePatch("color", d=3)

    with pytest.raises(ValueError, match="Patch label must be one of"):
        lsp.CodePatch(patch_label="bad")  # type: ignore[arg-type]


def test_code_patch_from_qldpc_code() -> None:
    qldpc = pytest.importorskip("qldpc")
    from qldpc import codes

    patch = lsp.CodePatch.from_qldpc_code(codes.FiveQubitCode(), patch_label="distil")

    assert qldpc.__version__
    assert patch.code_type == "FiveQubitCode"
    assert patch.code_params == (5, 1, 3)
    assert patch.num_data_qubits == 5
    assert patch.num_measure_qubits == 4
    assert patch.is_qldpc_backed
    assert patch.patch_label == "distil"
    assert len(patch.logical_ops("X")) == 1
    assert len(patch.transversal_ops()) == 2


def test_code_patch_from_qldpc_family() -> None:
    pytest.importorskip("qldpc")

    patch = lsp.CodePatch.from_qldpc_family("toric", d=2)

    assert patch.code_type == "toric"
    assert patch.qldpc_family == "ToricCode"
    assert patch.qldpc_args == (2,)
    assert patch.qldpc_kwargs == {}
    assert patch.code_params == (4, 2, 2)
    assert patch.patch_label == "compute"


def test_code_patch_from_qldpc_factory() -> None:
    pytest.importorskip("qldpc")
    from qldpc import codes

    patch = lsp.CodePatch.from_qldpc_factory(
        "custom_five", codes.FiveQubitCode, patch_label="memory"
    )

    assert patch.code_type == "custom_five"
    assert patch.qldpc_family == "FiveQubitCode"
    assert patch.qldpc_args == ()
    assert patch.code_params == (5, 1, 3)
    assert patch.patch_label == "memory"


def test_farm_empty() -> None:
    farm = lsp.Farm()

    assert farm.code_patches == []
    assert farm.num_physical_qubits == 0
    assert farm.num_logical_qubits == 0
    assert farm.num_code_patches == 0


def test_farm_sums_cultivation_patches() -> None:
    patch_a = lsp.CodePatch(patch_label="cultivate")
    patch_b = lsp.CodePatch("custom", d=3, n=15, k=2, patch_label="cultivate")

    farm = lsp.Farm([patch_a])
    farm.add_patch(patch_b)

    assert farm.code_patches == [patch_a, patch_b]
    assert farm.num_physical_qubits == 64
    assert farm.num_logical_qubits == 3
    assert farm.num_code_patches == 2


def test_farm_initializes_default_cultivation_patches() -> None:
    farm = lsp.Farm(3)

    assert farm.num_physical_qubits == 147
    assert farm.num_logical_qubits == 3
    assert farm.num_code_patches == 3
    assert [patch.code_type for patch in farm.code_patches] == ["surface"] * 3
    assert [patch.patch_label for patch in farm.code_patches] == ["cultivate"] * 3
    assert [patch.code_params for patch in farm.code_patches] == [(49, 1, 7)] * 3


def test_farm_rejects_non_cultivation_patches() -> None:
    patch = lsp.CodePatch()

    with pytest.raises(ValueError, match="patch_label='cultivate'"):
        lsp.Farm([patch])

    farm = lsp.Farm()
    with pytest.raises(ValueError, match="patch_label='cultivate'"):
        farm.add_patch(patch)


def test_farm_rejects_negative_patch_count() -> None:
    with pytest.raises(ValueError, match="patch count must be nonnegative"):
        lsp.Farm(-1)


@pytest.mark.parametrize(("label", "num_code_patches"), [("CCZ", 9), ("T", 11)])
def test_distillery_initializes_default_distil_patches(
    label: lsp.DistilleryLabel, num_code_patches: int
) -> None:
    distillery = lsp.Distillery(label)

    assert distillery.label == label
    assert distillery.num_physical_qubits == 49 * num_code_patches
    assert distillery.num_logical_qubits == num_code_patches
    assert distillery.num_code_patches == num_code_patches
    assert [patch.code_type for patch in distillery.code_patches] == [
        "surface"
    ] * num_code_patches
    assert [patch.patch_label for patch in distillery.code_patches] == [
        "distil"
    ] * num_code_patches
    assert [patch.code_params for patch in distillery.code_patches] == [
        (49, 1, 7)
    ] * num_code_patches


def test_distillery_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="Distillery label must be one of"):
        lsp.Distillery("bad")  # type: ignore[arg-type]


def test_factory_empty() -> None:
    factory = lsp.Factory()

    assert factory.ccz_distilleries == []
    assert factory.t_distilleries == []
    assert factory.num_ccz_distilleries == 0
    assert factory.num_t_distilleries == 0
    assert factory.num_distilleries == 0
    assert factory.num_physical_qubits == 0
    assert factory.num_logical_qubits == 0


def test_factory_initializes_distilleries_from_counts() -> None:
    factory = lsp.Factory(2, 3)

    assert [distillery.label for distillery in factory.ccz_distilleries] == ["CCZ"] * 2
    assert [distillery.label for distillery in factory.t_distilleries] == ["T"] * 3
    assert factory.num_ccz_distilleries == 2
    assert factory.num_t_distilleries == 3
    assert factory.num_distilleries == 5
    assert factory.num_physical_qubits == 49 * (2 * 9 + 3 * 11)
    assert factory.num_logical_qubits == 2 * 9 + 3 * 11


def test_factory_initializes_distilleries_from_iterables() -> None:
    ccz_distillery = lsp.Distillery("CCZ")
    t_distilleries = [lsp.Distillery("T"), lsp.Distillery("T")]

    factory = lsp.Factory([ccz_distillery], t_distilleries)

    assert factory.ccz_distilleries == [ccz_distillery]
    assert factory.t_distilleries == t_distilleries
    assert factory.num_ccz_distilleries == 1
    assert factory.num_t_distilleries == 2
    assert factory.num_distilleries == 3
    assert factory.num_physical_qubits == 49 * (9 + 2 * 11)
    assert factory.num_logical_qubits == 9 + 2 * 11


def test_factory_initializes_distilleries_from_mixed_inputs() -> None:
    t_distillery = lsp.Distillery("T")

    factory = lsp.Factory(1, [t_distillery])

    assert factory.num_ccz_distilleries == 1
    assert factory.t_distilleries == [t_distillery]
    assert factory.num_distilleries == 2
    assert factory.num_physical_qubits == 49 * (9 + 11)
    assert factory.num_logical_qubits == 9 + 11


def test_factory_adds_distilleries() -> None:
    ccz_distillery = lsp.Distillery("CCZ")
    t_distillery = lsp.Distillery("T")
    factory = lsp.Factory()

    factory.add_ccz_distillery(ccz_distillery)
    factory.add_t_distillery(t_distillery)

    assert factory.ccz_distilleries == [ccz_distillery]
    assert factory.t_distilleries == [t_distillery]
    assert factory.num_distilleries == 2
    assert factory.num_physical_qubits == 49 * (9 + 11)
    assert factory.num_logical_qubits == 9 + 11


def test_factory_rejects_negative_distillery_counts() -> None:
    with pytest.raises(ValueError, match="distillery count must be nonnegative"):
        lsp.Factory(-1)
    with pytest.raises(ValueError, match="distillery count must be nonnegative"):
        lsp.Factory(0, -1)


def test_factory_rejects_non_distillery_objects() -> None:
    with pytest.raises(TypeError, match="Factory can only contain Distillery objects"):
        lsp.Factory([object()])  # type: ignore[list-item]

    factory = lsp.Factory()
    with pytest.raises(TypeError, match="Factory can only contain Distillery objects"):
        factory.add_ccz_distillery(object())  # type: ignore[arg-type]


def test_factory_rejects_label_mismatches() -> None:
    with pytest.raises(ValueError, match="Expected a CCZ Distillery"):
        lsp.Factory([lsp.Distillery("T")])
    with pytest.raises(ValueError, match="Expected a T Distillery"):
        lsp.Factory(0, [lsp.Distillery("CCZ")])

    factory = lsp.Factory()
    with pytest.raises(ValueError, match="Expected a CCZ Distillery"):
        factory.add_ccz_distillery(lsp.Distillery("T"))
    with pytest.raises(ValueError, match="Expected a T Distillery"):
        factory.add_t_distillery(lsp.Distillery("CCZ"))


@pytest.mark.parametrize("label", ["T:cultivated", "T:distilled", "CCZ:distilled"])
def test_vault_empty(label: lsp.VaultLabel) -> None:
    vault = lsp.Vault(label)

    assert vault.label == label
    assert vault.code_patches == []
    assert vault.num_physical_qubits == 0
    assert vault.num_logical_qubits == 0
    assert vault.num_code_patches == 0


def test_vault_initializes_default_shyps_memory_patches() -> None:
    pytest.importorskip("qldpc")

    with pytest.warns(RuntimeWarning, match="Computing qLDPC code distance"):
        vault = lsp.Vault("T:distilled", 2)

    assert vault.label == "T:distilled"
    assert vault.num_physical_qubits == 98
    assert vault.num_logical_qubits == 18
    assert vault.num_code_patches == 2
    assert [patch.code_type for patch in vault.code_patches] == ["shyps"] * 2
    assert [patch.qldpc_family for patch in vault.code_patches] == ["SHYPSCode"] * 2
    assert [patch.qldpc_args for patch in vault.code_patches] == [(3,)] * 2
    assert [patch.patch_label for patch in vault.code_patches] == ["memory"] * 2
    assert [patch.code_params for patch in vault.code_patches] == [(49, 9, 4)] * 2
    assert all(patch.is_qldpc_backed for patch in vault.code_patches)


def test_vault_sums_memory_patches() -> None:
    patch_a = lsp.CodePatch("custom", d=3, n=15, k=2, patch_label="memory")
    patch_b = lsp.CodePatch(patch_label="memory")

    vault = lsp.Vault("CCZ:distilled", [patch_a])
    vault.add_patch(patch_b)

    assert vault.code_patches == [patch_a, patch_b]
    assert vault.num_physical_qubits == 64
    assert vault.num_logical_qubits == 3
    assert vault.num_code_patches == 2


def test_vault_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="Vault label must be one of"):
        lsp.Vault("bad")  # type: ignore[arg-type]


def test_vault_rejects_non_memory_patches() -> None:
    patch = lsp.CodePatch()

    with pytest.raises(ValueError, match="patch_label='memory'"):
        lsp.Vault("T:cultivated", [patch])

    vault = lsp.Vault("T:cultivated")
    with pytest.raises(ValueError, match="patch_label='memory'"):
        vault.add_patch(patch)


def test_vault_rejects_non_code_patch_objects() -> None:
    with pytest.raises(TypeError, match="Vault can only contain CodePatch objects"):
        lsp.Vault("T:cultivated", [object()])  # type: ignore[list-item]

    vault = lsp.Vault("T:cultivated")
    with pytest.raises(TypeError, match="Vault can only contain CodePatch objects"):
        vault.add_patch(object())  # type: ignore[arg-type]


def test_vault_rejects_negative_patch_count() -> None:
    with pytest.raises(ValueError, match="patch count must be nonnegative"):
        lsp.Vault("T:cultivated", -1)


def test_bank_empty() -> None:
    bank = lsp.Bank()

    assert bank.t_cultivated_vault is None
    assert bank.t_distilled_vault is None
    assert bank.ccz_distilled_vault is None
    assert not bank.has_t_cultivated_vault
    assert not bank.has_t_distilled_vault
    assert not bank.has_ccz_distilled_vault
    assert bank.num_vaults == 0
    assert bank.num_code_patches == 0
    assert bank.num_physical_qubits == 0
    assert bank.num_logical_qubits == 0


def test_bank_initializes_default_vaults() -> None:
    pytest.importorskip("qldpc")

    with pytest.warns(RuntimeWarning, match="Computing qLDPC code distance"):
        bank = lsp.Bank(t_cultivated=True, t_distilled=True, ccz_distilled=True)

    assert bank.has_t_cultivated_vault
    assert bank.has_t_distilled_vault
    assert bank.has_ccz_distilled_vault
    assert bank.t_cultivated_vault is not None
    assert bank.t_distilled_vault is not None
    assert bank.ccz_distilled_vault is not None
    assert bank.t_cultivated_vault.label == "T:cultivated"
    assert bank.t_distilled_vault.label == "T:distilled"
    assert bank.ccz_distilled_vault.label == "CCZ:distilled"
    assert bank.num_vaults == 3
    assert bank.num_code_patches == 3
    assert bank.num_physical_qubits == 147
    assert bank.num_logical_qubits == 27


def test_bank_adds_vaults() -> None:
    patch_a = lsp.CodePatch("custom", d=3, n=15, k=2, patch_label="memory")
    patch_b = lsp.CodePatch("custom", d=5, n=25, k=3, patch_label="memory")
    patch_c = lsp.CodePatch("custom", d=7, n=35, k=4, patch_label="memory")
    t_cultivated_vault = lsp.Vault("T:cultivated", [patch_a])
    t_distilled_vault = lsp.Vault("T:distilled", [patch_b])
    ccz_distilled_vault = lsp.Vault("CCZ:distilled", [patch_c])
    bank = lsp.Bank()

    bank.add_t_cultivated_vault(t_cultivated_vault)
    bank.add_t_distilled_vault(t_distilled_vault)
    bank.add_ccz_distilled_vault(ccz_distilled_vault)

    assert bank.t_cultivated_vault == t_cultivated_vault
    assert bank.t_distilled_vault == t_distilled_vault
    assert bank.ccz_distilled_vault == ccz_distilled_vault
    assert bank.num_vaults == 3
    assert bank.num_code_patches == 3
    assert bank.num_physical_qubits == 75
    assert bank.num_logical_qubits == 9


def test_bank_rejects_non_vault_objects() -> None:
    bank = lsp.Bank()

    with pytest.raises(TypeError, match="Bank can only contain Vault objects"):
        bank.add_t_cultivated_vault(object())  # type: ignore[arg-type]


def test_bank_rejects_label_mismatches() -> None:
    patch = lsp.CodePatch("custom", d=3, n=15, k=2, patch_label="memory")
    bank = lsp.Bank()

    with pytest.raises(ValueError, match="Expected a T:cultivated Vault"):
        bank.add_t_cultivated_vault(lsp.Vault("T:distilled", [patch]))
    with pytest.raises(ValueError, match="Expected a T:distilled Vault"):
        bank.add_t_distilled_vault(lsp.Vault("CCZ:distilled", [patch]))
    with pytest.raises(ValueError, match="Expected a CCZ:distilled Vault"):
        bank.add_ccz_distilled_vault(lsp.Vault("T:cultivated", [patch]))


def test_bank_rejects_duplicate_vaults() -> None:
    patch_a = lsp.CodePatch("custom", d=3, n=15, k=2, patch_label="memory")
    patch_b = lsp.CodePatch("custom", d=5, n=25, k=3, patch_label="memory")
    bank = lsp.Bank()

    bank.add_t_cultivated_vault(lsp.Vault("T:cultivated", [patch_a]))

    with pytest.raises(ValueError, match="already has a T:cultivated Vault"):
        bank.add_t_cultivated_vault(lsp.Vault("T:cultivated", [patch_b]))


def test_rotated_code_patch() -> None:
    with pytest.raises(AssertionError, match="CodePatches must be odd distance"):
        lsp.RotatedCodePatch(4)

    d = 3
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 3
    assert patch.rows == patch.cols == 5
    assert patch.num_physical_qubits == 17
    assert patch.num_data_qubits == 9
    assert patch.num_measure_qubits == 8
    assert patch.num_z_stabs(full=True) == 2
    assert patch.num_z_stabs(full=False) == 2
    assert patch.num_x_stabs(full=True) == 2
    assert patch.num_x_stabs(full=False) == 2
    assert patch.total_z_syndrome_cnots() == 12
    assert patch.total_x_syndrome_cnots() == 12

    d = 5
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 5
    assert patch.rows == patch.cols == 9
    assert patch.num_physical_qubits == 49
    assert patch.num_data_qubits == 25
    assert patch.num_measure_qubits == 24
    assert patch.num_z_stabs(full=True) == 8
    assert patch.num_z_stabs(full=False) == 4
    assert patch.num_x_stabs(full=True) == 8
    assert patch.num_x_stabs(full=False) == 4
    assert patch.total_z_syndrome_cnots() == 40
    assert patch.total_x_syndrome_cnots() == 40

    d = 7
    patch = lsp.RotatedCodePatch(d)
    assert patch.d == 7
    assert patch.rows == patch.cols == 13
    assert patch.num_physical_qubits == 97
    assert patch.num_data_qubits == 49
    assert patch.num_measure_qubits == 48
    assert patch.num_z_stabs(full=True) == 18
    assert patch.num_z_stabs(full=False) == 6
    assert patch.num_x_stabs(full=True) == 18
    assert patch.num_x_stabs(full=False) == 6
    assert patch.total_z_syndrome_cnots() == 84
    assert patch.total_x_syndrome_cnots() == 84


def test_buffer() -> None:
    d = 7
    smooth_buff = lsp.BufferCodePatch(d=d, smooth=True)
    rough_buff = lsp.BufferCodePatch(d=d, smooth=False)

    assert_array_equal(
        [
            smooth_buff.num_z_stabs(full=True),
            smooth_buff.num_x_stabs(full=True),
            rough_buff.num_z_stabs(full=True),
            rough_buff.num_x_stabs(full=True),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_buff.num_x_stabs(full=False),
            rough_buff.num_z_stabs(full=False),
        ],
        2,
    )
    assert_array_equal(
        [
            smooth_buff.num_z_stabs(full=False),
            rough_buff.num_x_stabs(full=False),
        ],
        0,
    )


def test_intermediate_patch() -> None:
    d = 7
    smooth_inter = lsp.IntermediatePatch(d=d, smooth=True)
    rough_inter = lsp.IntermediatePatch(d=d, smooth=False)
    assert_array_equal(
        [
            smooth_inter.num_z_stabs(full=True),
            smooth_inter.num_x_stabs(full=True),
            rough_inter.num_z_stabs(full=True),
            rough_inter.num_x_stabs(full=True),
        ],
        18,
    )
    assert_array_equal(
        [
            smooth_inter.num_x_stabs(full=False),
            rough_inter.num_z_stabs(full=False),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_inter.num_z_stabs(full=False),
            rough_inter.num_x_stabs(full=False),
        ],
        0,
    )


def test_endpoint_patch() -> None:
    d = 7
    smooth_end = lsp.EndpointPatch(d=d, smooth=True)
    rough_end = lsp.EndpointPatch(d=d, smooth=False)
    assert_array_equal(
        [
            smooth_end.num_z_stabs(full=True),
            smooth_end.num_x_stabs(full=True),
            rough_end.num_z_stabs(full=True),
            rough_end.num_x_stabs(full=True),
        ],
        18,
    )
    assert_array_equal(
        [
            smooth_end.num_x_stabs(full=False),
            rough_end.num_z_stabs(full=False),
        ],
        6,
    )
    assert_array_equal(
        [
            smooth_end.num_z_stabs(full=False),
            rough_end.num_x_stabs(full=False),
        ],
        3,
    )


def test_serialization() -> None:
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    factory_block = cirq.LineQubit.range(31)
    circuit = cirq.Circuit(
        [
            lsp.Merge(2, True).on(qubit_a, qubit_b),
            lsp.Split([1, 1], True).on(qubit_a, qubit_b),
            lsp.SyndromeExtract(1, 1).on(qubit_a),
            lsp.ErrorCorrect(1).on(qubit_b),
            lsp.Cultivate(1.0).on(qubit_a),
            lsp.Move(zone="interact").on_each(qubit_a, qubit_b),
            lsp.Move(zone=None).on(qubit_a, qubit_b),
            lsp.Move(zone="measure").on(qubit_a),
            lsp.Distil("T").on(*factory_block),
            lsp.Distil("Toffoli").on(*factory_block[:23]),
        ]
    )
    json_str = cirq.to_json(circuit)
    new_circuit = cirq.read_json(
        json_text=json_str, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )
    cirq.testing.assert_json_roundtrip_works(
        circuit, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )

    circuit = cirq.Circuit(
        [
            lsp.Merge(2, True).on(qubit_a, qubit_b),
            lsp.Split([1, 1], True).on(qubit_a, qubit_b),
            lsp.SyndromeExtract(1, 1).on(qubit_a),
            lsp.ErrorCorrect(1).on(qubit_b),
            lsp.Distil("T").on(*factory_block),
            lsp.Distil("Toffoli").on(*factory_block[:23]),
            lsp.Move(zone="interact").on_each(qubit_a, qubit_b),
            lsp.Move(zone=None).on(qubit_a, qubit_b),
            lsp.Move(zone="measure").on(qubit_a),
        ]
    )
    json_str = cirq.to_json(circuit)
    new_circuit = cirq.read_json(
        json_text=json_str, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )
    assert new_circuit == circuit
    cirq.testing.assert_json_roundtrip_works(
        circuit, resolvers=[lsp.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    )


def test_repr() -> None:
    qa, qb = cirq.LineQubit.range(2)
    factory_block = cirq.LineQubit.range(31)
    merge = lsp.Merge(2, smooth=False).on(qa, qb)
    assert (
        repr(merge)
        == "lsp.Merge(num_qubits=2, smooth=False).on(cirq.LineQubit(0), cirq.LineQubit(1))"
    )

    split = lsp.Split([1, 1], smooth=False).on(qa, qb)
    assert (
        repr(split)
        == "lsp.Split(partitions=[1, 1], smooth=False).on(cirq.LineQubit(0), cirq.LineQubit(1))"
    )

    se = lsp.SyndromeExtract(1, 5).on(qa)
    assert repr(se) == "lsp.SyndromeExtract(num_qubits=1, rounds=5).on(cirq.LineQubit(0))"

    ec = lsp.ErrorCorrect(1).on(qa)
    assert repr(ec) == "lsp.ErrorCorrect(num_qubits=1).on(cirq.LineQubit(0))"

    cult = lsp.Cultivate(7).on(qa)
    assert repr(cult) == "lsp.Cultivate(theta=7).on(cirq.LineQubit(0))"

    dist_t = lsp.Distil("T").on(*factory_block)
    assert (
        repr(dist_t)
        == "lsp.Distil(T)(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3), cirq.LineQubit(4), cirq.LineQubit(5), cirq.LineQubit(6), cirq.LineQubit(7), cirq.LineQubit(8), cirq.LineQubit(9), cirq.LineQubit(10), cirq.LineQubit(11), cirq.LineQubit(12), cirq.LineQubit(13), cirq.LineQubit(14), cirq.LineQubit(15), cirq.LineQubit(16), cirq.LineQubit(17), cirq.LineQubit(18), cirq.LineQubit(19), cirq.LineQubit(20), cirq.LineQubit(21), cirq.LineQubit(22), cirq.LineQubit(23), cirq.LineQubit(24), cirq.LineQubit(25), cirq.LineQubit(26), cirq.LineQubit(27), cirq.LineQubit(28), cirq.LineQubit(29), cirq.LineQubit(30))"
    )

    dist_toff = lsp.Distil("Toffoli").on(*factory_block[:23])
    assert (
        repr(dist_toff)
        == "lsp.Distil(Toffoli)(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), cirq.LineQubit(3), cirq.LineQubit(4), cirq.LineQubit(5), cirq.LineQubit(6), cirq.LineQubit(7), cirq.LineQubit(8), cirq.LineQubit(9), cirq.LineQubit(10), cirq.LineQubit(11), cirq.LineQubit(12), cirq.LineQubit(13), cirq.LineQubit(14), cirq.LineQubit(15), cirq.LineQubit(16), cirq.LineQubit(17), cirq.LineQubit(18), cirq.LineQubit(19), cirq.LineQubit(20), cirq.LineQubit(21), cirq.LineQubit(22))"
    )
    move = lsp.Move(zone="interact").on_each(qa, qb)
    assert (
        repr(move)
        == "[lsp.Move(zone=interact).on(cirq.LineQubit(0)), lsp.Move(zone=interact).on(cirq.LineQubit(1))]"
    )

    move = lsp.Move(zone=None).on(qa, qb)
    assert repr(move) == "lsp.Move(zone=None).on(cirq.LineQubit(0), cirq.LineQubit(1))"

    move = lsp.Move(zone="measure").on(qa)
    assert repr(move) == "lsp.Move(zone=measure).on(cirq.LineQubit(0))"


def test_patch_eq_and_hash() -> None:
    patch1 = lsp.RotatedCodePatch(3)
    patch2 = lsp.RotatedCodePatch(5)
    assert patch1 != patch2
    assert hash(patch1) == 3
    assert hash(patch2) == 5

    patch3 = lsp.BufferCodePatch(3, smooth=True)
    patch4 = lsp.BufferCodePatch(3, smooth=False)
    assert patch3 != patch4

    patch5 = lsp.IntermediatePatch(3, smooth=True)
    patch6 = lsp.IntermediatePatch(5, smooth=False)

    patch7 = lsp.EndpointPatch(3, smooth=True)
    patch8 = lsp.EndpointPatch(5, smooth=False)

    assert patch3 != patch4
    assert patch5 != patch6
    assert patch7 != patch8

    assert patch1 != patch3
