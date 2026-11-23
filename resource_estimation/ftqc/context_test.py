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

import resource_estimation.ftqc.layout as lyt
import resource_estimation.ftqc.context as ctx


def simplified_movement_layout(
    program_qubits: int,
    t_cultivation_factories: int,
    t_distillation_factories: int,
    ccz_distillation_factories: int,
) -> lyt.Layout:
    return lyt.generate_movement_layout(
        surface_code_program_qubits=program_qubits,
        surface_code_distance=3,
        t_cultivation_factories=t_cultivation_factories,
        t_distillation_factories=t_distillation_factories,
        ccz_distillation_factories=ccz_distillation_factories,
        interaction_zone=False,
        measure_zone=False,
    )


def test_bare_movement_context() -> None:
    circuit = cirq.Circuit(cirq.I.on_each(cirq.LineQubit.range(3)))
    layout = simplified_movement_layout(3, 0, 0, 0)
    context = ctx.build_movement_context(circuit, layout)
    expected_roles = {0: 1, 1: 1, 2: 1}

    assert context.roles == expected_roles
    assert set(context.qid_to_pid) == set(cirq.LineQubit.range(3))
    assert set(context.qid_to_pid.values()) == {0, 1, 2}
    assert context.pid_to_qid == {pid: qid for qid, pid in context.qid_to_pid.items()}
    assert context.factories == {}
    assert context.blocks == {}
    assert context.distil_t is False
    assert context.cult_t is False
    assert context.dynamic is False
    assert context.transversal_cnot is True
    assert context.num_ccz_factories is 0
    assert context.num_t_factories is 0
    assert context.num_s_factories == 0
    assert context.post_op_syndrome_extraction is True
    assert context.idling is False

    # Test that removing a patch raises an error
    layout.remove(1)
    with pytest.raises(ValueError, match="does not equal patches expected"):
        _ = ctx.build_movement_context(circuit, layout)


@pytest.mark.parametrize(
    "t_cults, t_dists, ccz_dists",
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 0),  # Should raise an error
    ],
)
def test_movement_context_with_factories(t_cults: int, t_dists: int, ccz_dists: int) -> None:
    circuit = cirq.Circuit(cirq.I.on_each(cirq.LineQubit.range(3)))
    layout = simplified_movement_layout(3, t_cults, t_dists, ccz_dists)
    if t_cults and t_dists:
        with pytest.raises(
            ValueError, match="Current compilation does not support two types of T generation"
        ):
            _ = ctx.build_movement_context(
                circuit,
                layout,
                t_cultivation_factories=t_cults,
                t_distillation_factories=t_dists,
                ccz_distillation_factories=ccz_dists,
            )
        return
    context = ctx.build_movement_context(
        circuit,
        layout,
        t_cultivation_factories=t_cults,
        t_distillation_factories=t_dists,
        ccz_distillation_factories=ccz_dists,
    )
    assert sum(role == 1 for role in context.roles.values()) == 3
    assert sum(role == 4 for role in context.roles.values()) == t_cults + 31 * t_dists
    assert sum(role == 5 for role in context.roles.values()) == 23 * ccz_dists

    assert context.cult_t is bool(t_cults)
    assert context.distil_t is bool(t_dists)
    assert context.num_t_factories == t_cults + t_dists
    assert context.num_ccz_factories == ccz_dists


def test_column_context() -> None:
    circuit = cirq.Circuit(cirq.I.on_each(cirq.LineQubit.range(3)))
    layout = lyt.generate_column_layout(3, 3)
    context = ctx.build_column_context(circuit, layout, 3)
    expected_roles = {
        0: 3,
        1: 2,
        2: 1,
        3: 2,
        4: 1,
        5: 2,
        6: 3,  # SAQAQAS
        7: 4,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 4,  # TAAAAAT
        14: 3,
        15: 2,
        16: 1,
        17: 2,
        18: 2,
        19: 2,
        20: 3,  # SAQAAAS
        21: 4,
        22: 2,
        23: 2,
        24: 2,
        25: 2,
        26: 2,
        27: 4,  # TAAAAAT
    }
    assert context.roles == expected_roles

    # Logical qubits occupy three of the four available logical slots.
    assert set(context.qid_to_pid[q] for q in cirq.LineQubit.range(3)) == {2, 4, 16}

    # The unused logical slot becomes an ancilla.
    assert context.qid_to_pid[cirq.NamedQubit("2-4-2")] == 18

    # Mappings are reversible.
    assert context.pid_to_qid == {pid: qid for qid, pid in context.qid_to_pid.items()}

    assert context.factories == {}
    assert context.blocks == {}
    assert context.num_s_factories == 8
    assert context.num_t_factories == 8
    assert context.num_ccz_factories == 0

    assert context.distil_t is False
    assert context.cult_t is True
    assert context.dynamic is False
    assert context.transversal_cnot is False

    # Test that removing a patch raises an error
    layout.remove(1)
    with pytest.raises(ValueError, match="does not equal patches expected"):
        _ = ctx.build_column_context(circuit, layout, 3)
