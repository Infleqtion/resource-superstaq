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

from typing import TYPE_CHECKING

import cirq
import cirq_superstaq as css
import numpy as np

if TYPE_CHECKING:
    from types import NotImplementedType
    from typing import Iterator

    from resource_estimation.ftqc.cost_types import GateKey

# warnings.filterwarnings(category=FutureWarning, action="ignore")


@cirq.transformer
def eject_z(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    atol: float = 1e-8,
) -> cirq.Circuit:
    circuit = cirq.Circuit(circuit)
    """Pushes Z gates towards the end of the circuit"""
    backlog = dict.fromkeys(circuit.all_qubits(), 0.0)

    def _map_fn(op: cirq.Operation) -> Iterator[cirq.Operation]:
        if isinstance(op.gate, cirq.ZPowGate):
            backlog[op.qubits[0]] += op.gate.exponent
        else:
            for q in op.qubits:
                exponent = cirq.canonicalize_half_turns(backlog[q])
                if not (
                    np.isclose(exponent, 0, atol=atol)
                    or cirq.is_measurement(op)
                    or cirq.definitely_commutes(cirq.Z(q), op)
                ):
                    yield cirq.Z(q) ** exponent
                    backlog[q] = 0.0

            yield op

    circuit = circuit.map_operations(_map_fn)
    for q, exponent in backlog.items():
        exponent = cirq.canonicalize_half_turns(exponent)
        if not np.isclose(exponent, 0, atol=atol):
            circuit += cirq.Z(q) ** exponent

    return circuit


@cirq.transformer
def phx_to_zhzhz(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    atol: float = 1e-8,
) -> cirq.Circuit:
    """Converts PhasedX gates to ZPOW gates and Hadamards with handling for special angles"""

    # Adding this as its own thing
    def _map_fn(op: cirq.Operation, _: int) -> cirq.OP_TREE:
        if not isinstance(op.gate, cirq.PhasedXPowGate):
            return op

        q = op.qubits[0]
        p, t = op.gate.phase_exponent, op.gate.exponent
        t = cirq.canonicalize_half_turns(t)

        if np.isclose(t, 0.0, atol=atol):
            return []

        if np.isclose(t, 0.5, atol=atol):
            return [
                cirq.Z(q) ** (-0.5 - p),
                cirq.H(q),
                cirq.Z(q) ** (-0.5 + p),
            ]

        if np.isclose(t, -0.5, atol=atol):
            return [
                cirq.Z(q) ** (0.5 - p),
                cirq.H(q),
                cirq.Z(q) ** (0.5 + p),
            ]

        if np.isclose(t, 1.0, atol=atol) or np.isclose(t, -1.0, atol=atol):
            return [
                cirq.Z(q) ** -p,
                cirq.X(q),
                cirq.Z(q) ** p,
            ]

        return [
            cirq.ZPowGate(exponent=-p).on(q),
            cirq.H.on(q),
            cirq.ZPowGate(exponent=t).on(q),
            cirq.H.on(q),
            cirq.ZPowGate(exponent=p).on(q),
        ]

    return cirq.Circuit(
        cirq.map_operations_and_unroll(
            circuit,
            _map_fn,
            tags_to_ignore=context.tags_to_ignore if context else (),
            deep=context.deep if context else False,
        )
    )


@cirq.transformer
def zpow_to_rz(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    """Converts ZPOW gates to Rz gates minding special angle cases and including the angle factor"""

    # Maybe this should be a transformer or something?
    def _map_fn(
        op: cirq.Operation,
        _: int,
    ) -> (
        cirq.Operation
        | cirq.SingleQubitPauliStringGateOperation
        | list[cirq.Operation | cirq.SingleQubitPauliStringGateOperation]
    ):
        if not isinstance(op.gate, cirq.ZPowGate):
            return op
        if css.approx_eq_mod(op.gate.exponent, 1.0, 2, atol=1e-9):
            return cirq.Z.on(*op.qubits)
        # mod 1.5 should be first because anything mod 1.5 is also mod .5
        if css.approx_eq_mod(op.gate.exponent, 1.5, 2, atol=1e-5):
            return [cirq.Z.on(op.qubits[0]), cirq.S.on(op.qubits[0])]
        if css.approx_eq_mod(op.gate.exponent, 0.5, 2, atol=1e-5):
            return cirq.S.on(op.qubits[0])
        return cirq.Rz(rads=op.gate.exponent * np.pi).on(op.qubits[0])

    return cirq.Circuit(
        cirq.map_operations_and_unroll(
            circuit,
            _map_fn,
            tags_to_ignore=context.tags_to_ignore if context else (),
            deep=context.deep if context else False,
        )
    )


class CliffRzGateset(cirq.TwoQubitCompilationTargetGateset):
    """
    A Gateset for a Clifford + Rz
    """

    def __init__(self, atol: float = 1e-8, allow_ccz: bool = False) -> None:
        self._atol = atol
        gateset: list[GateKey | cirq.GateFamily] = [
            cirq.GateFamily(cirq.CX, ignore_global_phase=False),
            cirq.MeasurementGate,
            cirq.PhasedXZGate,
            cirq.GlobalPhaseGate,
            css.Barrier,
        ]
        if allow_ccz:
            gateset.append(cirq.GateFamily(cirq.CCZ, ignore_global_phase=False))

        super().__init__(
            *gateset,
            preserve_moment_structure=False,
            reorder_operations=False,  # Enabling makes a shorter circuit but probably way too slow
        )
        self._allow_ccz = allow_ccz

    @property
    def num_qubits(self) -> int:
        return 3 if self._allow_ccz else 2

    def _decompose_two_qubit_operation(
        self,
        op: cirq.Operation,
        moment_idx: int = -1,
    ) -> cirq.OP_TREE:
        if op in self:  # Had to re-add this line because CXPowGate made its way in here
            return op

        q0, q1 = op.qubits
        if op.gate == cirq.CZ:
            return [cirq.H.on(q1), cirq.CNOT.on(q0, q1), cirq.H.on(q1)]
        mat = cirq.unitary(op)
        return cirq.two_qubit_matrix_to_cz_operations(
            q0,
            q1,
            mat,
            allow_partial_czs=False,
            atol=self._atol,
        )

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, moment_idx: int = -1
    ) -> cirq.OP_TREE | None | NotImplementedType:
        if op in self:
            return op

        if op.gate == cirq.CCX:
            return [
                cirq.H(op.qubits[2]),
                cirq.CCZ(*op.qubits),
                cirq.H(op.qubits[2]),
            ]

        return super()._decompose_multi_qubit_operation(op, moment_idx)

    @property
    def preprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run before decomposing individual operations."""
        return [cirq.drop_negligible_operations, *super().preprocess_transformers]

    @property
    def postprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        """List of transformers which should be run after decomposing individual operations."""
        return [
            cirq.merge_single_qubit_gates_to_phased_x_and_z,
            cirq.create_transformer_with_kwargs(eject_z, atol=self._atol),
            cirq.create_transformer_with_kwargs(cirq.drop_negligible_operations, atol=self._atol),
            cirq.drop_empty_moments,
            phx_to_zhzhz,
            cirq.create_transformer_with_kwargs(eject_z, atol=self._atol),
            zpow_to_rz,
            cirq.create_transformer_with_kwargs(cirq.drop_negligible_operations, atol=self._atol),
            cirq.align_left,
            cirq.synchronize_terminal_measurements,
            cirq.drop_empty_moments,
        ]

    # TODO: add a special decomposition for toffoli


_CLIFFORD_T_REQUIRED_GATES = (cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.T)
_CLIFFORD_T_OPTIONAL_GATES = (cirq.I, cirq.MeasurementGate, cirq.ResetChannel)


class CliffordTGateset(cirq.Gateset):
    def __init__(self, atol: float) -> None:
        super().__init__(
            *_CLIFFORD_T_REQUIRED_GATES,
            *_CLIFFORD_T_OPTIONAL_GATES,
        )
        self.atol = atol
