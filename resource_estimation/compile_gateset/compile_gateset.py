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

import cirq

from resource_estimation.compile_gateset.cliff_rz import CliffRzGateset, CliffPhXZGateset
from resource_estimation.compile_gateset.clifford_t import compile_cirq_to_clifford_t

_CLIFFORD_T_REQUIRED_GATES = (cirq.H, cirq.S, cirq.Z, cirq.X, cirq.CNOT, cirq.T)
_CLIFFORD_T_OPTIONAL_GATES = (cirq.I, cirq.MeasurementGate, cirq.ResetChannel)
_CLIFFORD_T_REQUIRED_FAMILIES = frozenset(
    cirq.GateFamily(gate) for gate in _CLIFFORD_T_REQUIRED_GATES
)
_CLIFFORD_T_ALLOWED_FAMILIES = _CLIFFORD_T_REQUIRED_FAMILIES | frozenset(
    cirq.GateFamily(gate) for gate in _CLIFFORD_T_OPTIONAL_GATES
)


def clifford_rz_gateset(atol: float = 1e-8) -> cirq.Gateset:
    """Returns the default Clifford + Rz gateset.

    Args:
        atol: Absolute tolerance used when decomposing and simplifying operations.

    Returns:
        A Cirq gateset for compiling circuits to Clifford + Rz.
    """
    return CliffRzGateset(atol=atol)


def clifford_phxz_gateset(atol: float = 1e-8) -> cirq.Gateset:
    """Returns the default Clifford + PhasedXZ gateset
    Args:
        atol: Absolute tolerance used when decomposing and simplifying operations.
    Returns:
        A Cirq gateset for compiling circuits to Clifford + PhasedXZ.
    """
    return CliffPhXZGateset(atol=atol)


def clifford_t_gateset(atol: float) -> cirq.Gateset:
    """Returns the default Clifford + T gateset.

    Args:
        atol: Maximum allowable approximation error for each synthesized Rz rotation.

    Returns:
        A Cirq gateset for Clifford + T compilation. The gateset carries `_atol` so
        `compile_gateset` can dispatch to the custom Rz synthesis path.
    """
    gateset = cirq.Gateset(*_CLIFFORD_T_REQUIRED_GATES, *_CLIFFORD_T_OPTIONAL_GATES)
    gateset._atol = atol
    return gateset


def _is_clifford_t_gateset(gateset: cirq.Gateset) -> bool:
    """Returns whether a gateset matches the supported Clifford + T gate families."""
    gate_families = gateset.gates
    return _CLIFFORD_T_REQUIRED_FAMILIES <= gate_families <= _CLIFFORD_T_ALLOWED_FAMILIES


def compile_gateset(
    circuit: cirq.Circuit,
    gateset: cirq.Gateset = clifford_rz_gateset(),
    verbose: bool = True,
) -> cirq.Circuit:
    """Compiles a circuit into the specified gateset.

    Args:
        circuit: Input Cirq circuit to compile.
        gateset: Target Cirq gateset. Clifford + T gatesets are handled by the
            custom synthesis implementation; all other gatesets are passed to
            `cirq.optimize_for_target_gateset`.
        verbose: Whether to show progress output for Clifford + T synthesis.

    Returns:
        A compiled Cirq circuit using operations from the requested gateset.

    Raises:
        ValueError: If `gateset` is Clifford + T but does not carry an `_atol`
            attribute. Use `clifford_t_gateset(atol=...)` to construct the
            default Clifford + T target.
    """
    if _is_clifford_t_gateset(gateset):
        if not hasattr(gateset, "_atol"):
            raise ValueError("Clifford + T gatesets must define an `_atol` attribute.")
        return compile_cirq_to_clifford_t(circuit, eps=gateset._atol, verbose=verbose)

    return cirq.optimize_for_target_gateset(circuit, gateset=gateset)
