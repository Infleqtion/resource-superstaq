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
import math
from functools import cache
import cirq
import mpmath
import numpy as np
import pygridsynth
from tqdm import tqdm


# pygridsynth comes from https://www.mathstat.dal.ca/~selinger/newsynth/
@cache
def approx_rz(theta: float, epsilon: float) -> list[str]:
    if math.isclose(theta, np.pi, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -np.pi, abs_tol=epsilon, rel_tol=0.0
    ):
        return "Z"
    if math.isclose(theta, np.pi / 2, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -3 * np.pi / 2, abs_tol=epsilon, rel_tol=0.0
    ):
        return "S"
    if math.isclose(theta, np.pi / 4, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -7 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0
    ):
        return "T"
    if math.isclose(theta, 3 * np.pi / 2, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -np.pi / 2, abs_tol=epsilon, rel_tol=0.0
    ):
        return "ZS"
    if math.isclose(theta, 3 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -5 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0
    ):
        return "ST"
    if math.isclose(theta, 5 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -3 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0
    ):
        return "ZT"
    if math.isclose(theta, 7 * np.pi / 4, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, -np.pi / 4, abs_tol=epsilon, rel_tol=0.0
    ):
        return "ZST"
    if math.isclose(theta, 0, abs_tol=epsilon, rel_tol=0.0) or math.isclose(
        theta, 2 * np.pi, abs_tol=epsilon, rel_tol=0.0
    ):
        return "I"
    mpmath.mp.dps = 128
    mpmath.mp.pretty = True
    stdout = pygridsynth.gridsynth_gates(mpmath.mpmathify(theta), mpmath.mpmathify(epsilon))
    hst_str = str(stdout)
    return hst_str


def process_cirq_str(
    circ: cirq.Circuit, gates: list[str], q: cirq.GridQubit | cirq.LineQubit | cirq.NamedQubit
) -> cirq.Operation:
    """
    Maps list of strings representing an Rz angle decomposition to a cirq gate
    The list is reversed because gridsynth returns gates in matrix order instead of circuit operation order
    """
    for g in gates[::-1]:
        if g == "H":
            circ += cirq.H.on(q)
        elif g == "T":
            circ += cirq.T.on(q)
        elif g == "W":  # global phase
            pass
        elif g == "S":
            circ += cirq.S.on(q)
        elif g == "X":
            circ += cirq.X.on(q)
        elif g == "I":
            pass
        elif g == "Z":
            circ += cirq.Z.on(q)
        else:
            raise ValueError(f"{g} is not in [H, S, T, W, X, I, Z]")


def cin_cliffs(gate: cirq.Gate) -> bool:
    """
    Helper function for checking Cliffordness
    """
    return gate in [cirq.H, cirq.S, cirq.Z, cirq.CNOT, cirq.I, cirq.X]


def compile_cirq_to_clifford_t(circ: cirq.Circuit, eps: float, verbose=True) -> cirq.Circuit:
    """
    Synthesizes the Clifford + Rz circuit into a Clifford + T circuit
    The eps parameter defines the maximum allowable error in the angle of each synthesized Rz gate
    """
    newcirc = cirq.Circuit()
    for moment in tqdm(circ.moments, colour="cyan", disable=not verbose):
        for op in moment:
            qubits = op.qubits
            gate = op.gate
            if cin_cliffs(gate) and gate is not None:
                newcirc += gate.on(*qubits)
            elif gate in cirq.Gateset(cirq.MeasurementGate, cirq.ResetChannel):
                newcirc += gate.on(*qubits)
            else:
                if not isinstance(gate, cirq.Rz):
                    print("Non clifford+Rz gate!")
                    print(gate)
                    raise ValueError
                else:
                    theta = gate._rads
                    gates = approx_rz(theta, eps)
                    process_cirq_str(newcirc, gates, qubits[0])
    return newcirc


def toffoli_decompose(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Decomposes TOFFOLI gates in circuit according to canned decomposition
    Does not optimize the circuit
    Implements T dagger with ZST
    """

    # TODO: Pretty sure there is a faster way to do this like the way we do ft compile now
    def mapper(op, idx):
        if op in cirq.GateFamily(cirq.TOFFOLI):
            a, b, c = op.qubits
            return cirq.Circuit(
                [
                    cirq.H.on(c),
                    cirq.CNOT.on(b, c),
                    cirq.Z.on(c),
                    cirq.S.on(c),
                    cirq.T.on(c),
                    cirq.CNOT(a, c),
                    cirq.T.on(c),
                    cirq.CNOT.on(b, c),
                    cirq.Z.on(c),
                    cirq.S.on(c),
                    cirq.T.on(c),
                    cirq.CNOT.on(a, c),
                    cirq.T.on_each(b, c),
                    cirq.H.on(c),
                    cirq.CNOT.on(a, b),
                    cirq.T.on_each(a, b),
                    cirq.Z.on(b),
                    cirq.S.on(b),
                    cirq.CNOT.on(a, b),
                ]
            )
        return op

    return cirq.map_operations_and_unroll(circuit, mapper)
