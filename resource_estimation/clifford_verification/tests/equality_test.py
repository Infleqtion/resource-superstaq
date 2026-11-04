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
import pytest

from resource_estimation.clifford_verification import cliffords_equal


def test_unitaries_and_idle_wires() -> None:
    a, b = cirq.LineQubit.range(2)
    assert cliffords_equal(
        cirq.Circuit(cirq.H(b), cirq.CNOT(a, b), cirq.H(b)), cirq.Circuit(cirq.CZ(a, b))
    )
    assert cliffords_equal(cirq.Circuit(), cirq.Circuit(cirq.I(a)))
    assert not cliffords_equal(cirq.Circuit(), cirq.Circuit(cirq.Z(a)))
    assert cliffords_equal(
        cirq.Circuit(cirq.X(a)), cirq.Circuit(cirq.X(a), cirq.global_phase_operation(-1))
    )
    assert not cliffords_equal(cirq.Circuit(cirq.H(a)), cirq.Circuit(cirq.H(b)))


def test_cnot_ppm_interface() -> None:
    a, anc, b = cirq.LineQubit.range(3)
    original = cirq.Circuit(cirq.CNOT(a, b))
    body = cirq.Circuit(
        cirq.reset(anc),
        cirq.measure_single_paulistring(cirq.X(a) * cirq.X(anc) * cirq.X(b), key="p"),
        cirq.measure(a, key="m"),
    )
    corrected = body + cirq.Circuit(
        cirq.Z(anc).with_classical_controls("p"),
        cirq.X(anc).with_classical_controls("m"),
        cirq.X(b).with_classical_controls("m"),
    )
    assert cliffords_equal(original, corrected)
    assert not cliffords_equal(original, body)
    # Remove each correction separately: the channel must change.
    corrections = list(corrected.all_operations())[3:]
    for omitted in range(3):
        broken = body + cirq.Circuit(op for j, op in enumerate(corrections) if j != omitted)
        assert not cliffords_equal(original, broken)


def test_records_are_traced_not_summed_as_amplitudes() -> None:
    a = cirq.LineQubit(0)
    measure = cirq.measure_single_paulistring(cirq.Z(a), key="m")
    dephase = cirq.Circuit(measure)
    assert not cliffords_equal(dephase, cirq.Circuit(cirq.I(a)))
    assert cliffords_equal(dephase, cirq.Circuit(measure, measure))
    assert cliffords_equal(
        dephase, cirq.Circuit(cirq.measure_single_paulistring(-cirq.Z(a), key="renamed"))
    )
    # Z conditioned on its own Z outcome changes branch phase only.
    assert cliffords_equal(dephase, dephase + cirq.Circuit(cirq.Z(a).with_classical_controls("m")))
    # X conditioned on its own Z outcome resets the output and changes the channel.
    assert not cliffords_equal(
        dephase, dephase + cirq.Circuit(cirq.X(a).with_classical_controls("m"))
    )


def test_ancilla_preparation_and_discard() -> None:
    a, b = cirq.LineQubit.range(2)
    original = cirq.Circuit(cirq.I(a))
    assert cliffords_equal(
        original, cirq.Circuit(cirq.I(a), cirq.reset(b), cirq.H(b), cirq.measure(b))
    )
    assert not cliffords_equal(
        original, cirq.Circuit(cirq.reset(b), cirq.CNOT(a, b), cirq.measure(b))
    )
    assert cliffords_equal(
        cirq.Circuit(cirq.reset(a), cirq.H(a)), cirq.Circuit(cirq.reset(b), cirq.H(b))
    )
    assert cliffords_equal(cirq.Circuit(cirq.reset(a), cirq.H(a), cirq.measure(a)), cirq.Circuit())


def test_invalid_interfaces_and_resets() -> None:
    a = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="counts"):
        cliffords_equal(cirq.Circuit(cirq.I(a)), cirq.Circuit(cirq.reset(a)))
    circuit = cirq.Circuit(cirq.H(a), cirq.reset(a))
    with pytest.raises(ValueError, match="resets"):
        cliffords_equal(circuit, circuit)
    with pytest.raises(ValueError, match="Moment"):
        cliffords_equal(cirq.Circuit(cirq.T(a)), cirq.Circuit())


def test_inferred_gadget_interface() -> None:
    a, anc, b = cirq.NamedQubit("a_in"), cirq.NamedQubit("a_out"), cirq.NamedQubit("b")
    original = cirq.Circuit(cirq.CNOT(a, b))
    gadget = cirq.Circuit(
        cirq.reset(anc),
        cirq.measure_single_paulistring(cirq.X(a) * cirq.X(anc) * cirq.X(b), key="p"),
        cirq.measure(a, key="m"),
        cirq.Z(anc).with_classical_controls("p"),
        cirq.X(anc).with_classical_controls("m"),
        cirq.X(b).with_classical_controls("m"),
    )
    assert cliffords_equal(original, gadget)
    assert cliffords_equal(gadget, original)
    assert not cliffords_equal(cirq.Circuit(cirq.CZ(a, b)), gadget)


def test_readout_conventions() -> None:
    a = cirq.LineQubit(0)
    assert cliffords_equal(
        cirq.Circuit(cirq.measure(a)), cirq.Circuit(cirq.measure(a, invert_mask=(True,)))
    )
    # A later quantum operation makes the measured wire an output again.
    measurement = cirq.Circuit(cirq.measure(a, key="m"), cirq.X(a))
    ppm = cirq.Circuit(cirq.measure_single_paulistring(cirq.Z(a), key="m"), cirq.X(a))
    assert cliffords_equal(measurement, ppm)
    with pytest.raises(ValueError, match="single-qubit"):
        cliffords_equal(
            cirq.Circuit(cirq.measure(a, cirq.LineQubit(1))),
            cirq.Circuit(cirq.measure(a, cirq.LineQubit(1))),
        )
