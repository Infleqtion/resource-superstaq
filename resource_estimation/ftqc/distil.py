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

from math import pi
from typing import TYPE_CHECKING, Literal

import cirq

from resource_estimation.ftqc.compile_ftqc import add_moves
from resource_estimation.ftqc.estimate import ResourceEstimator
from resource_estimation.ftqc.lattice_surgery_primitives import Cultivate
from resource_estimation.ftqc.layout import MovementDistillery
from resource_estimation.typing import CostDict

if TYPE_CHECKING:
    from resource_estimation.ftqc.architecture import Architecture


def distil_15_to_1() -> cirq.Circuit:
    """Generates a 15-to-1 non-recursive distillation circuit.
    The circuit is a compact version of the one in https://github.com/Infleqtion/client-superstaq/blob/main/cirq-superstaq/cirq_superstaq/circuits/msd.py
    T gates are produced via cultivation.
    The assumed qubit footprint is based on Movement Architectures.
    C0  Q0   Q8  C8
    C1  Q1   Q9  C9
    C2  Q2  Q10  C10
    C3  Q3  Q11  C11
    C4  Q4  Q12  C12
    C5  Q5  Q13  C13
    C6  Q6  Q14  C14
    C7  Q7    F  <- Output Factory Qubit
    """
    qubits = cirq.LineQubit.range(15) + [cirq.NamedQubit("F")]
    cults = [cirq.NamedQubit(f"C{i}") for i in range(15)]
    exp = cirq.Circuit(
        [
            cirq.ResetChannel().on_each(*qubits),
            Cultivate(pi / 4).on_each(*cults),
            cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[3]),
            cirq.H(qubits[7]),
            cirq.H(qubits[15]),
            cirq.CNOT.on(qubits[7], qubits[14]),
            cirq.CNOT.on(qubits[3], qubits[12]),
            cirq.CNOT.on(qubits[1], qubits[10]),
            cirq.CNOT.on(qubits[0], qubits[6]),
            cirq.CNOT.on(qubits[7], qubits[13]),
            cirq.CNOT.on(qubits[3], qubits[11]),
            cirq.CNOT.on(qubits[1], qubits[14]),
            cirq.CNOT.on(qubits[0], qubits[10]),
            cirq.CNOT.on(qubits[7], qubits[12]),
            cirq.CNOT.on(qubits[3], qubits[6]),
            cirq.CNOT.on(qubits[1], qubits[13]),
            cirq.CNOT.on(qubits[0], qubits[14]),
            cirq.CNOT.on(qubits[7], qubits[8]),
            cirq.CNOT.on(qubits[3], qubits[14]),
            cirq.CNOT.on(qubits[1], qubits[9]),
            cirq.CNOT.on(qubits[0], qubits[4]),
            cirq.CNOT.on(qubits[7], qubits[9]),
            cirq.CNOT.on(qubits[3], qubits[4]),
            cirq.CNOT.on(qubits[1], qubits[2]),
            cirq.CNOT.on(qubits[0], qubits[8]),
            cirq.CNOT.on(qubits[-1], qubits[14]),
            cirq.CNOT.on(qubits[14], qubits[11]),
            cirq.CNOT.on(qubits[7], qubits[10]),
            cirq.CNOT.on(qubits[3], qubits[5]),
            cirq.CNOT.on(qubits[1], qubits[6]),
            cirq.CNOT.on(qubits[0], qubits[12]),
            cirq.CNOT.on(qubits[14], qubits[5]),
            cirq.CNOT.on(qubits[7], qubits[11]),
            cirq.CNOT.on(qubits[3], qubits[13]),
            cirq.CNOT.on(qubits[1], qubits[5]),
            cirq.CNOT.on(qubits[0], qubits[2]),
            cirq.CNOT.on(qubits[14], qubits[9]),
            cirq.CNOT.on(qubits[14], qubits[8]),
            cirq.CNOT.on(qubits[14], qubits[4]),
            cirq.CNOT.on(qubits[14], qubits[2]),
        ],
    )
    exp.append(cirq.CNOT.on(ctrl, trgt) for ctrl, trgt in zip(cults, qubits[:-1]))
    exp.append(cirq.Moment(cirq.measure_each(*cults)))
    exp.append(
        cirq.Moment(
            cirq.S.on_each(*qubits[:-1])  # Technically should be based on the measurement outcome
        )
    )
    exp.append(cirq.Moment(cirq.H.on_each(*qubits[:-1])))
    exp.append(cirq.Moment(cirq.measure_each(*qubits[:-1])))

    # Remap circuit to a logical grid
    qmap: dict[cirq.Qid, cirq.Qid] = {qubits[-1]: cirq.GridQubit(7, 2)}
    for idx, (q, f) in enumerate(zip(qubits, cults)):
        row = idx if idx < 8 else idx - 8
        col1 = 1 if idx < 8 else 2
        col2 = 0 if idx < 8 else 3
        qmap[q] = cirq.GridQubit(row, col1)
        qmap[f] = cirq.GridQubit(row, col2)
    mapped_circuit = cirq.Circuit(moment.transform_qubits(qmap) for moment in exp)
    return mapped_circuit


def ccz_8_to_1() -> cirq.Circuit:
    """Function to perform a 8-to-1 CCZ magic state distillation.
       Takes eight Ts to make one CCZ
       Reference: http://arxiv.org/abs/1812.01238 page 7 figure 5.
       Q11 Q12 Q13 Q14
       C0  Q3  Q7  C4
       C1  Q4  Q8  C5
       C2  Q5  Q9  C6
       C3  Q6  Q10 C7
       Q0  Q1  Q2  __ <- three out qubits

    Returns:
        The magic state distillation circuit.
    """
    cir = cirq.Circuit()
    qubits = cirq.LineQubit.range(15)
    cults = [cirq.NamedQubit(f"C{i}") for i in range(8)]

    for q in qubits:
        cir.append(cirq.reset(q))
    cir.append(Cultivate(pi / 4).on_each(*cults))
    cir.append(cirq.H(qubits[i]) for i in range(11, 15))

    idx11 = [0, 3, 4, 5, 6]
    idx12 = list(range(3, 11))
    idx13 = [2, 3, 5, 7, 9]
    idx14 = [1, 3, 4, 7, 8]
    cir.append(cirq.CNOT(qubits[11], qubits[i]) for i in idx11)
    cir.append(cirq.CNOT(qubits[12], qubits[i]) for i in idx12)
    cir.append(cirq.CNOT(qubits[13], qubits[i]) for i in idx13)
    cir.append(cirq.CNOT(qubits[14], qubits[i]) for i in idx14)

    cir.append(cirq.CNOT.on(ctrl, trgt) for ctrl, trgt in zip(qubits[3:11], cults))

    cir.append(cirq.Moment(cirq.measure_each(*cults)))
    cir.append(
        cirq.Moment(cirq.S.on_each(*qubits[3:11]))
    )  # Technically should be based on the measurement outcome
    cir.append(cirq.Moment(cirq.H.on_each(*qubits[3:15])))
    cir.append(cirq.Moment(cirq.measure_each(*qubits[3:15])))

    # Remap circuit to a logical grid
    # the out qubits
    qmap: dict[cirq.Qid, cirq.Qid] = {
        qubits[0]: cirq.GridQubit(5, 0),
        qubits[1]: cirq.GridQubit(5, 1),
        qubits[2]: cirq.GridQubit(5, 2),
    }
    # bottom four qubits
    qmap[qubits[11]] = cirq.GridQubit(0, 0)
    qmap[qubits[12]] = cirq.GridQubit(0, 1)
    qmap[qubits[13]] = cirq.GridQubit(0, 2)
    qmap[qubits[14]] = cirq.GridQubit(0, 3)
    # qubits 3-6 where Ts act on
    qmap[qubits[3]] = cirq.GridQubit(1, 1)
    qmap[qubits[4]] = cirq.GridQubit(2, 1)
    qmap[qubits[5]] = cirq.GridQubit(3, 1)
    qmap[qubits[6]] = cirq.GridQubit(4, 1)
    # qubits 7-10 where Ts act on
    qmap[qubits[7]] = cirq.GridQubit(1, 2)
    qmap[qubits[8]] = cirq.GridQubit(2, 2)
    qmap[qubits[9]] = cirq.GridQubit(3, 2)
    qmap[qubits[10]] = cirq.GridQubit(4, 2)
    # cultivation qubits next to those that need them
    qmap[cults[0]] = cirq.GridQubit(1, 0)
    qmap[cults[1]] = cirq.GridQubit(2, 0)
    qmap[cults[2]] = cirq.GridQubit(3, 0)
    qmap[cults[3]] = cirq.GridQubit(4, 0)
    qmap[cults[4]] = cirq.GridQubit(1, 3)
    qmap[cults[5]] = cirq.GridQubit(2, 3)
    qmap[cults[6]] = cirq.GridQubit(3, 3)
    qmap[cults[7]] = cirq.GridQubit(4, 3)

    mapped_circuit = cirq.Circuit(moment.transform_qubits(qmap) for moment in cir)
    return mapped_circuit


def precompute_distil_cost(
    resource: Literal["T", "CCZ"], layout: MovementDistillery, arc: Architecture
) -> CostDict:
    # Grabs prepared resource state circuit
    # Grabs template distillation block (could be matched to the one actually used)
    # Remaps resource state circuit to qubits in distillation block
    # Adds moves to zones according to the layout's rules
    # Returns resources based on the resulting sub-circuit
    mapped_circuit_factory: tuple[cirq.GridQubit, ...]
    if resource == "T":
        mapped_circuit = distil_15_to_1()
        mapped_circuit_factory = (cirq.GridQubit(7, 2),)
        layout_factory = layout.all_factories("t")[0]
    elif resource == "CCZ":
        mapped_circuit = ccz_8_to_1()
        mapped_circuit_factory = (cirq.GridQubit(5, 0), cirq.GridQubit(5, 1), cirq.GridQubit(5, 2))
        layout_factory = layout.all_factories("ccz")[0]
    else:
        raise ValueError(f"Unknown distillation resource: {resource!r}")
    circuit_block = (
        tuple(sorted(q for q in mapped_circuit.all_qubits() if q not in mapped_circuit_factory))
        + mapped_circuit_factory
    )
    layout_block = layout.distillation_block(layout_factory)
    qmap: dict[cirq.Qid, cirq.Qid] = {
        q_circuit: q_layout for q_circuit, q_layout in zip(circuit_block, layout_block)
    }
    remapped_circuit = mapped_circuit.transform_qubits(qmap)
    embedded_circuit = add_moves(remapped_circuit, layout=layout)
    estimator = ResourceEstimator(arc)
    op_time = estimator.parallel_circuit_time(embedded_circuit, layout=layout)
    moment_cost = estimator.parallel_circuit_cost(embedded_circuit, layout=layout)
    gate_cost = estimator.serial_circuit_cost(embedded_circuit, layout=layout)

    return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)
