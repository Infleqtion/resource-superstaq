import cirq
import cirq_superstaq as css


def ccz_8_to_1() -> cirq.Circuit:
    """Function to perform a 8-to-1 CCZ magic state distillation.
       Takes eight Ts to make one CCZ
        Reference: http://arxiv.org/abs/1812.01238 page 7 figure 5.

    Returns:
        The magic state distillation circuit.
    """
    cir = cirq.Circuit()

    qubits = cirq.LineQubit.range(15)

    for q in qubits:
        cir.append([cirq.reset(q)])
    cir.append(css.Barrier(15).on(*qubits))

    cir.append(
        cirq.H(qubits[i]) for i in range(11, 15)
    )
    cir.append(css.Barrier(15).on(*qubits))

    idx11 = [0,3,4,5,6]
    idx12 = list(range(3,11))
    idx13 = [2,3,5,7,9]
    idx14 = [1,3,4,7,8]
    cir.append(
        cirq.CNOT(qubits[11], qubits[i]) for i in idx11
    )
    cir.append(
        cirq.CNOT(qubits[12], qubits[i]) for i in idx12
    )
    cir.append(
        cirq.CNOT(qubits[13], qubits[i]) for i in idx13
    )
    cir.append(
        cirq.CNOT(qubits[14], qubits[i]) for i in idx14
    )

    # cir.append(
    #     cirq.H(qubits[i]) for i in range(11, 15)
    # )
    cir.append(css.Barrier(15).on(*qubits))

    cir.append(
        cirq.T(qubits[i]) for i in range(3,11)
    )
    cir.append(css.Barrier(15).on(*qubits))

    cir.append(
        cirq.H(qubits[i]) for i in range(3,15)
    )
    cir.append(css.Barrier(15).on(*qubits))

    cir.append(
            cirq.measure(qubits[i], key="m" + str(i)) for i in range(3, 15)
    )

    return cir

x = ccz_8_to_1()
print(x)
for i, moment in enumerate(x):
    print(f"Moment {i}:")
    for op in moment:
        print(" ", op)
