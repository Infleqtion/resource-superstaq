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
from numpy.random import choice
from math import pi
import openfermion
import numpy as np
from openfermion.circuits import simulate_trotter
from openfermion.ops import FermionOperator
from collections import Counter
from time import time
from pathlib import Path
import sys
from sorger_shor.shor import factoring_circuit

import supermarq as sm

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# from resource_estimation.cliff_rz import GenericDevice
# from resource_estimation.layout import Layout, MovementLayout, ColumnLayout, SpeedMovementLayout
# from resource_estimation.architecture import Architecture, DefaultMovement, DefaultLattice
# from scripts.clifford_t import compile_cirq_to_clifford_t
# from resource_estimation.estimate import ResourceEstimator
# from resource_estimation.compile_ftqc import ft_compile


def give_crz(theta, controls):
    controlled_rz = cirq.ControlledGate(
        sub_gate=cirq.Rz(theta), num_controls=len(controls), control_values=controls
    )
    return controlled_rz


def give_crx(theta, controls):
    controlled_rx = cirq.ControlledGate(
        sub_gate=cirq.Rz(theta),
        num_controlls=len(controls),
        control_values=controls,
    )
    return controlled_rx


def random_pauli_phasor_circuit(num_qubits: int, moments: int):
    paulis = ["I", "X", "Y", "Z"]
    pauli_map = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    for moment in range(moments):
        random_string = "".join(choice(paulis, size=num_qubits, replace=True))
        random_qubits = choice(qubits, size=num_qubits, replace=False)
        operations = [pauli_map[s](q) for s, q in zip(random_string, random_qubits)]
        print(operations)
        # exit()
        print(random_string)
        print("*" * 100)
        s = cirq.PauliString(operations)
        print(s)

        print("*" * 100)
        phasor = cirq.PauliStringPhasor(s, exponent_neg=7 / pi)
        print(phasor)
        circuit += phasor
    return circuit


def minial_rz_circuit():
    return


# c = random_pauli_phasor_circuit(3, 1)
# print(c)
# c_ = cirq.optimize_for_target_gateset(circuit=c, gateset=cirq.CZTargetGateset())
# # c_= cirq.Circuit(cirq.decompose(c))
# # c_ = cirq.merge_single_qubit_gates_to_phxz(cirq.Circuit(cirq.decompose(c)))
# print(c_)
# angle_dict = {1: [cirq.Z], -1: [cirq.Z], .5: [cirq.S],  -.5: [cirq.Z, cirq.S]}
# decomposed_circuit = cirq.Circuit()
# for op in c_.all_operations():
#     if op.gate in cirq.GateFamily(cirq.CZ):
#         decomposed_circuit.append(op)
#     elif op.gate in cirq.GateFamily(cirq.PhasedXZGate):
#         # print(op.gate)
#         axis_phase_exponent = op.gate.axis_phase_exponent
#         axis_phase_exponent = round(axis_phase_exponent, 10)
#         z_exponent = op.gate.z_exponent
#         z_exponent = round(z_exponent, 10)
#         x_exponent = op.gate.x_exponent
#         x_exponent = round(x_exponent, 10)
#         # print("Phase Exponent:", axis_phase_exponent)
#         # print("Z Exponent:", z_exponent)
#         # print("X Exponent:", x_exponent)
#         first_z = -z_exponent
#         final_z = axis_phase_exponent + z_exponent
#         if first_z:
#             if first_z in angle_dict:
#                 for rotation in angle_dict.get(first_z):
#                     decomposed_circuit += [rotation.on(*op.qubits)]
#             else:
#                 decomposed_circuit += [cirq.Rz(rads=first_z).on(*op.qubits)]
#         if x_exponent:
#             decomposed_circuit += cirq.H.on(*op.qubits)
#             if z_exponent in angle_dict:
#                 for rotation in angle_dict.get(x_exponent):
#                     decomposed_circuit += [rotation.on(*op.qubits)]
#             else:
#                 decomposed_circuit += [cirq.Rz(rads=x_exponent).on(*op.qubits)]
#             decomposed_circuit += cirq.H.on(*op.qubits)
#         if final_z:
#             if final_z in angle_dict:
#                 for rotation in angle_dict.get(final_z):
#                     decomposed_circuit += [rotation.on(*op.qubits)]
#             else:
#                 decomposed_circuit += [cirq.Rz(rads=final_z).on(*op.qubits)]
#     else:
#         print("A"*100)
# print(decomposed_circuit)
# cirq.testing.circuit_compare.assert_circuits_have_same_unitary_given_final_permutation(c, decomposed_circuit, {qid: qid for qid in c.all_qubits()})
# print("done")


def fermi_hubbard(n, verbose=0):
    """
    Generate the circuit we have been using as our proxy for 'Hamiltonian Simulation' in the materials science context
    """
    U = 2.0
    J = -1.0
    time = 1
    final_rank = 1
    hubbard_fermion_hamiltonian = openfermion.fermi_hubbard(
        n, n, tunneling=-J, coulomb=U, periodic=False
    )
    hubbard_interaction_hamiltonian = openfermion.get_interaction_operator(
        hubbard_fermion_hamiltonian
    )
    n_qubits = openfermion.count_qubits(hubbard_interaction_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)
    if verbose > 0:
        print("Creating circuit")
    ham_circuit = cirq.Circuit(
        openfermion.circuits.trotter.simulate_trotter(qubits, hubbard_interaction_hamiltonian, time)
    )
    ham_circuit = cirq.drop_negligible_operations(ham_circuit)
    if verbose > 0:
        print("Fermi-Hubbard Circuit Qubits:", len(ham_circuit.all_qubits()))
        print("Fermi-Hubbard Circuit Moments:", len(ham_circuit))
    return ham_circuit


def map_orbitals(n_imp, n_b, method="impurity_centered"):
    impurity_sites = [f"I{ii}" for ii in range(0, n_imp)]
    bath_sites = [f"B{jj}" for jj in range(0, n_b)]
    all_sites = bath_sites + impurity_sites
    spins = ["up", "down"]
    if method == "normal":
        all_spin_orbitals = []
        for spin in spins:
            for site in all_sites:
                all_spin_orbitals.append(site + "_" + spin)

    elif method == "impurity_centered":
        all_spin_orbitals = [site + "_up" for site in all_sites] + [
            site + "_down" for site in reversed(all_sites)
        ]

    elif method == "paired":
        all_spin_orbitals = []
        for site in all_sites:
            for spin in spins:
                all_spin_orbitals.append(site + "_" + spin)

    else:
        raise ValueError("Not a valid orbital ordering method!")

    orbital_map = {}
    for ii, spin_orbital in enumerate(all_spin_orbitals):
        orbital_map[spin_orbital] = ii
    # print(all_spin_orbitals)
    # print(orbital_map)
    return impurity_sites, bath_sites, all_sites, spins, orbital_map


def three_orbital_kanamori_hamiltonian(epsilon_imp, epsilon_bath, v, u, j_ex, n_bath):
    n_imp = 3

    # Map orbitals to tensor indices
    impurity_sites, bath_sites, all_sites, spins, orbital_map = map_orbitals(
        n_imp=n_imp, n_b=n_bath, method="paired"
    )

    # Diagonal and degenerate impurity energies. For 5-site case, e.g., SrMnO3, will need to include CF splitting
    h_0 = FermionOperator()
    for ii in range(0, n_imp):
        for spin in spins:
            label = f"I{ii}_" + spin
            h_0 += FermionOperator(f"{orbital_map[label]}^ {orbital_map[label]}", epsilon_imp)
    # print("h_0", h_0, "\n")

    # U density-density term
    h_u = FermionOperator()
    for ii in range(0, n_imp):
        index_1 = orbital_map[f"I{ii}_up"]
        index_2 = orbital_map[f"I{ii}_down"]
        h_u += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u)
    # print("h_u", h_u, "\n")

    # U-2J density-density term
    h_u2j = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{jj}_" + spins[(sigma + 1) % 2]]
                h_u2j += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u - 2 * j_ex)
    # print("h_u2j", h_u2j, "\n")

    # U-3J density-density term
    h_u3j = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{jj}_" + spin]
                h_u3j += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u - 3 * j_ex)
    # print("h_u3j", h_u3j, "\n")

    # Spin-flip term
    # TODO This is where the problem is for up-down paired orbital ordering
    h_sf = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            """
            index_1 = orbital_map[f"I{ii}_" + "up"]
            index_2 = orbital_map[f"I{ii}_" + "down"]
            index_3 = orbital_map[f"I{jj}_" + "up"]
            index_4 = orbital_map[f"I{jj}_" + "down"]
            h_sf += FermionOperator(f"{index_1}^ {index_2} {index_3} {index_4}^", j_ex)
            h_sf += hermitian_conjugated(FermionOperator(f"{index_1}^ {index_2} {index_3} {index_4}^", j_ex))
            """
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{jj}_" + spins[(sigma + 1) % 2]]
                index_3 = orbital_map[f"I{ii}_" + spins[(sigma + 1) % 2]]
                index_4 = orbital_map[f"I{jj}_" + spin]
                h_sf += FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", 0.5 * j_ex)
                h_sf += openfermion.hermitian_conjugated(
                    FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", 0.5 * j_ex)
                )

    # print("h_sf", h_sf, "\n")

    # Pair-hopping term
    # TODO this is where the problem is if using impurity-centered orbital ordering
    h_ph = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            """
            index_1 = orbital_map[f"I{ii}_" + "up"]
            index_2 = orbital_map[f"I{ii}_" + "down"]
            index_3 = orbital_map[f"I{jj}_" + "up"]
            index_4 = orbital_map[f"I{jj}_" + "down"]
            h_ph += FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", -j_ex)
            h_ph += hermitian_conjugated(FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", -j_ex))
            """
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{ii}_" + spins[(sigma + 1) % 2]]
                index_3 = orbital_map[f"I{jj}_" + spins[(sigma + 1) % 2]]
                index_4 = orbital_map[f"I{jj}_" + spin]
                h_ph += FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", 0.5 * j_ex)
                h_ph += openfermion.hermitian_conjugated(
                    FermionOperator(f"{index_1}^ {index_2}^ {index_3} {index_4}", 0.5 * j_ex)
                )

    # print("h_ph", h_ph, "\n")

    # Diagonal bath
    h_b0 = FermionOperator()
    for ii in range(0, n_bath):
        for spin in spins:
            index_1 = orbital_map[f"B{ii}_" + spin]
            h_b0 += FermionOperator(f"{index_1}^ {index_1}", epsilon_bath)
    # print("h_b0", h_b0, "\n")

    # Hybridization
    h_hyb = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(0, n_bath):
            for spin in spins:
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"B{jj}_" + spin]
                h_hyb += FermionOperator(f"{index_1}^ {index_2}", v[ii, jj])
                h_hyb += openfermion.hermitian_conjugated(
                    FermionOperator(f"{index_1}^ {index_2}", v[ii, jj])
                )
    # print("h_hyb", h_hyb)

    # full_hamiltonian = h_0 + h_u + h_u2j + h_u3j + h_sf + h_ph + h_b0 + h_hyb
    full_hamiltonian = h_0 + h_u + h_u2j + h_u3j + h_ph + h_b0 + h_hyb  # Excludes spin-flip
    # full_hamiltonian = h_0 + h_u + h_u2j + h_u3j + h_sf + h_b0 + h_hyb  # Excludes pair-hopping
    # full_hamiltonian = h_sf

    return full_hamiltonian


def kanamori(n_bath, verbose=0):
    n_imp = 3  # Don't change this for now
    v = np.ones((n_imp, n_bath))
    # hamiltonian = normal_ordered(three_orbital_kanamori_hamiltonian(1, 1, v, 1, 1, n_bath))
    hamiltonian = three_orbital_kanamori_hamiltonian(1, 1, v, 1, 1, n_bath)

    # print("hamiltonian to simulate", "\n", hamiltonian)

    kanamori_interaction_hamiltonian = openfermion.get_interaction_operator(hamiltonian)
    # print("INTERACTION OPERATOR \n", kanamori_interaction_hamiltonian)
    n_qubits = openfermion.count_qubits(kanamori_interaction_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    # final_rank = n_qubits**2
    # custom_algorithm = trotter.LowRankTrotterAlgorithm(final_rank=final_rank)
    circuit = cirq.Circuit(
        simulate_trotter(qubits=qubits, hamiltonian=kanamori_interaction_hamiltonian, time=1)
    )
    #                                         n_steps=1,
    #                                         order=0,
    #                                         algorithm=trotter.LOW_RANK))

    circuit = cirq.drop_negligible_operations(circuit)
    circuit = cirq.drop_negligible_operations(circuit=circuit, atol=1e-7)
    if verbose > 0:
        print("Kanamori Circuit Qubits:", cirq.num_qubits(circuit))
        print("Kanamori Circuit Moments:", len(circuit))
    return circuit


def shor(n, verbose=0):
    while True:
        try:
            qc = factoring_circuit(n)
            break
        except:
            continue
    cirq_circuit = sm.converters.qiskit_to_cirq(qc)
    cirq_circuit = cirq.drop_negligible_operations(cirq_circuit)
    if verbose:
        print(
            f"Factoring: {n}\nQubits: {cirq.num_qubits(cirq_circuit)}\nMoments: {len(cirq_circuit)}"
        )
    return cirq_circuit


# TODO: Finish this
# def report(data):
#     message = (
#             f"Serial Circuit Time (s):\t{serial_time / 1e6:.3e}\n"
#             f"Parallel Ciruit Time (s):\t{parallel_time / 1e6:.3e}\n"
#             f"{'*' * 49}\n"
#     )
#     for name, example in [
#         ("Input Circuit", circuit),
#         ("Clifford+Rz", circuit_compiled),
#         ("Clifford+T", clifford_t_circuit),
#         ("Primitives", primitive_circuit),
#         ("Serial Gates", pretty_serial_cost),
#         ("Parallel Gates", pretty_parallel_cost),
#     ]:
#         gate_dict = (
#             Counter(type(op.gate).__name__ for op in example.all_operations())
#             if isinstance(example, cirq.Circuit)
#             else example
#         )
#         message += f"\33[1m{name}\33[0m\n"
#         for k, v in sorted(gate_dict.items(), key=lambda x: len(x[0])):
#             s = str(k)
#             message += f"\t\33[1m{k}\33[0m{'.' * (25 - len(s))}\33[1m{v:.3e}\33[0m\n"
#         message += "*" * 50
#     return message


# def simple_analysis(
#    circuit: cirq.Circuit,
#    d=7,
#    movement=True,
#    idling=True,
#    post_op_correction=True,
#    cultivation_repetition=1,
#    syndrome_rounds=1,
#    layout: Layout = MovementLayout,
#    eps: float = 1e-6,
# ):
#    if isinstance(layout, MovementLayout) and not movement:
#        print("MovementLayout not compatible with movement=False")
#        return
#
#    architecture = Architecture.from_dict(
#        {
#            "movement": movement,
#            "d": d,
#            "idling": idling,
#            "post_op_correction": post_op_correction,
#            "cultivation_repetition": cultivation_repetition,
#            "syndrome_rounds": syndrome_rounds,
#        }
#    )
#
#    data = {
#        "inputs": {"input_circuit": circuit, "layout": layout, "eps": eps},
#        "outputs": {},
#    }
#
#    device = GenericDevice(cirq.num_qubits(circuit), target="Abraham")
#    print(
#        f"Compiling circuit with width {cirq.num_qubits(circuit)} and depth {len(circuit)} to Clifford + Rz Gateset"
#    )
#    ta = time()
#    circuit_compiled = device.compile(circuit=circuit).circuit
#    tb = time()
#    data["outputs"]["compiled_circuit"] = circuit_compiled
#    data["outputs"]["rz_compile_time"] = tb - ta
#    print(
#        f"Compiled to Clifford + Rz Gateset ({round(data['outputs']['rz_compile_time'], 3)}s)"
#    )
#
#    print(f"Approximating circuit with Clifford + T Gateset (eps={eps})")
#    ta = time()
#    clifford_t_circuit = compile_cirq_to_clifford_t(circuit_compiled, eps=eps)
#    tb = time()
#    data["outputs"]["cliffort_t_circuit"] = clifford_t_circuit
#    data["outputs"]["t_compile_time"] = tb - ta
#    print(
#        f"Approximated to Clifford + T Gateset ({round(data['outputs']['t_compile_time'], 3)}s)"
#    )
#
#    layout = layout(clifford_t_circuit)
#    print("Recompiling to the input architecture")
#    ta = time()
#    primitive_circuit = ft_compile(layout=layout, arc=architecture)
#    tb = time()
#    data["outputs"]["primitive_circuit"] = primitive_circuit
#    data["outputs"]["ft_compile_time"] = tb - ta
#    print(
#        f"Recompiled to input architecture ({round(data['outputs']['ft_compile_time'], 3)}s)"
#    )
#
#    estimator = ResourceEstimator(arc=architecture)
#    print("Getting serial cost")
#    ta = time()
#    circuit_cost = estimator.circuit_cost(circuit=primitive_circuit, pretty=False)
#    tb = time()
#    data["outputs"]["circuit_cost_time"] = tb - ta
#    print("Getting parallel cost")
#    ta = time()
#    parallel_circuit_cost = estimator.parallel_circuit_cost(
#        primitive_circuit, pretty=False
#    )
#    tb = time()
#    data["outputs"]["circuit_moment_time"] = tb - ta
#    pretty_serial_cost = {
#        obj.__name__ if hasattr(obj, "__name__") else str(obj): val
#        for obj, val in circuit_cost.items()
#    }
#    data["outputs"]["serial_cost"] = pretty_serial_cost
#    pretty_parallel_cost = {
#        obj.__name__ if hasattr(obj, "__name__") else str(obj): val
#        for obj, val in parallel_circuit_cost.items()
#    }
#    data["outputs"]["parallel_cost"] = pretty_parallel_cost
#    print()
#    print(f"Gate{' ' * 21}Serial\t\tParallel")
#    for k in set(pretty_parallel_cost).union(pretty_serial_cost):
#        chars = len(k)
#        print(
#            f"{k}{' ' * (25 - chars)}{pretty_serial_cost.get(k, 0):.3e}\t{pretty_parallel_cost.get(k, 0):.3e}"
#        )
#    ta = time()
#    serial_time = architecture.total_time(circuit_cost)
#    tb = time()
#    data["outputs"]["serial_time_time"] = tb - ta
#    ta = time()
#    parallel_time = architecture.total_time(parallel_circuit_cost)
#    tb = time()
#    data["outputs"]["parallel_time"] = parallel_time
#    data["outputs"]["parallel_time_time"] = tb - ta
#    print("*" * 49)
#    print(f"Serial Circuit Time (s):\t{serial_time / 1e6:.3e}")
#    print(f"Parallel Ciruit Time (s):\t{parallel_time / 1e6:.3e}")
#    print("*" * 49)
#    for name, example in [
#        ("Input Circuit", circuit),
#        ("Clifford+Rz", circuit_compiled),
#        ("Clifford+T", clifford_t_circuit),
#        ("Primitives", primitive_circuit),
#        ("Serial Gates", pretty_serial_cost),
#        ("Parallel Gates", pretty_parallel_cost),
#    ]:
#        gate_dict = (
#            Counter(type(op.gate).__name__ for op in example.all_operations())
#            if isinstance(example, cirq.Circuit)
#            else example
#        )
#        print(f"\33[1m{name}\33[0m")
#        for k, v in sorted(gate_dict.items(), key=lambda x: len(x[0])):
#            s = str(k)
#            print(f"\t\33[1m{k}\33[0m{'.' * (25 - len(s))}\33[1m{v:.3e}\33[0m")
#        print("*" * 50)
#    return data


def show_all():
    print(fermi_hubbard(3, 1))
    print(kanamori(9, 1))


if __name__ == "__main__":
    from resource_estimation.estimate import ResourceEstimator
    from resource_estimation.layout import MovementLayout, ColumnLayout
    from resource_estimation.architecture import DefaultMovement, DefaultLattice
    from resource_estimation.compile_ftqc import ft_compile
    from resource_estimation.cliff_rz import GenericDevice
    from resource_estimation.clifford_t import compile_cirq_to_clifford_t
    import warnings
    import pickle
    from time import time

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tstart = time()
    shor_circuit = shor(3, 1)
    tfin = time()
    print(tfin - tstart)
    # ham_circuit = fermi_hubbard(7, 1)
    exit()

    input_circuit = ham_circuit
    ta = time()
    analysis = simple_analysis(
        circuit=input_circuit,
        d=7,
        movement=True,
        idling=False,
        eps=1e-6,
        post_op_correction=False,
    )
    # analysis = simple_analysis(ham_circuit, movement=False, layout=ColumnLayout)
    tb = time()
    print(tb - ta)
    # with open("data/example1.pkl", "wb") as f:
    #     pickle.dump(analysis, f)

    # kan_circuit = kanamori(3, 0)
    # analysis = simple_analysis(kan_circuit)
    # with open("data/example2.pkl", "wb") as f:
    #     pickle.dump(analysis, f)

    # shor_circuit = make_order_finding_circuit(17, 5)
    # device = GenericDevice(cirq.num_qubits(shor_circuit), target='Joseph')
    # shor_circuit_compiled = device.compile(shor_circuit).circuit
    # print(Counter(type(op.gate).__name__ for op in shor_circuit_compiled.all_operations()))
