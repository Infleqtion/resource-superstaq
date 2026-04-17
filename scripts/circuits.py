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

    # U density-density term
    h_u = FermionOperator()
    for ii in range(0, n_imp):
        index_1 = orbital_map[f"I{ii}_up"]
        index_2 = orbital_map[f"I{ii}_down"]
        h_u += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u)

    # U-2J density-density term
    h_u2j = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{jj}_" + spins[(sigma + 1) % 2]]
                h_u2j += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u - 2 * j_ex)

    # U-3J density-density term
    h_u3j = FermionOperator()
    for ii in range(0, n_imp):
        for jj in range(ii + 1, n_imp):
            for sigma, spin in enumerate(spins):
                index_1 = orbital_map[f"I{ii}_" + spin]
                index_2 = orbital_map[f"I{jj}_" + spin]
                h_u3j += FermionOperator(f"{index_1}^ {index_1} {index_2}^ {index_2}", u - 3 * j_ex)

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

    full_hamiltonian = h_0 + h_u + h_u2j + h_u3j + h_ph + h_b0 + h_hyb  # Excludes spin-flip

    return full_hamiltonian


def kanamori(n_bath, verbose=0):
    n_imp = 3  # Don't change this for now
    v = np.ones((n_imp, n_bath))
    hamiltonian = three_orbital_kanamori_hamiltonian(1, 1, v, 1, 1, n_bath)

    kanamori_interaction_hamiltonian = openfermion.get_interaction_operator(hamiltonian)
    n_qubits = openfermion.count_qubits(kanamori_interaction_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    circuit = cirq.Circuit(
        simulate_trotter(qubits=qubits, hamiltonian=kanamori_interaction_hamiltonian, time=1)
    )

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
