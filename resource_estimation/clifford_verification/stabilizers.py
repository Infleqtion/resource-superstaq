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
"""Partial trace and canonical comparison of signed stabilizer groups."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import cirq

# Python 3.9 remains supported; newer Python uses the native integer operation.
_popcount = getattr(int, "bit_count", lambda value: bin(value).count("1"))

# A Pauli is i**phase X**x Z**z, with Python integer masks.
Pauli = tuple[int, int, int]
Signature = tuple[int, tuple[Pauli, ...]]

# Small multiply helper, gives reasonable speedup compared to Cirq.
def _multiply(left: Pauli, right: Pauli) -> Pauli:
    x, z, phase = left
    a, b, other = right
    return x ^ a, z ^ b, (phase + other + 2 * _popcount(z & a)) % 4

# Reimplementation of row reduction using our _multiply + return rank to identify surviving generator locations
def _eliminate(rows: list[Pauli], columns: Sequence[tuple[int, int]]) -> int:
    """Signed RREF, with supplied columns interpreted as (X/Z component, bit)."""
    rank = 0
    for component, bit in columns:
        pivot = next((i for i in range(rank, len(rows)) if rows[i][component] >> bit & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][component] >> bit & 1:
                rows[i] = _multiply(rows[i], rows[rank])
        rank += 1
    return rank

# Extract stabilizers from a tableau and reduce it by eliminating discarded qubits and reducing/canoniclizing the resulting stabilizer state
def reduced_signature(tableau: cirq.CliffordTableau, keep: Sequence[int]) -> Signature:
    """Canonical mixed stabilizer state on ordered retained qubits.

    Eliminate environment X/Z columns first. The remaining rows generate the
    kernel of restriction to the environment, including products whose discarded
    support cancels. Simply deleting generators touching the environment is wrong.
    """
    # Number of qubits in the tableau
    n = tableau.n
    # Index check
    if len(set(keep)) != len(keep) or any(j < 0 or j >= n for j in keep):
        raise ValueError("Invalid retained tableau indices")

    rows = []
    # Read out the stabilizers from the tableau into rows
    for i in range(n, 2 * n):
        x = sum(1 << j for j in range(n) if tableau.xs[i, j])
        z = sum(1 << j for j in range(n) if tableau.zs[i, j])
        rows.append((x, z, (2 * int(tableau.rs[i]) + _popcount(x & z)) % 4))
    environment = [j for j in range(n) if j not in keep]

    rank = _eliminate(rows, [(component, j) for component in (0, 1) for j in environment])
    retained = []
    for x, z, phase in rows[rank:]:
        retained.append(
            (
                sum(((x >> old) & 1) << new for new, old in enumerate(keep)),
                sum(((z >> old) & 1) << new for new, old in enumerate(keep)),
                phase,
            )
        )
    rank = _eliminate(retained, [(component, j) for component in (0, 1) for j in range(len(keep))])
    assert rank == len(retained)
    return len(keep), tuple(retained)


def stabilizer_overlap(left: Signature, right: Signature) -> Fraction:
    """Exact Tr(rho_left rho_right), without expanding either stabilizer group."""
    n, a = left
    if n != right[0]:
        raise ValueError("Channel boundary sizes must match")
    b = right[1]
    # Find dependencies of unsigned generators while retaining their provenance.
    # Multiply within each commuting group separately: the union need not commute.
    basis: dict[int, tuple[int, int]] = {}
    for j, (x, z, _) in enumerate(a + b):
        vector, combination = x | (z << n), 1 << j
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector, combination
                break
            previous, provenance = basis[pivot]
            vector ^= previous
            combination ^= provenance
        if not vector:
            pa = pb = (0, 0, 0)
            for k, generator in enumerate(a):
                if combination >> k & 1:
                    pa = _multiply(pa, generator)
            for k, generator in enumerate(b, len(a)):
                if combination >> k & 1:
                    pb = _multiply(pb, generator)
            if pa != pb:
                return Fraction(0)
    return Fraction(2) ** (len(a) + len(b) - len(basis) - n)


def channel_mixtures_equal(
    left: Sequence[tuple[Fraction, Signature]], right: Sequence[tuple[Fraction, Signature]]
) -> bool:
    """Compare density operators via their exact squared Hilbert-Schmidt distance.

    Mixture decompositions are not unique. Comparing branch lists would reject
    equal channels; pairwise stabilizer overlaps compare their actual sums.
    """
    if len(left) == len(right) == 1:
        return left[0] == right[0]
    if left == right:
        return True
    terms = list(left) + [(-weight, state) for weight, state in right]
    distance = sum(
        (wa * wb * stabilizer_overlap(a, b) for wa, a in terms for wb, b in terms),
        Fraction(0),
    )
    return distance == 0
