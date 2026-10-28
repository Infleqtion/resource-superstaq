from __future__ import annotations

import dataclasses
import functools

import cirq
import numpy as np


@dataclasses.dataclass(frozen=True)
class CNOTMergeTag:
    """Metadata identifying CNOTs whose same-basis spiders can be merged.

    Attributes:
        group_id: Circuit-local identifier shared by all CNOTs in the merge group.
        basis: Whether the group merges target-side X spiders or control-side Z spiders.
    """

    group_id: int
    basis: cirq.Pauli


@dataclasses.dataclass(eq=False)
class OperationRecord:
    operation: cirq.Operation
    cnot_id: int | None = None


@dataclasses.dataclass(frozen=True)
class CNOTCorrections:
    """Tracks corrections associated with pushing single qubit cliffords forward through CNOTs."""

    cnot: OperationRecord
    pre_other: cirq.SingleQubitCliffordGate
    post_other: cirq.SingleQubitCliffordGate


@dataclasses.dataclass(frozen=True)
class MergeCandidate:
    basis: cirq.Pauli
    shared_qubit: cirq.Qid
    members: tuple[OperationRecord, ...]
    cnot_corrections: tuple[CNOTCorrections, ...]
    moved_cliffords: tuple[OperationRecord, ...]


def _cnot_qubits(operation: cirq.Operation) -> tuple[cirq.Qid, cirq.Qid] | None:
    untagged = operation.untagged
    if isinstance(untagged, cirq.GateOperation) and untagged.gate == cirq.CNOT:
        return untagged.qubits[0], untagged.qubits[1]
    return None


def _is_movable_clifford(operation: cirq.Operation) -> bool:
    return (
        len(operation.qubits) == 1
        and operation.gate is not None
        and not operation.tags
        and cirq.has_unitary(operation)
        and cirq.has_stabilizer_effect(operation)
    )


def _combined_clifford(records: list[OperationRecord]) -> cirq.SingleQubitCliffordGate:
    matrix = np.eye(2, dtype=np.complex128)
    for record in records:
        matrix = cirq.unitary(record.operation) @ matrix
    clifford = cirq.SingleQubitCliffordGate.from_unitary(matrix)
    assert clifford is not None
    return clifford


def _two_qubit_unitary(operation: cirq.Operation) -> np.ndarray:
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(operation).unitary(qubit_order=[q0, q1])


@functools.cache
def _local_corrections(
    frame: cirq.SingleQubitCliffordGate,
    shared_is_control: bool,
    desired_basis: cirq.Pauli,
) -> tuple[cirq.SingleQubitCliffordGate, cirq.SingleQubitCliffordGate] | None:
    """Brute force search to find local corrections that move a Clifford frame through one CNOT.

    Matrices use ``shared, other`` qubit order. The returned gates act on the other qubit before
    and after the rewritten CNOT, while ``frame`` itself moves to the shared-qubit output.
    """
    q0, q1 = cirq.LineQubit.range(2)
    original = cirq.CNOT(q0, q1) if shared_is_control else cirq.CNOT(q1, q0)
    rewritten = cirq.CNOT(q0, q1) if desired_basis is cirq.Z else cirq.CNOT(q1, q0)
    frame_matrix = cirq.unitary(frame)
    identity = np.eye(2)
    target = _two_qubit_unitary(original) @ np.kron(frame_matrix, identity)
    rewritten_matrix = _two_qubit_unitary(rewritten)

    for pre_other in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords:
        for post_other in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords:
            candidate = (
                np.kron(frame_matrix, cirq.unitary(post_other))
                @ rewritten_matrix
                @ np.kron(identity, cirq.unitary(pre_other))
            )
            if cirq.linalg.allclose_up_to_global_phase(target, candidate, atol=1e-8):
                return pre_other, post_other
    return None


def _find_candidate(
    records: list[OperationRecord],
    seed: OperationRecord,
    basis: cirq.Pauli,
    assigned_cnot_ids: set[int],
    max_weight: int,
) -> MergeCandidate:
    # Scans forward and identifies if a CNOT can be merged with the seed operation

    seed_qubits = _cnot_qubits(seed.operation)
    assert seed_qubits is not None
    shared_qubit = seed_qubits[0] if basis is cirq.Z else seed_qubits[1]
    seed_index = records.index(seed)
    members = [seed]
    cnot_corrections: list[CNOTCorrections] = []
    frame_records: list[OperationRecord] = []
    committed_frame_records: tuple[OperationRecord, ...] = ()

    for record in records[seed_index + 1 :]:
        operation = record.operation
        if shared_qubit not in operation.qubits:
            continue

        if _is_movable_clifford(operation):
            frame_records.append(record)
            continue

        cnot_qubits = _cnot_qubits(operation)
        if cnot_qubits is None or record.cnot_id in assigned_cnot_ids:
            break

        frame = _combined_clifford(frame_records)
        corrections = _local_corrections(
            frame,
            shared_is_control=cnot_qubits[0] == shared_qubit,
            desired_basis=basis,
        )
        if corrections is None:
            break

        members.append(record)
        cnot_corrections.append(CNOTCorrections(record, *corrections))
        committed_frame_records = tuple(frame_records)

        if max_weight > 0 and len(members) == max_weight:
            break

    return MergeCandidate(
        basis=basis,
        shared_qubit=shared_qubit,
        members=tuple(members),
        cnot_corrections=tuple(cnot_corrections),
        moved_cliffords=committed_frame_records,
    )


def _surrounding_group_basis(
    records: list[OperationRecord],
    seed: OperationRecord,
    groups: dict[int, tuple[cirq.Pauli, set[int]]],
) -> cirq.Pauli | None:
    # Identify if a seed CNOT is surounded by an existing CNOT merge group

    seed_index = records.index(seed)
    seed_qubits = set(seed.operation.qubits)
    positions = {
        record.cnot_id: index for index, record in enumerate(records) if record.cnot_id is not None
    }
    surrounding: list[tuple[int, int, cirq.Pauli]] = []

    for group_id, (basis, cnot_ids) in groups.items():
        left_distances = [
            seed_index - positions[cnot_id]
            for cnot_id in cnot_ids
            if positions[cnot_id] < seed_index
            and seed_qubits.intersection(records[positions[cnot_id]].operation.qubits)
        ]
        right_distances = [
            positions[cnot_id] - seed_index
            for cnot_id in cnot_ids
            if positions[cnot_id] > seed_index
            and seed_qubits.intersection(records[positions[cnot_id]].operation.qubits)
        ]
        if left_distances and right_distances:
            surrounding.append((min(left_distances) + min(right_distances), group_id, basis))

    if not surrounding:
        return None

    # Select closest surrounding group (in case > 1 nested groups)
    return min(surrounding)[2]


def _apply_candidate(
    records: list[OperationRecord], candidate: MergeCandidate
) -> list[OperationRecord]:
    cnot_corrections = {
        cnot_correction.cnot: cnot_correction for cnot_correction in candidate.cnot_corrections
    }
    moved_cliffords = set(candidate.moved_cliffords)
    last_member = candidate.members[-1]
    rewritten_records: list[OperationRecord] = []

    for record in records:
        if record in moved_cliffords:
            continue

        cnot_correction = cnot_corrections.get(record)
        if cnot_correction is None:
            rewritten_records.append(record)
            continue

        cnot_qubits = _cnot_qubits(record.operation)
        assert cnot_qubits is not None
        shared_qubit = candidate.shared_qubit
        if candidate.basis is cirq.Z:
            other_qubit = next(qubit for qubit in cnot_qubits if qubit != shared_qubit)
            rewritten_cnot = cirq.CNOT(shared_qubit, other_qubit)
        else:
            other_qubit = next(qubit for qubit in cnot_qubits if qubit != shared_qubit)
            rewritten_cnot = cirq.CNOT(other_qubit, shared_qubit)

        if cnot_correction.pre_other is not cirq.SingleQubitCliffordGate.I:
            rewritten_records.append(OperationRecord(cnot_correction.pre_other(other_qubit)))
        rewritten_records.append(
            OperationRecord(rewritten_cnot.with_tags(*record.operation.tags), record.cnot_id)
        )

        if cnot_correction.post_other is not cirq.SingleQubitCliffordGate.I:
            rewritten_records.append(OperationRecord(cnot_correction.post_other(other_qubit)))
        if record is last_member:
            rewritten_records.extend(
                OperationRecord(moved.operation) for moved in candidate.moved_cliffords
            )

    return rewritten_records


def identify_cnot_merge_groups(circuit: cirq.AbstractCircuit, max_weight: int = 0) -> cirq.Circuit:
    """
    Identifies and tags opportunities to merge CNOT gates with shared controls or targets.
    For any CNOT, there may be opportunities to merge its control or a target, but we can only choose one for PPM compiling.
    Addressing this choice is done using the following heuristic:
    1) If a CNOT is surrounded by a merged group choose the same basis
       (this enables commutation between PPMs to aid in the future scheduling pass)
    2) Otherwise choose the option that maximizes the number of merged CNOTs

    For each unassigned CNOT, the pass scans forward independently from its control-side Z spider
    and target-side X spider. Untagged single-qubit Clifford gates may be pushed through later
    CNOTs when an exact local-Clifford rewrite makes the encountered spider match the seed basis.
    Every CNOT belongs to exactly one group and therefore has only one selected merge side.

    Args:
        circuit: Clifford circuit whose CNOT spiders should be grouped.
        max_weight: Max weight (number of cnots + 1) of merged cnot group.
                    If 0 attempst to create as large of groups as possible

    Returns:
        An equivalent circuit whose CNOTs carry :class:`CNOTMergeTag` metadata. Single-qubit
        Clifford gates are moved only when required to expose a selected merge.
    """
    records: list[OperationRecord] = []
    next_cnot_id = 0
    for moment in circuit:
        for operation in moment.operations:
            retained_tags = tuple(
                tag for tag in operation.tags if not isinstance(tag, CNOTMergeTag)
            )
            normalized_operation = (
                operation.untagged.with_tags(*retained_tags)
                if len(retained_tags) != len(operation.tags)
                else operation
            )
            cnot_id = next_cnot_id if _cnot_qubits(normalized_operation) is not None else None
            records.append(OperationRecord(normalized_operation, cnot_id))
            if cnot_id is not None:
                next_cnot_id += 1

    assigned_cnot_ids: set[int] = set()
    groups: dict[int, tuple[cirq.Pauli, set[int]]] = {}
    group_id = 0

    while len(assigned_cnot_ids) < next_cnot_id:
        seed = next(
            record
            for record in records
            if record.cnot_id is not None and record.cnot_id not in assigned_cnot_ids
        )
        candidates = {
            basis: _find_candidate(records, seed, basis, assigned_cnot_ids, max_weight)
            for basis in [cirq.Z, cirq.X]
        }

        # Use heuristic as defined above to select merged control or target qubit
        preferred_basis = _surrounding_group_basis(records, seed, groups)
        if preferred_basis is None:
            preferred_basis = max(
                (cirq.Z, cirq.X),
                key=lambda basis: len(candidates[basis].members),
            )

        selected = candidates[preferred_basis]
        selected_ids = {member.cnot_id for member in selected.members if member.cnot_id is not None}
        assigned_cnot_ids.update(selected_ids)
        groups[group_id] = selected.basis, selected_ids
        records = _apply_candidate(records, selected)
        group_id += 1

    for record in records:
        if record.cnot_id is None:
            continue
        matching_group = next(
            (identifier, basis)
            for identifier, (basis, cnot_ids) in groups.items()
            if record.cnot_id in cnot_ids
        )
        record.operation = record.operation.with_tags(CNOTMergeTag(*matching_group))

    return cirq.Circuit(record.operation for record in records)
