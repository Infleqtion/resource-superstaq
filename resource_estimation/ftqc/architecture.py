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

import abc
import collections
from copy import copy
from functools import cached_property, lru_cache
from math import ceil
from random import randint
from typing import Callable, Literal

import cirq
import cirq_superstaq as css
import numpy as np

import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.distil import precompute_distil_cost
from resource_estimation.ftqc.stim_functions import cultivate
from resource_estimation.typing import CostDict, GateCounts, GateKey, _require_gate_operation

NEUTRAL_GATES: dict[GateKey, float] = {  # From Harvard paper (https://arxiv.org/pdf/2506.20661)
    cirq.CZ: 0.27,
    cirq.PhasedXZGate: 5.0,  # Based on single qubit gate times
    cirq.ResetChannel: 400,  # A few hundred us
    cirq.MeasurementGate: 1000,  # Best guess from 500us for atom movement during readout
    css.MovementGate: 500,
    cirq.CCZ: 0.27,
}

SUPERCOND_GATES: dict[GateKey, float] = {
    # Times are in microseconds
    cirq.PhasedXZGate: 0.020,  # 20ns Used to represent all single qubit gates
    cirq.CZ: 0.040,  # 40ns  These both come from https://web.physics.ucsb.edu/~martinisgroup/papers/Barends2014.pdf (page 5)
    cirq.ResetChannel: 1,  # Based on 1us cycle time assumed by https://arxiv.org/pdf/2505.15917 (Gidney RSA 2025)
    cirq.MeasurementGate: 0.5,  # https://arxiv.org/abs/2308.02079
}


@lru_cache(maxsize=128)
def _merge_cost(
    d: int,
    k: int,
    smooth: bool,
) -> CostDict:
    """Calculates the resources required to implement the merge operation.
    d is the code distance, k is the number of patches being merged, and smooth indicates if patches are being joined at the Z or X boundary
    """
    assert k >= 2

    endpoints = 2
    buffers = k - 1
    intermediates = k - 2

    end_patch = lsp.EndpointPatch(d=d, smooth=smooth)
    buff_patch = lsp.BufferCodePatch(d=d, smooth=smooth)
    inter_patch = lsp.IntermediatePatch(d=d, smooth=smooth)

    full_z_stabs = (
        endpoints * end_patch.num_z_stabs(full=True)
        + buffers * buff_patch.num_z_stabs(full=True)
        + intermediates * inter_patch.num_z_stabs(full=True)
    )
    full_x_stabs = (
        endpoints * end_patch.num_x_stabs(full=True)
        + buffers * buff_patch.num_x_stabs(full=True)
        + intermediates * inter_patch.num_x_stabs(full=True)
    )

    partial_z_stabs = (
        endpoints * end_patch.num_z_stabs(full=False)
        + buffers * buff_patch.num_z_stabs(full=False)
        + intermediates * inter_patch.num_z_stabs(full=False)
    )

    partial_x_stabs = (
        endpoints * end_patch.num_x_stabs(full=False)
        + buffers * buff_patch.num_x_stabs(full=False)
        + intermediates * inter_patch.num_x_stabs(full=False)
    )

    cz_gates = d * (2 * (partial_x_stabs + partial_z_stabs) + 4 * (full_x_stabs + full_z_stabs))
    rz_gates = d * (8 * full_x_stabs + 2 * full_z_stabs + 6 * partial_x_stabs + 2 * partial_z_stabs)
    measures = d * (full_z_stabs + full_x_stabs + partial_z_stabs + partial_x_stabs)
    resets = measures

    gate_cost: GateCounts = {
        cirq.CZ: cz_gates,
        cirq.PhasedXZGate: rz_gates,
        cirq.MeasurementGate: measures,
        cirq.ResetChannel: resets,
    }
    moment_cost: GateCounts = {
        cirq.CZ: 4 * d,
        cirq.PhasedXZGate: 2 * d,
        cirq.MeasurementGate: d,
        cirq.ResetChannel: d,
    }
    return CostDict(gate_cost=gate_cost, moment_cost=moment_cost, op_time=-1)


def _syndrome_extract_cost(
    rounds: int,
    num_logical_qubits: int,
    d: int,
) -> CostDict:
    """Calculates the cost of syndrome extraction in terms of physical gates"""
    # This is how SE should look...
    # ...for a (full) X stabilizer
    # RESET H CZ CZ CZ CZ H MEASURE  <--Measure Qubit
    #       H CZ  |  |  | H
    #       H    CZ  |  | H
    #       H       CZ  | H
    #       H          CZ H
    # ...for a (full) Z stabilizer
    # RESET H CZ CZ CZ CZ H MEASURE  <--Measure Qubit
    #         CZ  |  |  |
    #            CZ  |  |
    #               CZ  |
    #                  CZ
    patch = lsp.RotatedCodePatch(d)
    gate_cost: GateCounts = {
        cirq.MeasurementGate: patch.num_measure_qubits * num_logical_qubits * rounds,
        cirq.CZ: (patch.total_z_syndrome_cnots() + patch.total_x_syndrome_cnots())
        * num_logical_qubits
        * rounds,
        cirq.ResetChannel: patch.num_measure_qubits * num_logical_qubits * rounds,
        cirq.PhasedXZGate: num_logical_qubits
        * rounds
        * (
            (10 * patch.num_x_stabs(full=True))  # 5 Hadamards on left and 5 Hadamards on right
            + (2 * patch.num_z_stabs(full=True))  # 1 Hadamard on left and 1 Hadamard on right
            + (6 * patch.num_x_stabs(full=False))  # 3 Hadamards on left and 3 Hadamards on right
            + (2 * patch.num_z_stabs(full=False))  # 1 Hadamard on left and 1 Hadamard on right
        ),
    }
    moment_cost: GateCounts = {
        cirq.MeasurementGate: rounds,
        cirq.ResetChannel: rounds,
        cirq.CZ: 4 * rounds,  # 4 per stabilizer type
        cirq.PhasedXZGate: 2 * rounds,  # 2 per stabilizer type
    }
    return CostDict(gate_cost=gate_cost, moment_cost=moment_cost, op_time=-1)


@lru_cache(maxsize=128)
def _split_cost(smooth: bool, d: int) -> CostDict:
    """Calculates cost to perform a split operation
    Split operations can always be absorbed into a following moment, so the moment cost is null
    """
    if smooth:
        # Measuring in the X-basis costs an extra basis change
        gate_cost: GateCounts = {
            cirq.MeasurementGate: 2 * d + 1,
            cirq.PhasedXZGate: ceil((2 * d + 1) / 2),
        }
    else:
        gate_cost = {
            cirq.MeasurementGate: 2 * d + 1,
        }
    return CostDict(gate_cost=gate_cost, moment_cost={}, op_time=-1)


def _physical_move_time(l: float, a: float = 5500, base_cost: float = 200) -> float:
    """
    Calculates total time to travel a distance given a constant accelleration profile in μs
    l: physical distance in μm
    a: acceleration in  m/s^2
    startup_cost: flat cost every move pays in μs
    """
    l *= 10**-6  # convert μm to m
    # a is in m/s^2, so we make sure to convert answer to μs
    return 2 * np.sqrt(l / a) * 10**6 + base_cost


def _measurement_zone_move_precompiled(dx: int, dy: int, patch_length: int, site_spacing: float) -> CostDict:
    # A logical Move to a measurement zone
    #   - RShift by one site
    #   - Move measure qubits to zone
    # Current notation denotes the move operation between the logical qubit and the zone itself as arguments
    # Reversing the sequence of moves achives the inverse and has the same cost
    l1 = 1 * site_spacing
    t1 = _physical_move_time(l1)

    l2_x = dx * patch_length * site_spacing
    l2_y = dy * patch_length * site_spacing
    t2 = _physical_move_time(l2_x) + _physical_move_time(l2_y)

    op_time = t1 + t2
    return CostDict(gate_cost={css.MovementGate: 2}, moment_cost={css.MovementGate: 2}, op_time=op_time)


def _interaction_zone_move_precompiled(dx: int, dy: int, patch_length: int, site_spacing: float) -> CostDict:
    # A logical Move to an interaction zone
    #   - RShift by one site
    #   - Move data qubits to zone
    #   - Squeeze zone qubits
    # Current notation denotes an operation between a logical qubit and the zone itself as arguments
    # Therefore compiled circuits see two logical movement operations to prepare one logical CNOT
    # Reversing the sequence of moves achieves the inverse and has the same cost
    l1 = 1 * site_spacing
    t1 = _physical_move_time(l1)

    l2_x = dx * patch_length * site_spacing
    l2_y = dy * patch_length * site_spacing
    t2 = _physical_move_time(l2_x) + _physical_move_time(l2_y)

    # Couldn't we actually just do this in the same move as l2_x?
    l3 = 0.25 * site_spacing  # Squeeze sites for interaction
    t3 = _physical_move_time(l3)
    op_time = t1 + t2 + t3
    return CostDict(gate_cost={css.MovementGate: 3},moment_cost={css.MovementGate: 3}, op_time=op_time)


def _inplace_entanglement_move_precompiled(
    dx: int, dy: int, patch_length: int, site_spacing: float, scratch_dx: int, scratch_dy: int
) -> CostDict:
    # A logical CNOT operation performed inplace using movement
    #   - RShift by one unit to get alternating columns of data and ancilla qubits on control and target patches
    #   - Send measure qubits of the target patch to the corner of the array
    #   - Move data qubits from the control patch to the now free space in the target patch
    # Reversing the sequence of moves achieves the inverse and has the same cost

    # Shift -- align columns
    l1 = 1 * site_spacing
    t1 = _physical_move_time(l1)

    # Punt -- Move measure qubits to logical corner
    l2_x = patch_length * scratch_dx * site_spacing
    l2_y = patch_length * scratch_dy * site_spacing
    t2 = _physical_move_time(l2_x) + _physical_move_time(l2_y)

    # Interact -- Move datas from ctrl to trgt
    l3_x = patch_length * dx * site_spacing
    l3_y = patch_length * dy * site_spacing
    t3 = _physical_move_time(l3_x) + _physical_move_time(l3_y)

    op_time = t1 + t2 + t3
    return CostDict(op_time=op_time, gate_cost={css.MovementGate: 3}, moment_cost={css.MovementGate: 3})


class Architecture(abc.ABC):
    """Class for representing device architectures.

    Generally, only subclasses of this class should be used.
    Comes preloaded with many primitive costs that are shared among the current set of subclasses.
    """

    def __init__(
        self,
        idling: bool,
        post_op_correction: bool,
        movement: bool,
        d: int = 7,
        cultivation_repetition: int = 1,
        cultivation_fault_distance: Literal[3, 5] = 3,
        syndrome_rounds: int | None = None,
        fold_cultiv: bool = False,
    ) -> None:
        self.idling: bool = idling
        self.post_op_correction: bool = post_op_correction
        self.movement: bool = movement
        self.d: int = d
        self.patch: lsp.RotatedCodePatch = lsp.RotatedCodePatch(self.d)
        self.cultivation_repetition: int = cultivation_repetition
        self.cultivation_fault_distance: Literal[3, 5] = cultivation_fault_distance
        self.syndrome_rounds: int | None = syndrome_rounds
        self.fold_cultiv: bool = fold_cultiv

        self._primitives: cirq.Gateset = cirq.Gateset()
        self._phys_gate_times: dict[GateKey, float]
        self.__post_init__()

    ### Fundamental Cost Counting Methods ###
    # These should never be overwritten
    def gate_cost(self, op: cirq.Operation, **kwargs) -> GateCounts:
        gate_op = _require_gate_operation(op=op)
        try:
            return self.op_cost[type(gate_op.gate)](gate_op, **kwargs).gate_cost
        except KeyError:
            raise ValueError("Gate not recognized")

    def op_time(self, op: cirq.Operation, **kwargs) -> float:
        gate_op = _require_gate_operation(op=op)
        try:
            return self.op_cost[type(gate_op.gate)](gate_op, **kwargs).op_time
        except KeyError:
            raise ValueError("Gate not recognized")

    def moment_cost(self, op: cirq.Operation, **kwargs) -> GateCounts:
        gate_op = _require_gate_operation(op=op)
        try:
            return self.op_cost[type(gate_op.gate)](gate_op, **kwargs).moment_cost
        except KeyError:
            raise ValueError("Gate not recognized")

    def total_time(self, moment_cost_dict: GateCounts) -> float:
        return sum(
            num_ops * self.phys_gate_times[phys_op] for phys_op, num_ops in moment_cost_dict.items()
        )

    ### Properties ###
    # These are particular costs that are often repeated
    # In the future, they could be the counts of a NISQ compiler on a single Primitive
    # Some are cached because they can be expensive to call repeatedly

    # Since there are many ways to generate T states, each Architecture subclass must specify their particular method
    @cached_property
    def _cultivate_t_cost(self) -> CostDict:  # pragma: no cover
        raise NotImplementedError

    def _distil_cost(self, resource: Literal["T", "CCZ"]) -> CostDict:
        raise NotImplementedError(
            "Distillation is currently reserved to distillation movement architectures only"
        )

    @cached_property
    def _h_cost(self) -> CostDict:  # pragma: no cover
        raise NotImplementedError

    @cached_property
    def _cultivate_y_cost(self) -> CostDict:
        """Cost estimate for measuring a surface code patch in the Y basis. Measuring in the Y basis facilitates gate teleportation the same way that cultivating T does.
        The procedure is based on [Inplace access to the Surface Code Y Basis](https://arxiv.org/pdf/2302.07395v2).
        The cost estimates were generated by looking carefully at circuits produced from https://doi.org/10.5281/zenodo.7487893.
        """
        single_qubit_moments = 5
        reset_moments = 1
        cz_moments = 5
        measure_moments = 1
        Y_moment_cost: collections.Counter[GateKey] = collections.Counter(
            {
                cirq.PhasedXZGate: single_qubit_moments,
                cirq.CZ: cz_moments,
                cirq.MeasurementGate: measure_moments,
                cirq.ResetChannel: reset_moments,
            },
        )
        se_moment_cost = collections.Counter(
            _syndrome_extract_cost(
                rounds=ceil(self.d / 2), num_logical_qubits=1, d=self.d
            ).moment_cost,
        )

        # TODO: Perhaps cannonical cost includes SE before and afer for a total of two more units of SE
        moment_cost = se_moment_cost + Y_moment_cost + Y_moment_cost
        op_time = self.total_time(moment_cost_dict=moment_cost)

        # For the gate cost, let's just approximate it with one round of syndrome extraction with an additional d-1 diagonal of CZ gates
        se_gate_cost = collections.Counter(
            _syndrome_extract_cost(
                rounds=ceil(self.d / 2), num_logical_qubits=1, d=self.d
            ).gate_cost,
        )
        Y_gate_cost = se_gate_cost.copy()
        Y_gate_cost[cirq.CZ] += self.d - 1
        gate_cost = se_gate_cost + Y_gate_cost + Y_gate_cost

        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _x_cost(self) -> CostDict:
        return CostDict(op_time=0, moment_cost={}, gate_cost={})

    @cached_property
    def _z_cost(self) -> CostDict:
        return CostDict(op_time=0, moment_cost={}, gate_cost={})

    @cached_property
    def _i_cost(self) -> CostDict:
        return CostDict(op_time=0, moment_cost={}, gate_cost={})

    @cached_property
    def _measure_cost(self) -> CostDict:
        gate_cost: GateCounts = {cirq.MeasurementGate: self.patch.num_data_qubits}
        moment_cost: GateCounts = {cirq.MeasurementGate: 1}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @property
    def rounds(self) -> int:
        if self.syndrome_rounds is None:
            return self.d
        return self.syndrome_rounds

    @property
    def phys_gate_times(self) -> dict[GateKey, float]:
        return self._phys_gate_times

    @property
    @abc.abstractmethod
    def __name__(self) -> str:  # pragma: no cover
        pass

    @property
    def primitives(self) -> cirq.Gateset:
        return self._primitives

    zone_ops: cirq.Gateset = cirq.Gateset()

    alley_ops: cirq.Gateset = cirq.Gateset()

    ### Top Level Cost Methods ###
    # Functions used to interpret the costs of Primitives in the form of cirq operations
    # The ones here are common among all architectures currently
    def cultivate_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        assert isinstance(op.gate, lsp.Cultivate)
        theta = op.gate.theta
        if np.isclose(theta, np.pi / 2):
            return self._cultivate_y_cost
        if np.isclose(theta, np.pi / 4):
            return self._cultivate_t_cost
        raise ValueError(f"Cultivation cost is not defined for angle: {theta}")

    def syndrome_extract_cost(self, op: cirq.Operation, **kwargs) -> CostDict:
        cost_dict = _syndrome_extract_cost(
            rounds=self.rounds,
            num_logical_qubits=len(op.qubits),
            d=self.d,
        )
        cost_dict.op_time = self.total_time(moment_cost_dict=cost_dict.moment_cost)
        return cost_dict

    def error_correct_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return CostDict(op_time=0, moment_cost={}, gate_cost={})

    def measure_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._measure_cost

    def x_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._x_cost

    def z_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._z_cost

    def reset_channel_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        gate_cost: GateCounts = {
            type(op.gate): op.gate.num_qubits() * self.patch.num_physical_qubits
        }
        moment_cost: GateCounts = {cirq.ResetChannel: 1}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    def i_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._i_cost

    def h_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._h_cost

    ### Extra Methods ###
    def __post_init__(self) -> None:
        # Initialize with all shared Primitives then add special ones later
        self.op_cost: dict[type[cirq.Gate], Callable[[cirq.GateOperation], CostDict]] = {
            lsp.Cultivate: self.cultivate_cost,
            lsp.SyndromeExtract: self.syndrome_extract_cost,
            lsp.ErrorCorrect: self.error_correct_cost,
            type(cirq.X): self.x_cost,
            type(cirq.Z): self.z_cost,
            type(cirq.I): self.i_cost,
            type(cirq.H): self.h_cost,
            cirq.MeasurementGate: self.measure_cost,
            cirq.ResetChannel: self.reset_channel_cost,
        }

    def __str__(self) -> str:
        name = self.__name__
        distance = self.d
        cultivation_repetition = self.cultivation_repetition
        round_str = f", sr={self.syndrome_rounds}" if self.syndrome_rounds is not None else ""
        fault_str = f", fd={self.cultivation_fault_distance}"
        fold_str = f", fold={self.fold_cultiv}" if self.movement else ""
        return f"{name}(d={distance}, cr={cultivation_repetition}{fault_str}{round_str}{fold_str})"


class DefaultLattice(Architecture):
    """The subclass used to represent Dual Species without movement
    It uses lattice surgery operations assumes no correlated decoding
    """

    def __init__(
        self,
        idling: bool = True,
        post_op_correction: bool = True,
        d: int = 7,
        cultivation_repetition: int = 1,
        cultivation_fault_distance: Literal[3, 5] = 3,
        syndrome_rounds: int | None = None,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            movement=False,
            d=d,
            cultivation_repetition=cultivation_repetition,
            cultivation_fault_distance=cultivation_fault_distance,
            syndrome_rounds=syndrome_rounds,
            fold_cultiv=False,
        )
        self._primitives = cirq.Gateset(
            lsp.Merge,
            lsp.Split,
            lsp.Cultivate,
            lsp.ErrorCorrect,
            lsp.SyndromeExtract,
            cirq.I,
            cirq.H,
            cirq.X,
            cirq.Z,
            cirq.MeasurementGate,
            cirq.ResetChannel,
        )
        self._phys_gate_times = NEUTRAL_GATES.copy()
        del self._phys_gate_times[css.MovementGate]  # Remove MovementGate
        self.__post_init__()

    def split_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        assert isinstance(op.gate, lsp.Split)
        smooth = op.gate.smooth
        cached_cost = _split_cost(smooth, self.d)
        cost_dict = CostDict(
            gate_cost=cached_cost.gate_cost.copy(),
            moment_cost=cached_cost.moment_cost.copy(),
            op_time=cached_cost.op_time,
        )
        op_time = self.total_time(moment_cost_dict=cost_dict.moment_cost)
        cost_dict.op_time = op_time
        return cost_dict

    def merge_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        assert isinstance(op.gate, lsp.Merge)
        k = op.gate.num_qubits()
        cached_cost = _merge_cost(self.d, k, op.gate.smooth)
        cost = CostDict(
            gate_cost=cached_cost.gate_cost.copy(),
            moment_cost=cached_cost.moment_cost.copy(),
            op_time=cached_cost.op_time,
        )
        op_time = self.total_time(moment_cost_dict=cost.moment_cost)
        cost.op_time = op_time
        return cost

    @cached_property
    def _h_cost(self) -> CostDict:
        # See https://arxiv.org/pdf/2312.11605v1 Fig. 2 for details
        gate_cost: GateCounts = (
            collections.Counter({cirq.PhasedXZGate: 2 * self.patch.num_data_qubits})
            + collections.Counter(_merge_cost(d=self.d, k=2, smooth=True).gate_cost)
            + collections.Counter(_merge_cost(d=self.d, k=2, smooth=True).gate_cost)
            + collections.Counter(
                {
                    cirq.MeasurementGate: self.patch.num_physical_qubits,
                    cirq.ResetChannel: self.patch.num_physical_qubits,
                },
            )
        )
        # One Hadamard (GR, Rz, GR), two Merges, two patch-wide Measure/Reset moments
        # Following the prescription in https://arxiv.org/pdf/2312.11605v1 Fig. 2
        moment_cost: GateCounts = dict(
            collections.Counter({cirq.PhasedXZGate: 1})
            + collections.Counter(_merge_cost(d=self.d, k=2, smooth=True).moment_cost)
            + collections.Counter(_merge_cost(d=self.d, k=2, smooth=True).moment_cost)
            + collections.Counter({cirq.MeasurementGate: 2, cirq.ResetChannel: 2})
        )
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _cultivate_t_cost(self) -> CostDict:
        # fold should always be false here
        base_cultivation_cost = copy(
            cultivate(
                dsurface=self.d,
                fold=self.fold_cultiv,
                fault_distance=self.cultivation_fault_distance,
            )
        )

        # No penalties to any base gates
        moment_cost = base_cultivation_cost.parallel
        gate_cost = base_cultivation_cost.serial

        # Apply cultivation repetition penalty
        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }

        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.op_cost[lsp.Merge] = self.merge_cost
        self.op_cost[lsp.Split] = self.split_cost

    @property
    def __name__(self) -> str:
        return "DualSpeciesNoMovement"


class DefaultMovement(Architecture):
    """Class representing the set of Primitives available with access to movement for transversal operations.
    This default version assumes a single species of neutral atom qubits using a zoned architecture.
    """

    def __init__(
        self,
        idling: bool = False,
        post_op_correction: bool = True,
        d: int = 7,
        fold_cultiv: bool = False,
        cultivation_repetition: int = 1,
        distillation_repetition: int = 1,
        cultivation_fault_distance: Literal[3, 5] = 3,
        syndrome_rounds: int | None = 1,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            movement=True,
            d=d,
            cultivation_repetition=cultivation_repetition,
            cultivation_fault_distance=cultivation_fault_distance,
            syndrome_rounds=syndrome_rounds,
            fold_cultiv=fold_cultiv,
        )
        self.distillation_repetition = distillation_repetition
        self._primitives = cirq.Gateset(
            lsp.Cultivate,
            lsp.Distil,
            lsp.SyndromeExtract,
            lsp.ErrorCorrect,
            css.MovementGate,
            lsp.ResourceCorrection,
            cirq.CNOT,
            cirq.S,
            cirq.I,
            cirq.X,
            cirq.Z,
            cirq.H,
            cirq.MeasurementGate,
            cirq.ResetChannel,
        )
        self._phys_gate_times = NEUTRAL_GATES.copy()
        self.__post_init__()

    zone_ops = cirq.Gateset(cirq.CNOT, cirq.MeasurementGate)

    def cnot_cost(self, op: cirq.GateOperation, **kwargs) -> CostDict:
        return self._cnot_cost

    def syndrome_extract_cost(self, op: cirq.Operation, **kwargs) -> CostDict:
        # Build from the base cost of Syndrome Extraction by adding movement penalties CZ and Measurement moments
        base_cost = copy(super().syndrome_extract_cost(op))
        moment_cost = base_cost.moment_cost
        gate_cost = base_cost.gate_cost
        moment_cost[css.MovementGate] = 2 * (
            moment_cost[cirq.MeasurementGate] + moment_cost[cirq.CZ]
        )
        gate_cost[css.MovementGate] = moment_cost[css.MovementGate]
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(moment_cost=moment_cost, gate_cost=gate_cost, op_time=op_time)

    @cached_property
    def _cnot_cost(self) -> CostDict:
        gate_cost: GateCounts = {
            cirq.PhasedXZGate: 2 * self.patch.num_data_qubits,
            cirq.CZ: self.patch.num_data_qubits,
        }
        # TODO: Resolve this expense with the fact that in the compiler world, we should already have conjugated to CZ by the time we do CNOT
        moment_cost: GateCounts = {
            cirq.CZ: 1,  # Done in parallel
            cirq.PhasedXZGate: 2,  # 1 to conjugate + 1 to unconjugate
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _h_cost(self) -> CostDict:
        gate_cost: GateCounts = {
            cirq.PhasedXZGate: self.patch.num_data_qubits,
            css.MovementGate: 1,
        }
        # Transversal Hadamard with repermuted qubits
        # Technically the physical repermutation could be carried out digitally because there are no connectivity constraints
        moment_cost: GateCounts = {
            cirq.PhasedXZGate: 1,
            css.MovementGate: 1,
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    def correction_cost(self, op: cirq.GateOperation) -> CostDict:
        if not isinstance(op.gate, lsp.ResourceCorrection):
            raise TypeError("Operation is not an instance of ResourceCorrection")
        return self._correction_cost(op.gate._resource)

    def _correction_cost(self, resource: Literal["T", "CCZ"]) -> CostDict:
        # Total time for CCZ correction: t(H) + t(SE) + t(X) + t(SE) + 3 * t(CNOT) * t(SE) + t(H)
        # (can parallelize H gates, X gates, but 3 pairwise CNOTS means we have to do each one
        # sequentially) This means that the moment cost of this correction is one H moment cost, one
        # X moment cost, 3 CNOT coment costs, and one H moment cost, while the overall gate cost is
        # 3 hadamards, 3 X gates, 3 CNOT gates, and 3 hadamards Also flip a coin for both
        # corrections
        # 1: H, SE(n), X, SE(n), IZ, CNOT(0, 1), IZ, SE(n), IZ, CNOT(1, 2), IZ, SE(n), H, SE(n)
        # 2: H, SE(n), X, SE(n), IZ, CNOT(0, 2), IZ, SE(n), IZ, CNOT(1, 2), IZ, SE(n), H, SE(n)
        # H  SE  C  SE  C  SE         H  SE
        # H  SE  X  SE  |      C  SE  H  SE
        # H  SE         X  SE  X  SE  H  SE
        # We only count 4 rounds of syndrome extraction here since the 5th is added as a post-op
        # correction
        outcome = randint(0, 1)
        if resource == "T":
            if outcome:
                return CostDict(op_time=0.0, gate_cost={}, moment_cost={})
            else:
                return self._s_cost
        if outcome:
            return CostDict(op_time=0.0, gate_cost={}, moment_cost={})
        else:
            h_gate_cost = collections.Counter(self._h_cost.gate_cost)
            x_gate_cost = collections.Counter(self._x_cost.gate_cost)
            cnot_gate_cost = collections.Counter(self._cnot_cost.gate_cost)
            qubit = cirq.LineQubit(0)
            se_costs_dict = self.syndrome_extract_cost(
                lsp.SyndromeExtract(num_qubits=1, rounds=self.rounds).on(qubit)
            )
            se_gate_cost = se_costs_dict.gate_cost
            overall_gate_cost: collections.Counter[GateKey] = collections.Counter()
            overall_gate_cost += h_gate_cost
            overall_gate_cost += x_gate_cost
            overall_gate_cost += cnot_gate_cost
            overall_gate_cost += h_gate_cost
            # We do 3 of each of these gates
            for gate in overall_gate_cost:
                overall_gate_cost[gate] *= 3
            # We do 3 syndrome extractions after the first set of hadamards and then 2 after
            # each of the 3 CNOTs
            for gate in se_gate_cost:
                se_gate_cost[gate] *= 9
            overall_gate_cost += se_gate_cost

            h_moment_cost = collections.Counter(self._h_cost.moment_cost)
            x_moment_cost = collections.Counter(self._x_cost.moment_cost)
            cnot_moment_cost = collections.Counter(self._cnot_cost.moment_cost)
            se_moment_cost = collections.Counter(se_costs_dict.moment_cost)
            # CNOT cost is multiplied by 3 since those gates must happen serially
            for gate in cnot_moment_cost:
                cnot_moment_cost[gate] *= 3
            for gate in se_moment_cost:
                se_moment_cost[gate] *= 4
            overall_moment_cost = (
                h_moment_cost + x_moment_cost + cnot_moment_cost + h_moment_cost + se_moment_cost
            )
            # CNOT cost is multiplied by 3 since those gates must happen serially
            h_time = self._h_cost.op_time
            x_time = self._x_cost.op_time
            cnot_time = self._cnot_cost.op_time
            se_time = se_costs_dict.op_time
            total_time = h_time + x_time + 3 * cnot_time + h_time + 4 * se_time
            return CostDict(
                op_time=total_time, gate_cost=overall_gate_cost, moment_cost=overall_moment_cost
            )

    def s_cost(self, op: cirq.Operation, **kwargs) -> CostDict:
        return self._s_cost

    @cached_property
    def _s_cost(self) -> CostDict:
        """Resources the fold transversal S gate from https://arxiv.org/pdf/2412.01391.
        It looks like one Syndrome Extraction round with some CNOT gates across the main diagonal, as well as some physical S/Sdg gates.
        """
        # precompute syndrome extraction cost
        se_cost = self.syndrome_extract_cost(lsp.SyndromeExtract(1, 1).on(cirq.GridQubit(0, 0)))
        # Add the half-cycle fold to the Syndrome Extract gate cost
        gates_from_syndrome = se_cost.gate_cost
        gates_from_middle_fold: GateCounts = {
            cirq.CZ: (self.d - 1) ** 2,
            cirq.PhasedXZGate: self.d,
            css.MovementGate: 2,
        }
        gate_cost = collections.Counter(gates_from_syndrome) + collections.Counter(
            gates_from_middle_fold
        )

        # Add the half-cycle fold to the Syndrome Extract moment cost
        moments_from_syndrome = se_cost.moment_cost
        moments_from_middle_fold: GateCounts = {
            cirq.CZ: 1,
            cirq.PhasedXZGate: 1,
            css.MovementGate: 2,
        }
        moment_cost = collections.Counter(moments_from_syndrome) + collections.Counter(
            moments_from_middle_fold
        )
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    def move_cost(
        self, op, layout, **kwargs
    ) -> CostDict:
        """
        Costs for pre-compiled movement patterns for types of logical moves
        - Moves to/from an interaction zone to accomplish a logical CNOT
        - Moves to/from a measurement zone to accomplish a logical Measurement
        - Moves between logical qubit patches to accomplish a logical CNOT with inplace entanglement
        Total time is a function of physical distance is given by equation (1) of https://arxiv.org/pdf/2505.15907
        The SITE_SPACING parameter gives the distance in microns between qubits that are nearest neighbor in the atom array
        Moves with vertical and horizontal components get penalized independently for each direction
        """
        ctrl, trgt = op.qubits
        move_type = layout.layout_graph.nodes[trgt]["patch_type"]
        site_spacing = layout.site_spacing
        dx = abs(trgt.col - ctrl.col)  # number of logical patches horizontally
        dy = abs(trgt.row - ctrl.row)  # number of logical patches vertically
        patch_length = 2 * self.d
        if move_type == "mzone":
            return _measurement_zone_move_precompiled(
                dx=dx, dy=dy, patch_length=patch_length, site_spacing=site_spacing
            )
        elif move_type == "izone":
            return _interaction_zone_move_precompiled(
                dx=dx, dy=dy, patch_length=patch_length, site_spacing=site_spacing
            )
        else:
            bottom_right = max(layout.layout_graph.nodes)
            # +1 accounts for one extra unit to get to the corner from the bottom right
            scratch_dx = abs(bottom_right.col - trgt.row) + 1
            scratch_dy = abs(bottom_right.row - trgt.row) + 1
            return _inplace_entanglement_move_precompiled(
                dx=dx,
                dy=dy,
                patch_length=patch_length,
                site_spacing=site_spacing,
                scratch_dx=scratch_dx,
                scratch_dy=scratch_dy,
            )

    @cached_property
    def _cultivate_t_cost(self) -> CostDict:
        base_cultivation_cost = copy(
            cultivate(
                dsurface=self.d,
                fold=self.fold_cultiv,
                fault_distance=self.cultivation_fault_distance,
            )
        )
        # Penalize all Measure and CZ moments with QubitPermutationGates
        # Each penalized moment gets penalized with two Moves
        moment_cost = base_cultivation_cost.parallel
        penalties = 2 * (moment_cost.get(cirq.CZ, 0) + moment_cost.get(cirq.MeasurementGate, 0))
        moment_cost[css.MovementGate] = penalties

        # Adjust gate cost to reflect Moves
        gate_cost = base_cultivation_cost.serial
        gate_cost[css.MovementGate] = penalties

        # Apply cultivation repetition penalty
        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }

        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _cultivate_y_cost(self) -> CostDict:
        base_cultivation_cost = copy(super()._cultivate_y_cost)
        # To get the updated cost for the zoned architecture, just add movement where necessary
        new_moment_cost = base_cultivation_cost.moment_cost.copy()
        new_gate_cost = base_cultivation_cost.gate_cost.copy()
        movements_to_add = sum(
            v for k, v in new_moment_cost.items() if k is cirq.MeasurementGate or k is cirq.CZ
        )
        new_moment_cost[css.MovementGate] = movements_to_add
        new_gate_cost[css.MovementGate] = movements_to_add
        new_time = self.total_time(new_moment_cost)
        return CostDict(op_time=new_time, gate_cost=new_gate_cost, moment_cost=new_moment_cost)

    def distil_cost(
        self, op: cirq.Operation, layout, **kwargs
    ) -> CostDict:
        return self._distil_cost(op.gate._resource, layout)

    def _distil_cost(self, resource, layout) -> CostDict:
        # Calculates cost for single repetition based on precompiled circuit
        base_cost = precompute_distil_cost(resource=resource, layout=layout, arc=self)
        op_time = base_cost.op_time * self.distillation_repetition
        moment_cost = collections.Counter(
            {
                key: val * self.distillation_repetition
                for key, val in base_cost.moment_cost.items()
            }
        )
        gate_cost = collections.Counter(
            {key: val * self.distillation_repetition for key, val in base_cost.gate_cost.items()}
        )
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.op_cost[type(cirq.CNOT)] = self.cnot_cost
        self.op_cost[type(cirq.S)] = self.s_cost
        self.op_cost[css.MovementGate] = self.move_cost
        self.op_cost[lsp.Distil] = self.distil_cost
        self.op_cost[lsp.ResourceCorrection] = self.correction_cost

    @property
    def __name__(self) -> str:
        return "SingleSpeciesMovement"


class DualSpeciesMovement(DefaultMovement):
    """Architecture that gets the best of both worlds.
    Atoms of different species can be shuttled along alleyways to get close enough to entangle
    Inplace entanglement and readout are achieved via dual species (and possibly hiding beams)
    CZ gates take place between nearest neighbor physical qubits
    S and CNOT costs are the same as DefaultMovement
    SyndromeExtraction has no movement penalties
    Gidney Cultivation has no movement penalty
    Yale Cultivation penalizes each CZ with one move to penalize long-range interactions
    -
    """

    alley_ops = cirq.Gateset(cirq.CNOT)
    zone_ops = cirq.Gateset()

    # Syndrome Extract from Lattice Surgery
    def syndrome_extract_cost(self, op: cirq.Operation, **kwargs) -> CostDict:
        # Get the syndrome extraction cost without the atom shuttling
        cost_dict = _syndrome_extract_cost(
            rounds=self.rounds,
            num_logical_qubits=len(op.qubits),
            d=self.d,
        )
        cost_dict.op_time = self.total_time(cost_dict.moment_cost)
        return cost_dict

    # Cultivate from Lattice Surgery
    @cached_property
    def _cultivate_t_cost(self) -> CostDict:
        """Cached property for the cultivation circuit having the relevant parameters: code distance (d) and movement
        Values are multiplied by the repeat factor for the architecture instance
        """
        base_cultivation_cost = copy(
            cultivate(
                dsurface=self.d,
                fold=self.fold_cultiv,
                fault_distance=self.cultivation_fault_distance,
            )
        )
        gate_cost = base_cultivation_cost.serial
        moment_cost = base_cultivation_cost.parallel
        if self.fold_cultiv:
            moment_cost[css.MovementGate] = 1 * moment_cost.get(cirq.CZ, 0)
            gate_cost[css.MovementGate] = 1 * moment_cost.get(cirq.CZ, 0)

        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _cultivate_y_cost(self) -> CostDict:
        # Nearest neighbor Gidney style, so no movement penalty
        return Architecture._cultivate_y_cost.__get__(self, type(self))

    # Measurement from Lattice Surgery
    @cached_property
    def _measure_cost(self) -> CostDict:
        gate_cost: GateCounts = {cirq.MeasurementGate: self.patch.num_data_qubits}
        moment_cost: GateCounts = {cirq.MeasurementGate: 1}
        op_time: float = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @property
    def __name__(self) -> str:
        return "DualSpeciesMovement"


class MeasureZonesOnly(DefaultMovement):
    """A movement-based Architecture with a Measurement Zone
    Atoms can be shuttled along alleyways to get close to each other
    Inplace entanglement is enabled through the use of hiding, avoiding the Interaction Zone
    CZ gates can take place with nearest-neighbor physical qubits
    S and CNOT costs are the same as DefaultMovement
    SyndromeExtraction and Cultivate have a movement penalty for each Measurement
    Yale Cultivation penalizes each CZ with one move to penalize long-range interactions
    """

    zone_ops = cirq.Gateset(cirq.MeasurementGate)
    alley_ops = cirq.Gateset(cirq.CNOT)

    # TODO: How do we do S gates here?
    #       a) Perform S with fold transversal S gate enabled by motion
    #       b) Cultivate S with "inplace" procedure (Class must inherit from Lattice)
    # For now, I am going with option a), which is the same as DefaultMovement

    def syndrome_extract_cost(self, op: cirq.Operation, **kwargs) -> CostDict:
        """Uses lattice surgery Syndrome Extraction but adds moves associated with the measurements.
        Since this class is a Movement architecture, its rounds should be low, in accordance with the promise of correlated decoding.
        """
        base_cost = _syndrome_extract_cost(
            rounds=self.rounds,
            num_logical_qubits=len(op.qubits),
            d=self.d,
        )
        moment_cost = base_cost.moment_cost
        gate_cost = base_cost.gate_cost
        moment_cost[css.MovementGate] = 2 * moment_cost.get(cirq.MeasurementGate, 0)
        gate_cost[css.MovementGate] = moment_cost[css.MovementGate]
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(moment_cost=moment_cost, gate_cost=gate_cost, op_time=op_time)

    @cached_property
    def _cultivate_t_cost(self) -> CostDict:
        base_cultivation_cost = copy(
            cultivate(
                dsurface=self.d,
                fold=self.fold_cultiv,
                fault_distance=self.cultivation_fault_distance,
            )
        )
        gate_cost = base_cultivation_cost.serial
        moment_cost = base_cultivation_cost.parallel
        if self.fold_cultiv:
            # Penalize CZ by half
            moment_cost[css.MovementGate] = 1 * moment_cost.get(cirq.CZ, 0)
            gate_cost[css.MovementGate] = moment_cost[css.MovementGate]
        else:
            # Do not penalize at all
            moment_cost[css.MovementGate] = 0
            gate_cost[css.MovementGate] = 0
        # Penalize Measure by two moves per Measure to represent going to/from an Measurement Zone
        moment_cost[css.MovementGate] += 2 * moment_cost.get(cirq.MeasurementGate, 0)
        gate_cost[css.MovementGate] += 2 * moment_cost.get(cirq.MeasurementGate, 0)

        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return CostDict(op_time=op_time, moment_cost=moment_cost, gate_cost=gate_cost)

    @cached_property
    def _cultivate_y_cost(self) -> CostDict:
        base_cultivation_cost = copy(super()._cultivate_y_cost)
        # Penalize measurements but not entangling gates
        new_moment_cost = copy(base_cultivation_cost.moment_cost)
        new_gate_cost = copy(base_cultivation_cost.gate_cost)
        movements_to_add = sum(
            v for k, v in new_moment_cost.items() if k is cirq.MeasurementGate
        )
        new_moment_cost[css.MovementGate] = movements_to_add
        new_gate_cost[css.MovementGate] = movements_to_add
        new_time = self.total_time(new_moment_cost)
        return CostDict(op_time=new_time, gate_cost=new_gate_cost, moment_cost=new_moment_cost)

    @property
    def __name__(self) -> str:
        return "ReadoutZonesOnly"


class Superconductor(DefaultLattice):
    """Class to serve as a proxy for superconducting architectures.
    It features a gateset composed of CZ + 1Q
    It's main feature is its fast gate speeds compared to all other architectures
    """

    def __init__(
        self,
        idling: bool = True,
        post_op_correction: bool = True,
        d: int = 7,
        cultivation_repetition: int = 1,
        cultivation_fault_distance: Literal[3, 5] = 3,
        syndrome_rounds: int | None = None,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            d=d,
            cultivation_repetition=cultivation_repetition,
            cultivation_fault_distance=cultivation_fault_distance,
            syndrome_rounds=syndrome_rounds,
        )
        self._primitives: cirq.Gateset = cirq.Gateset(
            lsp.Merge,
            lsp.Split,
            lsp.Cultivate,
            lsp.ErrorCorrect,
            lsp.SyndromeExtract,
            cirq.I,
            cirq.H,
            cirq.X,
            cirq.Z,
            cirq.MeasurementGate,
            cirq.ResetChannel,
        )
        self._phys_gate_times: dict[GateKey, float] = SUPERCOND_GATES.copy()
        self.__post_init__()

    @property
    def __name__(self) -> str:
        return "Superconductor"
