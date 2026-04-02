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
import json
import numpy as np
from collections import Counter
from functools import cached_property, lru_cache
from math import ceil
from pathlib import Path

import cirq
from . import lattice_surgery_primitives as lsp
from cirq_superstaq.ops.qubit_gates import ParallelRGate
from resource_estimation.stim_functions import cultivate
import abc


HARVARD_GATES = {  # From Harvard paper (https://arxiv.org/pdf/2506.20661)
    cirq.CZ: 0.27,
    cirq.PhasedXZGate: 5.0,  # Based on single qubit gate times
    cirq.ResetChannel: 400,  # A few hundred us
    cirq.MeasurementGate: 1000,  # Best guess from 500us for atom movement during readout
    cirq.QubitPermutationGate: 500,
}
HILBERT_GATES = {
    cirq.CZ: 0.40,
    cirq.PhasedXZGate: 8,
    cirq.ResetChannel: 50000,  # Reset time from Dan: 50-100ms
    cirq.MeasurementGate: 10000.0,  # Measurement time from Dan: 10ms
    cirq.QubitPermutationGate: 500,  # Copying from Harvard since it's needed
}
SUPERCOND_GATES = {
    
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
) -> dict[str, dict[cirq.Gate, int]]:
    """
    Calculates the resources required to implement the merge operation.
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

    gate_cost = {
        cirq.CZ: cz_gates,
        cirq.PhasedXZGate: rz_gates,
        cirq.MeasurementGate: measures,
        cirq.ResetChannel: resets,
    }
    moment_cost = {
        cirq.CZ: 4 * d,
        cirq.PhasedXZGate: 2 * d,
        cirq.MeasurementGate: d,
        cirq.ResetChannel: d,
    }
    return {"gate_cost": gate_cost, "moment_cost": moment_cost}


def _syndrome_extract_cost(
    rounds: int,
    num_logical_qubits: int,
    d: int,
) -> dict[str, dict[cirq.Gate, int]]:
    """
    Calculates the cost of syndrome extraction in terms of physical gates
    """
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
    gate_cost = {
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
    moment_cost = {
        cirq.MeasurementGate: rounds,
        cirq.ResetChannel: rounds,
        cirq.CZ: 4 * rounds,  # 4 per stabilizer type
        cirq.PhasedXZGate: 2 * rounds,  # 2 per stabilizer type
    }
    return {"gate_cost": gate_cost, "moment_cost": moment_cost}


@lru_cache(maxsize=128)
def _split_cost(smooth: bool, d: int) -> dict[str, dict[cirq.Gate, int]]:
    """
    Calculates cost to perform a split operation
    Split operations can always be absorbed into a following moment, so the moment cost is null
    """
    if smooth:
        # Measuring in the X-basis costs an extra basis change
        gate_cost = {
            cirq.MeasurementGate: 2 * d + 1,
            cirq.PhasedXZGate: ceil((2 * d + 1) / 2),
        }
    else:
        gate_cost = {
            cirq.MeasurementGate: 2 * d + 1,
        }
    moment_cost = {}
    return {"gate_cost": gate_cost, "moment_cost": moment_cost}


class Architecture(abc.ABC):
    """
    Class for representing device architectures.

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
        syndrome_rounds: int | None = None,
        fold_cultiv: bool = False,
    ) -> None:
        self.idling: bool = idling
        self.post_op_correction = post_op_correction
        self.movement = movement
        self.d = d
        self.patch = lsp.RotatedCodePatch(self.d)
        self.cultivation_repetition = cultivation_repetition
        self.syndrome_rounds = syndrome_rounds
        self.fold_cultiv = fold_cultiv

        self._primitives = None
        self._phys_gate_times = None
        self.__post_init__()

    ### Class Methods ###
    # TODO: Deprecate these
    @classmethod
    def from_dict(cls, d: dict):
        movement = d["movement"]
        if movement:
            base_arc = DefaultMovement(
                idling=d["idling"],
                post_op_correction=d["post_op_correction"],
                d=d["d"],
                cultivation_repetition=d["cultivation_repetition"],
                syndrome_rounds=d["syndrome_rounds"],
                fold_cultiv=d.get("fold_cultiv", False),
            )
        else:
            base_arc = DefaultLattice(
                idling=d["idling"],
                post_op_correction=d["post_op_correction"],
                d=d["d"],
                cultivation_repetition=d["cultivation_repetition"],
                syndrome_rounds=d["syndrome_rounds"],
            )
        # TODO: Check once the flag
        base_arc.__post_init__()
        if "gate_times" in d:
            for gate, gate_time in d["gate_times"].items():
                if gate in base_arc.phys_gate_times:
                    base_arc.phys_gate_times[gate] = gate_time
                elif isinstance(gate, str) and gate in {
                    obj.__name__ if hasattr(obj, "__name__") else str(obj)
                    for obj in base_arc.phys_gate_times
                }:
                    cirq_gates = {
                        obj.__name__ if hasattr(obj, "__name__") else str(obj): obj
                        for obj in base_arc.phys_gate_times
                    }
                    if gate in cirq_gates:
                        base_arc.phys_gate_times[cirq_gates[gate]] = gate_time
                else:
                    raise ValueError(f"Gate time not found for {gate}")
        return base_arc

    @classmethod
    def from_json(cls, fp: str | Path):
        with open(fp) as f:
            input_dict = json.load(f)
        return cls.from_dict(input_dict)

    ### Fundamental Cost Counting Methods ###
    # These should never be overwritten
    def gate_cost(self, op: cirq.Operation) -> dict:
        try:
            return self.op_cost[type(op.gate)](op)["gate_cost"]
        except KeyError:
            raise ValueError("Gate not recognized")

    def op_time(self, op: cirq.Operation) -> int:
        try:
            return self.op_cost[type(op.gate)](op)["op_time"]
        except KeyError:
            raise ValueError("Gate not recognized")

    def moment_cost(self, op: cirq.Operation) -> dict:
        try:
            return self.op_cost[type(op.gate)](op)["moment_cost"]
        except KeyError:
            raise ValueError("Gate not recognized")

    def total_time(self, moment_cost_dict: dict) -> float:
        return sum(
            num_ops * self.phys_gate_times[phys_op] for phys_op, num_ops in moment_cost_dict.items()
        )

    ### Properties ###
    # These are particular costs that are often repeated
    # In the future, they could be the counts of a NISQ compiler on a single Primitive
    # Some are cached because they can be expensive to call repeatedly

    # HOT TAKE: Everybody handles t in their own way
    @cached_property
    def _cultivate_t_cost(self):  # pragma: no cover
        raise NotImplementedError

    @cached_property
    def _cultivate_y_cost(self):
        """
        Cost estimate for measuring a surface code patch in the Y basis. Measuring in the Y basis facilitates gate teleportation the same way that cultivating T does.
        The procedure is based on [Inplace access to the Surface Code Y Basis](https://arxiv.org/pdf/2302.07395v2).
        The cost estimates were generated by looking carefully at circuits produced from https://doi.org/10.5281/zenodo.7487893.
        """
        # TODO: Consider using the Zenodo code directly
        single_qubit_moments = 5
        reset_moments = 1
        cz_moments = 5
        measure_moments = 1
        Y_moment_cost = Counter(
            {
                cirq.PhasedXZGate: single_qubit_moments,
                cirq.CZ: cz_moments,
                cirq.MeasurementGate: measure_moments,
                cirq.ResetChannel: reset_moments,
            }
        )
        se_moment_cost = Counter(
            _syndrome_extract_cost(rounds=ceil(self.d / 2), num_logical_qubits=1, d=self.d)[
                "moment_cost"
            ]
        )

        # TODO: Perhaps cannonical cost includes SE before and afer for a total of two more units of SE
        moment_cost = se_moment_cost + Y_moment_cost + Y_moment_cost
        op_time = self.total_time(moment_cost_dict=moment_cost)

        # For the gate cost, let's just approximate it with one round of syndrome extraction with an additional d-1 diagonal of CZ gates
        se_gate_cost = Counter(
            _syndrome_extract_cost(rounds=ceil(self.d / 2), num_logical_qubits=1, d=self.d)[
                "gate_cost"
            ]
        )
        Y_gate_cost = se_gate_cost.copy()
        Y_gate_cost[cirq.CZ] += self.d - 1
        gate_cost = se_gate_cost + Y_gate_cost + Y_gate_cost

        return {
            "gate_cost": gate_cost,
            "moment_cost": moment_cost,
            "op_time": op_time,
        }

    @cached_property
    def _x_cost(self):
        gate_cost = {}
        moment_cost = {}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _z_cost(self):
        gate_cost = {}
        moment_cost = {}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _i_cost(self):
        gate_cost = {}
        moment_cost = {}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _measure_cost(self) -> dict:
        gate_cost = {cirq.MeasurementGate: self.patch.num_data_qubits}
        moment_cost = {cirq.MeasurementGate: 1}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @property
    def rounds(self):
        if self.syndrome_rounds is None:
            return self.d
        else:
            return self.syndrome_rounds

    @property
    def phys_gate_times(self) -> dict[cirq.Gate, float]:
        return self._phys_gate_times

    @property
    @abc.abstractmethod
    def __name__(self) -> str:  # pragma: no cover
        pass

    @property
    def primitives(self) -> cirq.Gateset:
        return self._primitives

    zone_ops = None

    alley_ops = None

    ### Top Level Cost Methods ###
    # Functions used to interpret the costs of Primitives in the form of cirq operations
    # The ones here are common among all architectures currently
    def cultivate_cost(self, op: cirq.Operation) -> dict:
        theta = op.gate.theta
        if np.isclose(theta, np.pi / 2):
            return self._cultivate_y_cost
        elif np.isclose(theta, np.pi / 4):
            return self._cultivate_t_cost
        else:
            raise ValueError(f"Cultivation cost is not defined for angle: {theta}")

    def syndrome_extract_cost(self, op: cirq.Operation) -> dict:
        cost_dict = _syndrome_extract_cost(
            rounds=self.rounds, num_logical_qubits=len(op.qubits), d=self.d
        )
        cost_dict["op_time"] = self.total_time(moment_cost_dict=cost_dict["moment_cost"])
        return cost_dict

    def error_correct_cost(self, op: cirq.Operation) -> dict:
        gate_cost = {}
        moment_cost = {}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    def measure_cost(self, op: cirq.Operation) -> dict:
        return self._measure_cost

    def x_cost(self, op: cirq.Operation) -> dict:
        return self._x_cost

    def z_cost(self, op: cirq.Operation) -> dict:
        return self._z_cost

    def reset_channel_cost(self, op: cirq.Operation) -> dict:
        gate_cost = {type(op.gate): op.gate.num_qubits() * self.patch.num_physical_qubits}
        moment_cost = {cirq.ResetChannel: 1}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    def i_cost(self, op: cirq.Operation) -> dict:
        return self._i_cost

    def h_cost(self, op: cirq.Operation) -> dict:
        return self._h_cost

    ### Extra Methods ###
    def __post_init__(self):
        # Initialize with all shared Primitives then add special ones later
        self.op_cost = {
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
        fold_str = f", fold={self.fold_cultiv}" if self.movement else ""
        return f"{name}(d={distance}, cr={cultivation_repetition}{round_str}{fold_str})"


class DefaultLattice(Architecture):
    """
    The subclass used to represent Dual Species without movement
    It uses lattice surgery operations assumes no correlated decoding
    """

    def __init__(
        self,
        idling: bool = True,
        post_op_correction: bool = True,
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=None,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            movement=False,
            d=d,
            cultivation_repetition=cultivation_repetition,
            syndrome_rounds=syndrome_rounds,
            fold_cultiv=False,
        )
        self._primitives = cirq.Gateset(
            *[
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
            ]
        )
        self._phys_gate_times = HARVARD_GATES.copy()
        del self._phys_gate_times[cirq.QubitPermutationGate]  # Remove PermutationGate
        self.__post_init__()

    def split_cost(self, op: cirq.Operation) -> dict:
        smooth = op.gate.smooth
        cost_dict = _split_cost(smooth, self.d)
        op_time = self.total_time(moment_cost_dict=cost_dict["moment_cost"])
        cost_dict["op_time"] = op_time
        return cost_dict

    def merge_cost(self, op: cirq.Operation) -> dict:
        k = op.gate.num_qubits()
        cost = _merge_cost(self.d, k, op.gate.smooth)
        op_time = self.total_time(moment_cost_dict=cost["moment_cost"])
        cost["op_time"] = op_time
        return cost

    @cached_property
    def _h_cost(self) -> dict:
        # See https://arxiv.org/pdf/2312.11605v1 Fig. 2 for details
        gate_cost = (
            Counter({cirq.PhasedXZGate: 2 * self.patch.num_data_qubits})
            + Counter(_merge_cost(d=self.d, k=2, smooth=True)["gate_cost"])
            + Counter(_merge_cost(d=self.d, k=2, smooth=True)["gate_cost"])
            + Counter(
                {
                    cirq.MeasurementGate: self.patch.num_physical_qubits,
                    cirq.ResetChannel: self.patch.num_physical_qubits,
                }
            )
        )
        # One Hadamard (GR, Rz, GR), two Merges, two patch-wide Measure/Reset moments
        # Following the prescription in https://arxiv.org/pdf/2312.11605v1 Fig. 2
        moment_cost = dict(
            Counter({cirq.PhasedXZGate: 1})
            + Counter(_merge_cost(d=self.d, k=2, smooth=True)["moment_cost"])
            + Counter(_merge_cost(d=self.d, k=2, smooth=True)["moment_cost"])
            + Counter({cirq.MeasurementGate: 2, cirq.ResetChannel: 2})
        )
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _cultivate_t_cost(self):
        # fold should always be false here
        base_cultivation_cost = cultivate(dsurface=self.d, fold=self.fold_cultiv).copy()

        # No penalties to any base gates
        moment_cost = base_cultivation_cost["parallel"]
        gate_cost = base_cultivation_cost["serial"]

        # Apply cultivation repetition penalty
        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }

        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    def __post_init__(self) -> None:
        super().__post_init__()
        self.op_cost[lsp.Merge] = self.merge_cost
        self.op_cost[lsp.Split] = self.split_cost

    @property
    def __name__(self) -> str:
        return "DualSpeciesNoMovement"


class DefaultMovement(Architecture):
    """
    Class representing the set of Primitives available with access to movement for transversal operations.
    This default version is just the Harvard architecture based on atom shuttling.
    """

    def __init__(
        self,
        idling: bool = False,
        post_op_correction: bool = True,
        d: int = 7,
        fold_cultiv=False,
        cultivation_repetition=1,
        syndrome_rounds=1,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            movement=True,
            d=d,
            cultivation_repetition=cultivation_repetition,
            syndrome_rounds=syndrome_rounds,
            fold_cultiv=fold_cultiv,
        )
        self._primitives = cirq.Gateset(
            *[
                lsp.Cultivate,
                lsp.SyndromeExtract,
                lsp.ErrorCorrect,
                lsp.Move,
                cirq.CNOT,
                cirq.S,
                cirq.I,
                cirq.X,
                cirq.Z,
                cirq.H,
                cirq.MeasurementGate,
                cirq.ResetChannel,
            ]
        )
        self._phys_gate_times = HARVARD_GATES.copy()
        self.__post_init__()

    zone_ops = cirq.Gateset(cirq.CNOT, cirq.MeasurementGate)

    def cnot_cost(self, op: cirq.Operation) -> dict:
        return self._cnot_cost

    def syndrome_extract_cost(self, op):
        # Build from the base cost of Syndrome Extraction by adding movement penalties CZ and Measurement moments
        base_cost = super().syndrome_extract_cost(op).copy()
        moment_cost = base_cost["moment_cost"]
        gate_cost = base_cost["gate_cost"]
        moment_cost[cirq.QubitPermutationGate] = 2 * (
            moment_cost[cirq.MeasurementGate] + moment_cost[cirq.CZ]
        )
        gate_cost[cirq.QubitPermutationGate] = moment_cost[cirq.QubitPermutationGate]
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"moment_cost": moment_cost, "gate_cost": gate_cost, "op_time": op_time}

    @cached_property
    def _cnot_cost(self) -> dict:
        gate_cost = {
            cirq.PhasedXZGate: 2 * self.patch.num_data_qubits,
            cirq.CZ: self.patch.num_data_qubits,
        }
        # TODO: Resolve this expense with the fact that in the compiler world, we should already have conjugated to CZ by the time we do CNOT
        moment_cost = {
            cirq.CZ: 1,  # Done in parallel
            cirq.PhasedXZGate: 2,  # 1 to conjugate + 1 to unconjugate
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _h_cost(self) -> dict:
        gate_cost = {
            cirq.PhasedXZGate: self.patch.num_data_qubits,
            cirq.QubitPermutationGate: 1,
        }
        # Transversal Hadamard with repermuted qubits
        # Technically the physical repermutation could be carried out digitally because there are no connectivity constraints
        moment_cost = {
            cirq.PhasedXZGate: 1,
            cirq.QubitPermutationGate: 1,
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    def s_cost(self, op: cirq.Operation) -> dict:
        return self._s_cost

    @cached_property
    def _s_cost(self) -> dict:
        """
        Resources the fold transversal S gate from https://arxiv.org/pdf/2412.01391.
        It looks like one Syndrome Extraction round with some CNOT gates across the main diagonal, as well as some physical S/Sdg gates.
        """
        # precompute syndrome extraction cost
        se_cost = self.syndrome_extract_cost(lsp.SyndromeExtract(1, 1).on(cirq.GridQubit(0, 0)))
        # Add the half-cycle fold to the Syndrome Extract gate cost
        gates_from_syndrome = se_cost["gate_cost"]
        gates_from_middle_fold = {
            cirq.CZ: (self.d - 1) ** 2,
            cirq.PhasedXZGate: self.d,
            cirq.QubitPermutationGate: 2,
        }
        gate_cost = Counter(gates_from_syndrome) + Counter(gates_from_middle_fold)

        # Add the half-cycle fold to the Syndrome Extract moment cost
        moments_from_syndrome = se_cost["moment_cost"]
        moments_from_middle_fold = {cirq.CZ: 1, cirq.PhasedXZGate: 1, cirq.QubitPermutationGate: 2}
        moment_cost = Counter(moments_from_syndrome) + Counter(moments_from_middle_fold)
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    def move_cost(self, op):
        """
        Method to handle both types of movement
        The maximum move time should be 500us, which corresponds to travelling to a zone
        Everything else should be penalized by distance away up to a distance of 500us
        This reference says something about .55um/us (https://www.nature.com/articles/s41586-022-04592-6.pdf)
        To make things easier, I'm going to call that .5um/us
        A surface code patch has a side length of ~d physical qubits
        If we assume qubits are spaced by ~1um, it takes about 2*d us to move a qubit to an adjacent patch
        So if the L1 distance between logical qubits A and B is C, then we penalize Move(A, B) with time 2*C*d (up to a maximum of 500us)
        This feels a little too weighted in favor of alleyway movement, but it is at least a rule, and it's something worth debating
        """
        gate_cost = {cirq.QubitPermutationGate: 1}
        moment_cost = {cirq.QubitPermutationGate: 1}
        if op.gate.zone is None:
            ctrl, trgt = op.qubits
            distance = abs(trgt.row - ctrl.row) + abs(trgt.col - ctrl.col)
            penalty_factor = 2 * self.d * distance
            time_cap = self.phys_gate_times[cirq.QubitPermutationGate]
            op_time = min(penalty_factor, time_cap)
        else:
            op_time = self.phys_gate_times[
                cirq.QubitPermutationGate
            ]  # Just a basic penalty based on the literature
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _cultivate_t_cost(self):
        base_cultivation_cost = cultivate(dsurface=self.d, fold=self.fold_cultiv).copy()
        # Penalize all Measure and CZ moments with QubitPermutationGates
        # Each penalized moment gets penalized with two Moves
        moment_cost = base_cultivation_cost["parallel"]
        penalties = 2 * (moment_cost.get(cirq.CZ, 0) + moment_cost.get(cirq.MeasurementGate, 0))
        moment_cost[cirq.QubitPermutationGate] = penalties

        # Adjust gate cost to reflect Moves
        gate_cost = base_cultivation_cost["serial"]
        gate_cost[cirq.QubitPermutationGate] = penalties

        # Apply cultivation repetition penalty
        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }

        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _cultivate_y_cost(self):
        base_cultivation_cost = super()._cultivate_y_cost.copy()
        # To get the updated cost for the zoned architecture, just add movement where necessary
        new_moment_cost = base_cultivation_cost["moment_cost"].copy()
        new_gate_cost = base_cultivation_cost["gate_cost"].copy()
        permutations_to_add = sum(
            v for k, v in new_moment_cost.items() if k is cirq.MeasurementGate or k is cirq.CZ
        )
        new_moment_cost[cirq.QubitPermutationGate] = permutations_to_add
        new_gate_cost[cirq.QubitPermutationGate] = permutations_to_add
        new_time = self.total_time(new_moment_cost)
        return {"op_time": new_time, "gate_cost": new_gate_cost, "moment_cost": new_moment_cost}

    def __post_init__(self) -> None:
        super().__post_init__()
        self.op_cost[type(cirq.CNOT)] = self.cnot_cost
        self.op_cost[type(cirq.S)] = self.s_cost
        self.op_cost[lsp.Move] = self.move_cost

    @property
    def __name__(self) -> str:
        return "SingleSpeciesMovement"


class DualSpeciesMovement(DefaultMovement):
    """
    Architecture that gets the best of both worlds.
    Atoms of different species can be shuttled along alleyways to get close enough to entangle
    Inplace entanglement and readout are achieved via dual species (and possibly hiding beams)
    CZ gates take place between nearest neighbor physical qubits
    S and CNOT costs are the same as DefaultMovement
    SyndromeExtraction has no movement penalties
    Gidney Cultivation has no movement penalty
    Yale Cultivation penalizes each CZ with one move to penalize long-range interactions
    -
    """

    zone_ops = None
    alley_ops = cirq.Gateset(cirq.CNOT)

    # Syndrome Extract from Lattice Surgery
    def syndrome_extract_cost(self, op: cirq.Operation) -> dict:
        # Get the syndrome extraction cost without the atom shuttling
        cost_dict = _syndrome_extract_cost(
            rounds=self.rounds, num_logical_qubits=len(op.qubits), d=self.d
        )
        cost_dict["op_time"] = self.total_time(cost_dict["moment_cost"])
        return cost_dict

    # Cultivate from Lattice Surgery
    @cached_property
    def _cultivate_t_cost(self):
        """
        Cached property for the cultivation circuit having the relevant parameters: code distance (d) and movement
        Values are multiplied by the repeat factor for the architecture instance
        """
        base_cultivation_cost = cultivate(dsurface=self.d, fold=self.fold_cultiv).copy()
        gate_cost = base_cultivation_cost["serial"]
        moment_cost = base_cultivation_cost["parallel"]
        if self.fold_cultiv:
            moment_cost[cirq.QubitPermutationGate] = 1 * moment_cost.get(cirq.CZ, 0)
            gate_cost[cirq.QubitPermutationGate] = 1 * moment_cost.get(cirq.CZ, 0)

        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _cultivate_y_cost(self):
        # Nearest neighbor Gidney style, so no movement penalty
        return Architecture._cultivate_y_cost.__get__(self, type(self))

    # Measurement from Lattice Surgery
    @cached_property
    def _measure_cost(self):
        gate_cost = {cirq.MeasurementGate: self.patch.num_data_qubits}
        moment_cost = {cirq.MeasurementGate: 1}
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @property
    def __name__(self):
        return "DualSpeciesMovement"


class MeasureZonesOnly(DefaultMovement):
    """
    A movement-based Architecture with a Measurement Zone
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

    def syndrome_extract_cost(self, op: cirq.Operation) -> dict:
        """
        Uses lattice surgery Syndrome Extraction but adds moves associated with the measurements.
        Since this class is a Movement architecture, its rounds should be low, in accordance with the promise of correlated decoding.
        """
        base_cost = _syndrome_extract_cost(
            rounds=self.rounds, num_logical_qubits=len(op.qubits), d=self.d
        )
        moment_cost = base_cost["moment_cost"]
        gate_cost = base_cost["gate_cost"]
        moment_cost[cirq.QubitPermutationGate] = 2 * moment_cost.get(cirq.MeasurementGate, 0)
        gate_cost[cirq.QubitPermutationGate] = moment_cost[cirq.QubitPermutationGate]
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"moment_cost": moment_cost, "gate_cost": gate_cost, "op_time": op_time}

    @cached_property
    def _cultivate_t_cost(self):
        base_cultivation_cost = cultivate(dsurface=self.d, fold=self.fold_cultiv).copy()
        gate_cost = base_cultivation_cost["serial"]
        moment_cost = base_cultivation_cost["parallel"]
        if self.fold_cultiv:
            # Penalize CZ by half
            moment_cost[cirq.QubitPermutationGate] = 1 * moment_cost.get(cirq.CZ, 0)
            gate_cost[cirq.QubitPermutationGate] = moment_cost[cirq.QubitPermutationGate]
        else:
            # Do not penalize at all
            moment_cost[cirq.QubitPermutationGate] = 0
            gate_cost[cirq.QubitPermutationGate] = 0
        # Penalize Measure by two moves per Measure to represent going to/from an Measurement Zone
        moment_cost[cirq.QubitPermutationGate] += 2 * moment_cost.get(cirq.MeasurementGate, 0)
        gate_cost[cirq.QubitPermutationGate] += 2 * moment_cost.get(cirq.MeasurementGate, 0)

        gate_cost = {gate: cost * self.cultivation_repetition for gate, cost in gate_cost.items()}
        moment_cost = {
            moment: cost * self.cultivation_repetition for moment, cost in moment_cost.items()
        }
        op_time = self.total_time(moment_cost_dict=moment_cost)
        return {"op_time": op_time, "gate_cost": gate_cost, "moment_cost": moment_cost}

    @cached_property
    def _cultivate_y_cost(self):
        base_cultivation_cost = super()._cultivate_y_cost.copy()
        # Penalize measurements but not entangling gates
        new_moment_cost = base_cultivation_cost["moment_cost"].copy()
        new_gate_cost = base_cultivation_cost["gate_cost"].copy()
        permutations_to_add = sum(
            v for k, v in new_moment_cost.items() if k is cirq.MeasurementGate
        )
        new_moment_cost[cirq.QubitPermutationGate] = permutations_to_add
        new_gate_cost[cirq.QubitPermutationGate] = permutations_to_add
        new_time = self.total_time(new_moment_cost)
        return {"op_time": new_time, "gate_cost": new_gate_cost, "moment_cost": new_moment_cost}

    @property
    def __name__(self) -> str:
        return "ReadoutZonesOnly"


class Hilbert(DefaultMovement):  # pragma: no cover
    # No need to cover currently because this is just a home for the commented out gate times that used to live in the top level architecture class.
    """
    Class inspired by the Hilbert architecture detailed in https://arxiv.org/pdf/2408.08288
    It exists primarily to highlight the impact how gate times bottleneck performance
    """

    @property
    def phys_gate_times(self):
        return HILBERT_GATES

    def __name__(self) -> str:
        return "Hilbert"


class Superconductor(DefaultLattice):
    """
    Class to serve as a proxy for superconducting architectures.
    It features a gateset composed of CZ + 1Q
    It's main feature is its fast gate speeds compared to all other architectures
    """

    def __init__(
        self,
        idling: bool = True,
        post_op_correction: bool = True,
        d=7,
        cultivation_repetition=1,
        syndrome_rounds=None,
    ) -> None:
        super().__init__(
            idling=idling,
            post_op_correction=post_op_correction,
            d=d,
            cultivation_repetition=cultivation_repetition,
            syndrome_rounds=syndrome_rounds,
        )
        self._primitives = cirq.Gateset(
            *[
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
            ]
        )
        self._phys_gate_times = SUPERCOND_GATES.copy()
        self.__post_init__()

    @property
    def __name__(self) -> str:
        return "Superconductor"


# Deprecated by removing GR gates
def convert_globals_to_phasedxz(architecture: Architecture, cost_with_globals: dict) -> dict:
    """
    Converts costs defined by GR and Rz into PhasedXZ by removing GR and replacing Rz with PhasedXZ to represent arbitrary single qubit rotations
    """
    gate_cost = cost_with_globals["gate_cost"].copy()
    if ParallelRGate in gate_cost:
        del gate_cost[ParallelRGate]
    rz_gates = gate_cost.get(cirq.Rz, 0)
    if rz_gates:
        gate_cost[cirq.PhasedXZGate] = rz_gates
        del gate_cost[cirq.Rz]

    moment_cost = cost_with_globals["moment_cost"].copy()
    if ParallelRGate in moment_cost:
        del moment_cost[ParallelRGate]
    rz_moments = moment_cost.get(cirq.Rz, 0)
    if rz_moments:
        moment_cost[cirq.PhasedXZGate] = moment_cost.get(cirq.Rz, 0)
        del moment_cost[cirq.Rz]

    op_time = architecture.total_time(moment_cost_dict=moment_cost)
    return {"gate_cost": gate_cost, "moment_cost": moment_cost, "op_time": op_time}
