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
from functools import cached_property
import cirq
from typing import Literal

# TODO: Add cirq diagram info


def custom_resolver(cirq_type: str) -> type[cirq.Gate] | None:
    """
    Tells cirq.json how to deserialize custom gates
    """
    if cirq_type == "lsp.Merge":
        return Merge
    if cirq_type == "lsp.Split":
        return Split
    if cirq_type == "lsp.SyndromeExtract":
        return SyndromeExtract
    if cirq_type == "lsp.Cultivate":
        return Cultivate
    if cirq_type == "lsp.ErrorCorrect":
        return ErrorCorrect
    if cirq_type == "lsp.Move":
        return Move


@cirq.value_equality
class Merge(cirq.Gate):
    def __init__(self, num_qubits: int, smooth: bool = True):
        """
        Subclassed cirq gate to represent the Merge operation in lattice surgery.
        The Merge operation combines the stabilizers of a set of distinct surface code patches along the boundary qubits.
        Depending on these boundaries, the merge can be smooth or rough.
        See https://arxiv.org/pdf/1111.4022 for details.

        Currently this gate expects to merge patches representing well-defined qubits.
        In reality, merging blobs of various sizes can frustrate the notion of 'num_qubits' for this operation.
        However, for the purposes of resource estimation, it is expedient to sweep much of complexity under the rug.

        num_qubits: The number patches (corresponding to logical qubits) to merge
        smooth: Boolean value representing whether the boundary being merged is X type (rough) or Z type (smooth)
        """
        self._num_qubits = num_qubits
        self._smooth = smooth

    def num_qubits(self):
        return self._num_qubits

    @property
    def smooth(self):
        return self._smooth

    def __str__(self):
        return "MERGE"

    def _json_dict_(self):
        return {"num_qubits": self._num_qubits, "smooth": self._smooth}
        # return cirq.obj_to_dict_helper(self, ["num_qubits", "smooth"])

    def __repr__(self) -> str:
        return f"lsp.Merge(num_qubits={self._num_qubits}, smooth={self._smooth})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[int, bool]:
        return self._num_qubits, self._smooth


@cirq.value_equality
class Split(cirq.Gate):
    """
    Subclassed cirq gate to represent the Split operation in lattice surgery.
    The Split operation turns a surface code patch into several distinct surface code patches by measuring the boundary qubits.
    See https://arxiv.org/pdf/1111.4022 for more information.
    This version of split assumes that there are a number of underlying well-defined qubits, ensuring we always split along known boundaries.

    partions: list of indices upon which to split
    smooth: Boolean value representing whether the boundary is getting an X type (rough) or Z type (smooth) measurement.

    Spilt([1, 3, 2]).on([X, Y, Z, P, Q , R]) --> [X], [Y, Z, P], [Q, R]
    """

    def __init__(self, partitions: list[int], smooth=True):
        self._num_qubits = sum(partitions)
        self._partitions = partitions
        self._smooth = smooth

    def num_qubits(self):
        return self._num_qubits

    @property
    def smooth(self):
        return self._smooth

    @property
    def partitions(self):
        return self._partitions

    def __str__(self):
        return f"SPLIT"

    def _json_dict_(self):
        return {"smooth": self._smooth, "partitions": self._partitions}

    def __repr__(self) -> str:
        return f"lsp.Split(partitions={self._partitions}, smooth={self._smooth})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[list[int], bool]:
        return *self._partitions, self._smooth


@cirq.value_equality
class SyndromeExtract(cirq.Gate):  # For now we are sort of ignoring the "buffer" physical qubits
    """
    Subclassed cirq gate to represent the process of measuring the stabilizers of surface code patch.
    This gate is treated as a single logical qubit operation, and ignores the buffer physical qubits that live between code patches to facilitate merge and split operations.

    num_qubits: Number of logical qubits being stabilized
    """

    # TODO: Should this be limited to a single qubit gate?
    def __init__(self, num_qubits, rounds):
        self._num_qubits = num_qubits
        self._rounds = rounds

    def _num_qubits_(self):
        return self._num_qubits

    @property
    def rounds(self):
        return self._rounds

    def __str__(self):
        return f"SE({self.rounds})"

    def _json_dict_(self):
        return {"num_qubits": self._num_qubits, "rounds": self._rounds}

    def __repr__(self) -> str:
        return f"lsp.SyndromeExtract(num_qubits={self._num_qubits}, rounds={self._rounds})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[int, int]:
        return self._num_qubits, self._rounds


@cirq.value_equality
class ErrorCorrect(cirq.Gate):
    """
    Subclassed cirq gate to represent the correction part of the error correction cycle.
    In a proper implementation this gate might have both digital bookkeeping and physical correction components to it.
    For the purposes of resource estimation, we leave it as a pretty bare-bones gate.
    It should always follow a SyndromeExtract gate.

    num_qubits: Number of logical qubits being corrected
    """

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def _num_qubits_(self):
        return self._num_qubits

    def __str__(self):
        return "ERROR CORRECT"

    def _json_dict_(self):
        return {"num_qubits": self._num_qubits}

    def __repr__(self) -> str:
        return f"lsp.ErrorCorrect(num_qubits={self._num_qubits})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> int:
        return self._num_qubits


@cirq.value_equality
class Cultivate(cirq.Gate):
    """
    Subclassed cirq gate to represent the cultivation of a single magic state on single code patch.
    The underlying implementation is assumed to be the one in https://arxiv.org/pdf/2409.17595, and is treated as single qubit gate.

    theta: The angle for the magic state to be prepared.

    Cultivate(θ)|0> --> (|0> + e^(iθ)|1>)/√2
    """

    def __init__(self, theta):
        self._theta = theta

    @property
    def theta(self):
        return self._theta

    def num_qubits(self):
        return 1

    def __str__(self):
        return f"CULT({round(self.theta, 3)})"

    def _json_dict_(self):
        return {"theta": self._theta}

    def __repr__(self) -> str:
        return f"lsp.Cultivate(theta={self._theta})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> int:
        return self._theta


@cirq.value_equality
class Move(cirq.Gate):
    """
    Subclassed cirq gate to represent a iter-patch movement operation

    It is currently used to describe both movement to a zone and movement through alleyways to other
    logical qubit patches.
    """

    def __init__(self, zone: Literal[None, "measure", "interact"] = None):
        self._num_qubits = 2 if zone is None else 1
        self._zone = zone

    def num_qubits(self):
        return self._num_qubits

    @property
    def zone(self):
        return self._zone

    def __str__(self):
        if self.zone is None:
            return "MOVE"
        else:
            return "MOVE_MZ" if self.zone == "measure" else "MOVE_IZ"

    def _json_dict_(self):
        return {"zone": self._zone}

    def __repr__(self) -> str:
        return f"lsp.Move(zone={self._zone})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> int:
        return self._num_qubits, self._zone


class RotatedCodePatch:
    """
    Extremely rough implementation of the rotated surface code.
    Assumed to be square patches.

    d: Code distance defining the surface code patch

    d = 3 CodePatch
          m
        d   d   d
          m   m   m
        d   d   d
      m   m   m
        d   d   d
              m

    d = 5 CodePatch
          m       m
        d   d   d   d   d
          m   m   m   m   m
        d   d   d   d   d
      m   m   m   m   m
        d   d   d   d   d
          m   m   m   m   m
        d   d   d   d   d
      m   m   m   m   m
        d   d   d   d   d
              m       m
    """

    def __init__(self, d: int):
        assert (d - 1) % 2 == 0, "CodePatches must be odd distance"
        self.d = d
        self.rows = 2 * d - 1
        self.cols = 2 * d - 1
        self.num_physical_qubits = 2 * (d**2) - 1

    @cached_property
    def num_data_qubits(self):
        """
        The number of data qubits in surface code patch
        """
        return self.d**2

    @cached_property
    def num_measure_qubits(self):
        """
        The number of measure qubits in a surface code patch
        """
        return self.d**2 - 1

    def num_z_stabs(self, full=True):  # Still assuming square lattice
        """
        The number of Z-type stabilizers in the patch.
        The full flag determines whether to count the complete plaquettes or the incomplete ones.
        Incomplete plaquettes have different costs in terms of resource estimation.
        """
        if full:
            return (self.d - 1) ** 2 // 2
        else:
            return self.d - 1

    def num_x_stabs(self, full=True):  # Still assuming square lattice here
        """
        The number of X-type stabilizers in the patch (should be same as Z)
        """
        if full:
            return (self.d - 1) ** 2 // 2
        else:
            return self.d - 1

    def total_x_syndrome_cnots(self):
        """
        The total number of CNOT parity checks incurred when measuring all X stabilizers.
        """
        return 4 * self.num_x_stabs(full=True) + 2 * self.num_x_stabs(full=False)

    def total_z_syndrome_cnots(self):
        """
        The total number of CNOT parity checks incurred when measuring all Z stabilizers.
        """
        return 4 * self.num_z_stabs(full=True) + 2 * self.num_z_stabs(full=False)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, RotatedCodePatch) and (self.d == value.d)

    def __hash__(self) -> int:
        return hash(self.d)


class BufferCodePatch(RotatedCodePatch):
    """
    2 x d buffer zone formed between qubit patches
    Includes two partial X stabilizers if the merge is smooth, else two partial Z stabilizers
    """

    def __init__(self, d: int, smooth: bool):
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full=True) -> int:
        if full:
            return self.d - 1
        if self.smooth:
            return 2
        else:
            return 0

    def num_z_stabs(self, full=True) -> int:
        if full:
            return self.d - 1
        if self.smooth:
            return 0
        return 2

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, BufferCodePatch) and self.smooth == value.smooth and self.d == value.d
        )


class IntermediatePatch(RotatedCodePatch):
    """
    (d - 1) x  (d - 1) patch formed between distant patches during a merge operation
    Has the X partial stabilizers of a full patch if smooth else the Z partial stabilizers from a full patch
    """

    def __init__(self, d: int, smooth=True):
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full=True) -> int:
        if full:
            return super().num_x_stabs(full=True)
        if self.smooth:
            return super().num_x_stabs(full=False)
        return 0

    def num_z_stabs(self, full=True) -> int:
        if full:
            return super().num_z_stabs(full=True)
        if self.smooth:
            return 0
        return super().num_z_stabs(full=False)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, IntermediatePatch)
            and self.smooth == value.smooth
            and self.d == value.d
        )

    # TODO: Overwrite other methods


class EndpointPatch(RotatedCodePatch):
    """
    (d - 1) x (d - 1) patch at the endpoints of a merge operation
    Looks like a normal rotated code patch with three 'flaps' instead of four
    If the merge is smooth, the flaps are X stabilizers else Z
    """

    def __init__(self, d: int, smooth=True):
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full=True) -> int:
        if full:
            return super().num_x_stabs(full=True)
        if self.smooth:
            return super().num_x_stabs(full=False)
        return super().num_x_stabs(full=False) // 2  # 1 set of 'flaps' instead of 2

    def num_z_stabs(self, full=True) -> int:
        if full:
            return super().num_z_stabs(full=True)
        if self.smooth:
            return super().num_z_stabs(full=False) // 2  # 1 set of 'flaps' instead of 2
        return super().num_z_stabs(full=False)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, EndpointPatch) and self.smooth == value.smooth and self.d == value.d
        )
