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

import typing
from functools import cached_property
from typing import Any, Callable, Iterable, Literal, cast

import cirq

# TODO: Add cirq diagram info


def custom_resolver(cirq_type: str) -> type[cirq.Gate] | None:
    """Tells cirq.json how to deserialize custom gates"""
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
    if cirq_type == "lsp.Distil":
        return Distil


@cirq.value_equality
class Merge(cirq.Gate):
    def __init__(self, num_qubits: int, smooth: bool = True) -> None:
        """Subclassed cirq gate to represent the Merge operation in lattice surgery.
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

    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def smooth(self) -> bool:
        return self._smooth

    def __str__(self) -> str:
        return "MERGE"

    def _json_dict_(self) -> dict[str, bool | int]:
        return {"num_qubits": self._num_qubits, "smooth": self._smooth}

    def __repr__(self) -> str:
        return f"lsp.Merge(num_qubits={self._num_qubits}, smooth={self._smooth})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[int, bool]:
        return self._num_qubits, self._smooth


@cirq.value_equality
class Split(cirq.Gate):
    """Subclassed cirq gate to represent the Split operation in lattice surgery.
    The Split operation turns a surface code patch into several distinct surface code patches by measuring the boundary qubits.
    See https://arxiv.org/pdf/1111.4022 for more information.
    This version of split assumes that there are a number of underlying well-defined qubits, ensuring we always split along known boundaries.

    partions: list of indices upon which to split
    smooth: Boolean value representing whether the boundary is getting an X type (rough) or Z type (smooth) measurement.

    Spilt([1, 3, 2]).on([X, Y, Z, P, Q , R]) --> [X], [Y, Z, P], [Q, R]
    """

    def __init__(self, partitions: list[int], smooth: bool = True) -> None:
        self._num_qubits = sum(partitions)
        self._partitions = partitions
        self._smooth = smooth

    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def smooth(self) -> bool:
        return self._smooth

    @property
    def partitions(self) -> list[int]:
        return self._partitions

    def __str__(self) -> str:
        return "SPLIT"

    def _json_dict_(self) -> dict[str, bool | list[int]]:
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
    """Subclassed cirq gate to represent the process of measuring the stabilizers of surface code patch.
    This gate is treated as a single logical qubit operation, and ignores the buffer physical qubits that live between code patches to facilitate merge and split operations.

    num_qubits: Number of logical qubits being stabilized
    """

    # TODO: Should this be limited to a single qubit gate?
    def __init__(self, num_qubits, rounds) -> None:
        self._num_qubits = num_qubits
        self._rounds = rounds

    def _num_qubits_(self) -> int:
        return self._num_qubits

    @property
    def rounds(self) -> int:
        return self._rounds

    def __str__(self) -> str:
        return f"SE({self.rounds})"

    def _json_dict_(self) -> dict[str, bool | int]:
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
    """Subclassed cirq gate to represent the correction part of the error correction cycle.
    In a proper implementation this gate might have both digital bookkeeping and physical correction components to it.
    For the purposes of resource estimation, we leave it as a pretty bare-bones gate.
    It should always follow a SyndromeExtract gate.

    num_qubits: Number of logical qubits being corrected
    """

    def __init__(self, num_qubits) -> None:
        self._num_qubits = num_qubits

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def __str__(self) -> str:
        return "ERROR CORRECT"

    def _json_dict_(self) -> dict[str, int]:
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
    """Subclassed cirq gate to represent the cultivation of a single magic state on single code patch.
    The underlying implementation is assumed to be the one in https://arxiv.org/pdf/2409.17595, and is treated as single qubit gate.

    theta: The angle for the magic state to be prepared.

    Cultivate(θ)|0> --> (|0> + e^(iθ)|1>)/√2
    """

    def __init__(self, theta: float) -> None:
        self._theta = theta

    @property
    def theta(self) -> float:
        return self._theta

    def num_qubits(self) -> int:
        return 1

    def __str__(self) -> str:
        return f"CULT({round(self.theta, 3)})"

    def _json_dict_(self) -> dict[str, float]:
        return {"theta": self._theta}

    def __repr__(self) -> str:
        return f"lsp.Cultivate(theta={self._theta})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> float:
        return self._theta


@cirq.value_equality
class Distil(cirq.Gate):
    """Subclassed cirq gate to represent the distillation of a resource state.
    T leads to a single T state using 16 code patches.
    The underlying implementation is assumed to be the one in https://arxiv.org/abs/quant-ph/0403025.
    Noisy T gates are assumed to come from cultivation, resulting in 15 additional logical patches.
    Distil|0^31> --> (|0> + e^(1j*pi/4)|1>)/√2 |0^30>

    CCZ leads to a CCZ state
    """

    def __init__(self, resource: Literal["T", "CCZ"]) -> None:
        if resource not in ("T", "CCZ"):
            raise ValueError(f"Invalid resource for Distil gate: {resource!r}")
        self._resource = resource
        self._num_qubits = 23 if resource == "CCZ" else 31

    def num_qubits(self) -> int:
        return self._num_qubits

    def __str__(self) -> str:
        return f"DISTIL({self._resource})"

    def _json_dict_(self) -> dict[str, object]:
        return {"resource": self._resource}

    def __repr__(self) -> str:
        return f"lsp.Distil({self._resource})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> str:
        return self._resource


@cirq.value_equality
class Move(cirq.Gate):
    """Subclassed cirq gate to represent a iter-patch movement operation

    It is currently used to describe both movement to a zone and movement through alleyways to other
    logical qubit patches.
    """

    def __init__(self, zone: typing.Optional[typing.Literal["measure", "interact"]] = None) -> None:
        self._num_qubits = 2 if zone is None else 1
        self._zone = zone

    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def zone(self) -> typing.Literal["interact", "measure"] | None:
        return self._zone

    def __str__(self) -> str:
        if self.zone is None:
            return "MOVE"
        return "MOVE_MZ" if self.zone == "measure" else "MOVE_IZ"

    def _json_dict_(self) -> dict[str, typing.Literal["interact", "measure"] | None]:
        return {"zone": self._zone}

    def __repr__(self) -> str:
        return f"lsp.Move(zone={self._zone})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[int, str | None]:
        return (self._num_qubits, self._zone)


_QLDPC_FAMILY_ALIASES = {
    "surface": "SurfaceCode",
    "surface_code": "SurfaceCode",
    "rotated_surface": "SurfaceCode",
    "rotated_surface_code": "SurfaceCode",
    "toric": "ToricCode",
    "toric_code": "ToricCode",
    "five_qubit": "FiveQubitCode",
    "five_qubit_code": "FiveQubitCode",
    "steane": "SteaneCode",
    "steane_code": "SteaneCode",
    "bb": "BBCode",
    "bb_code": "BBCode",
    "bivariate_bicycle": "BBCode",
    "bivariate_bicycle_code": "BBCode",
    "hgp": "HGPCode",
    "hgps": "HGPCode",
    "hypergraph_product": "HGPCode",
    "hypergraph_product_code": "HGPCode",
    "shyps": "SHYPSCode",
    "shyps_code": "SHYPSCode",
}

_QLDPC_DISTANCE_FAMILIES = {"SurfaceCode", "ToricCode"}

LogicalQubitLabel = Literal["zero", "one", "plus", "minus", "data"]
_LOGICAL_QUBIT_LABELS = {"zero", "one", "plus", "minus", "data"}
PatchLabel = Literal["memory", "compute", "cultivate", "distil"]
_PATCH_LABELS = {"memory", "compute", "cultivate", "distil"}


def _normalize_code_type(code_type: str) -> str:
    return code_type.lower().replace("-", "_").replace(" ", "_")


def _import_qldpc() -> Any:
    try:
        from qldpc import codes
    except ImportError as ex:  # pragma: no cover - exercised only when qLDPC is absent
        raise ImportError(
            "qLDPC-backed CodePatch objects require the optional `qldpc` package."
        ) from ex
    return codes


def _resolve_qldpc_family_name(code_type: str, codes_module: Any) -> str:
    normalized = _normalize_code_type(code_type)
    if normalized in _QLDPC_FAMILY_ALIASES:
        return _QLDPC_FAMILY_ALIASES[normalized]
    if hasattr(codes_module, code_type):
        return code_type
    for name in getattr(codes_module, "__all__", ()):
        if _normalize_code_type(name) == normalized:
            return name
    raise ValueError(f"qLDPC code family not found for code_type={code_type!r}")


def _validate_logical_qubit_label(label: str) -> LogicalQubitLabel:
    if label not in _LOGICAL_QUBIT_LABELS:
        raise ValueError(
            f"Logical qubit label must be one of {sorted(_LOGICAL_QUBIT_LABELS)}, not {label!r}"
        )
    return cast(LogicalQubitLabel, label)


def _validate_patch_label(patch_label: str) -> PatchLabel:
    if patch_label not in _PATCH_LABELS:
        raise ValueError(f"Patch label must be one of {sorted(_PATCH_LABELS)}, not {patch_label!r}")
    return cast(PatchLabel, patch_label)


class LogicalQubit:
    """Metadata for a logical qubit primitive."""

    def __init__(
        self,
        x_support: Iterable[int],
        z_support: Iterable[int],
        label: LogicalQubitLabel = "zero",
    ) -> None:
        self.label = _validate_logical_qubit_label(label)
        self.x_support = self._validate_support(x_support, "x_support")
        self.z_support = self._validate_support(z_support, "z_support")
        self.num_qubits = len(self.x_support | self.z_support)
        if self.num_qubits < 1:
            raise ValueError("LogicalQubit supports must include at least one physical qubit.")

    @staticmethod
    def _validate_support(support: Iterable[int], name: str) -> frozenset[int]:
        qubits = set()
        for qubit in support:
            if not isinstance(qubit, int) or isinstance(qubit, bool):
                raise TypeError(f"LogicalQubit {name} entries must be integers.")
            if qubit < 0:
                raise ValueError(f"LogicalQubit {name} entries must be nonnegative.")
            qubits.add(qubit)
        return frozenset(qubits)


class CodePatch:
    """Metadata for a logical code patch.

    A CodePatch represents a qLDPC-backed code block together with the metadata needed
    to reason about its size and intended use. The patch label indicates whether the
    patch is intended for memory, computation, cultivation, or distillation.
    """

    def __init__(
        self,
        code_type: str | Callable[..., object],
        *code_args: object,
        d: int | None = None,
        patch_label: PatchLabel = "compute",
        **code_kwargs: object,
    ) -> None:
        codes = _import_qldpc()
        self.patch_label = _validate_patch_label(patch_label)

        if callable(code_type):
            code_factory = code_type
            self.code_type = getattr(code_type, "__name__", repr(code_type))
        else:
            self.code_type = code_type
            code_factory = getattr(codes, _resolve_qldpc_family_name(code_type, codes))

        qldpc_args = tuple(code_args)
        if (
            not qldpc_args
            and d is not None
            and getattr(code_factory, "__name__", None) in _QLDPC_DISTANCE_FAMILIES
        ):
            qldpc_args = (d,)

        self._qldpc_code = code_factory(*qldpc_args, **code_kwargs)
        self.n, self.k, self.d = self._metadata_from_qldpc_code(self._qldpc_code)
        if d is not None and self.d is not None and self.d != d:
            raise ValueError(
                f"Provided distance d={d} does not match qLDPC code distance {self.d}."
            )
        self.num_data_qubits = self.n
        self.num_measure_qubits = int(getattr(self._qldpc_code, "num_checks"))
        self.logical_qubits = self._logical_qubits_from_qldpc_code()

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def num_physical_qubits(self) -> int:
        """The total number of data and measurement qubits in the code patch."""
        return self.num_data_qubits + self.num_measure_qubits

    @property
    def is_qldpc_backed(self) -> bool:
        return True

    @property
    def qldpc_code(self) -> Any:
        return self._qldpc_code

    def num_x_stabs(self) -> int:
        """Return the number of X-type stabilizer checks for CSS qLDPC codes."""
        return self._qldpc_css_check_count("x")

    def num_z_stabs(self) -> int:
        """Return the number of Z-type stabilizer checks for CSS qLDPC codes."""
        return self._qldpc_css_check_count("z")

    def total_x_syndrome_cnots(self) -> int:
        """Return the data-check interactions needed to measure all X stabilizers."""
        return self._qldpc_css_interaction_count("x")

    def total_z_syndrome_cnots(self) -> int:
        """Return the data-check interactions needed to measure all Z stabilizers."""
        return self._qldpc_css_interaction_count("z")

    def _qldpc_css_check_count(self, pauli: Literal["x", "z"]) -> int:
        attr = f"num_checks_{pauli}"
        if hasattr(self.qldpc_code, attr):
            return int(getattr(self.qldpc_code, attr))

        matrix = getattr(self.qldpc_code, f"matrix_{pauli}", None)
        if matrix is not None and getattr(matrix, "shape", None) is not None:
            return int(matrix.shape[0])

        raise self._unsupported_stabilizer_count_error()

    def _qldpc_css_interaction_count(self, pauli: Literal["x", "z"]) -> int:
        matrix = getattr(self.qldpc_code, f"matrix_{pauli}", None)
        if matrix is not None and getattr(matrix, "shape", None) is not None:
            return len(matrix.nonzero()[0])

        raise self._unsupported_stabilizer_count_error()

    def _logical_qubits_from_qldpc_code(self) -> list[LogicalQubit]:
        from qldpc.objects import Pauli

        logical_xs = self.qldpc_code.get_logical_ops(Pauli.X)
        logical_zs = self.qldpc_code.get_logical_ops(Pauli.Z)
        self._validate_logical_ops_count(logical_xs, "X")
        self._validate_logical_ops_count(logical_zs, "Z")

        supports = [
            (
                self._logical_op_support(logical_xs[index]),
                self._logical_op_support(logical_zs[index]),
            )
            for index in range(self.k)
        ]
        support_sizes = [len(x_support | z_support) for x_support, z_support in supports]
        if len(set(support_sizes)) > 1:
            raise ValueError(
                "Logical qubit physical supports must have the same size within a CodePatch."
            )
        return [
            LogicalQubit(x_support=x_support, z_support=z_support, label="zero")
            for x_support, z_support in supports
        ]

    def _validate_logical_ops_count(self, logical_ops: Any, pauli: str) -> None:
        if len(logical_ops) != self.k:
            raise ValueError(
                f"qLDPC returned {len(logical_ops)} logical {pauli} operators for "
                f"a CodePatch with k={self.k}."
            )

    def _logical_op_support(self, logical_op: Any) -> set[int]:
        row = [int(value) for value in logical_op]
        width = len(row)
        if width == self.n:
            return {index for index, value in enumerate(row) if value}
        if width == 2 * self.n:
            return {index for index in range(self.n) if row[index] or row[index + self.n]}
        raise ValueError(
            f"Logical operator rows must have length n={self.n} or 2n={2 * self.n}, not {width}."
        )

    @staticmethod
    def _unsupported_stabilizer_count_error() -> ValueError:
        return ValueError(
            "X/Z stabilizer counts are only available for CSS qLDPC-backed CodePatch objects."
        )

    @staticmethod
    def _metadata_from_qldpc_code(qldpc_code: Any) -> tuple[int, int, int | float | None]:
        try:
            n, k, d = qldpc_code.get_code_params()
        except AttributeError:
            n = len(qldpc_code)
            k = qldpc_code.dimension
            d = qldpc_code.get_distance_if_known()
        return int(n), int(k), d

    def __repr__(self) -> str:
        args = [
            f"code_type={self.code_type!r}",
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
            f"patch_label={self.patch_label!r}",
        ]
        return f"lsp.CodePatch({', '.join(args)})"


class RotatedCodePatch:
    """Extremely rough implementation of the rotated surface code.
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

    def __init__(self, d: int) -> None:
        assert (d - 1) % 2 == 0, "CodePatches must be odd distance"
        self.d = d
        self.rows = 2 * d - 1
        self.cols = 2 * d - 1
        self.num_physical_qubits = 2 * (d**2) - 1

    @cached_property
    def num_data_qubits(self) -> int:
        """The number of data qubits in surface code patch"""
        return self.d**2

    @cached_property
    def num_measure_qubits(self) -> int:
        """The number of measure qubits in a surface code patch"""
        return self.d**2 - 1

    def num_z_stabs(self, full: bool = True) -> int:  # Still assuming square lattice
        """The number of Z-type stabilizers in the patch.
        The full flag determines whether to count the complete plaquettes or the incomplete ones.
        Incomplete plaquettes have different costs in terms of resource estimation.
        """
        if full:
            return (self.d - 1) ** 2 // 2
        return self.d - 1

    def num_x_stabs(self, full: bool = True) -> int:  # Still assuming square lattice here
        """The number of X-type stabilizers in the patch (should be same as Z)"""
        if full:
            return (self.d - 1) ** 2 // 2
        return self.d - 1

    def total_x_syndrome_cnots(self) -> int:
        """The total number of CNOT parity checks incurred when measuring all X stabilizers."""
        return 4 * self.num_x_stabs(full=True) + 2 * self.num_x_stabs(full=False)

    def total_z_syndrome_cnots(self) -> int:
        """The total number of CNOT parity checks incurred when measuring all Z stabilizers."""
        return 4 * self.num_z_stabs(full=True) + 2 * self.num_z_stabs(full=False)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, RotatedCodePatch) and (self.d == value.d)

    def __hash__(self) -> int:
        return hash(self.d)


class BufferCodePatch(RotatedCodePatch):
    """2 x d buffer zone formed between qubit patches
    Includes two partial X stabilizers if the merge is smooth, else two partial Z stabilizers
    """

    def __init__(self, d: int, smooth: bool) -> None:
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full: bool = True) -> int:
        if full:
            return self.d - 1
        if self.smooth:
            return 2
        return 0

    def num_z_stabs(self, full: bool = True) -> int:
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
    """(d - 1) x  (d - 1) patch formed between distant patches during a merge operation
    Has the X partial stabilizers of a full patch if smooth else the Z partial stabilizers from a full patch
    """

    def __init__(self, d: int, smooth: bool = True) -> None:
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full: bool = True) -> int:
        if full:
            return super().num_x_stabs(full=True)
        if self.smooth:
            return super().num_x_stabs(full=False)
        return 0

    def num_z_stabs(self, full: bool = True) -> int:
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
    """(d - 1) x (d - 1) patch at the endpoints of a merge operation
    Looks like a normal rotated code patch with three 'flaps' instead of four
    If the merge is smooth, the flaps are X stabilizers else Z
    """

    def __init__(self, d: int, smooth: bool = True) -> None:
        super().__init__(d=d)
        self.smooth = smooth

    def num_x_stabs(self, full: bool = True) -> int:
        if full:
            return super().num_x_stabs(full=True)
        if self.smooth:
            return super().num_x_stabs(full=False)
        return super().num_x_stabs(full=False) // 2  # 1 set of 'flaps' instead of 2

    def num_z_stabs(self, full: bool = True) -> int:
        if full:
            return super().num_z_stabs(full=True)
        if self.smooth:
            return super().num_z_stabs(full=False) // 2  # 1 set of 'flaps' instead of 2
        return super().num_z_stabs(full=False)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, EndpointPatch) and self.smooth == value.smooth and self.d == value.d
        )
