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
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import cached_property
from typing import Any, Literal, Optional, cast
import warnings

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
    """Subclassed cirq gate to represent the distillation of a single T state using 16 code patches.
    The underlying implementation is assumed to be the one in https://arxiv.org/abs/quant-ph/0403025.
    Noisy T gates are assumed to come from cultivation, resulting in 15 additional logical patches.


    Distil|0^31> --> (|0> + e^(1j*pi/4)|1>)/√2 |0^30>

    """

    def __init__(self) -> None:
        pass

    def num_qubits(self) -> int:
        return 31

    def __str__(self) -> str:
        return "DISTIL"

    def _json_dict_(self) -> dict[str, object]:
        return {}

    def __repr__(self) -> str:
        return "lsp.Distil()"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[()]:
        return ()


@cirq.value_equality
class Move(cirq.Gate):
    """Subclassed cirq gate to represent a iter-patch movement operation

    It is currently used to describe both movement to a zone and movement through alleyways to other
    logical qubit patches.
    """

    def __init__(self, zone: Optional[Literal["measure", "interact"]] = None) -> None:
        self._num_qubits = 2 if zone is None else 1
        self._zone = zone

    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def zone(self) -> Literal["interact", "measure"] | None:
        return self._zone

    def __str__(self) -> str:
        if self.zone is None:
            return "MOVE"
        return "MOVE_MZ" if self.zone == "measure" else "MOVE_IZ"

    def _json_dict_(self) -> dict[str, Literal["interact", "measure"] | None]:
        return {"zone": self._zone}

    def __repr__(self) -> str:
        return f"lsp.Move(zone={self._zone})"

    @classmethod
    def _json_namespace_(cls) -> str:
        return "lsp"

    def _value_equality_values_(self) -> tuple[int, str | None]:
        return (self._num_qubits, self._zone)


_SURFACE_CODE_TYPES = {
    "surface",
    "surface_code",
    "rotated_surface",
    "rotated_surface_code",
}

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

PatchLabel = Literal["memory", "compute", "cultivate", "distil"]
_PATCH_LABELS = {"memory", "compute", "cultivate", "distil"}
DistilleryLabel = Literal["CCZ", "T"]
_DISTILLERY_LABELS = {"CCZ", "T"}
_DISTILLERY_PATCH_COUNTS: dict[DistilleryLabel, int] = {"CCZ": 9, "T": 11}
VaultLabel = Literal["T:cultivated", "T:distilled", "CCZ:distilled"]
_VAULT_LABELS = {"T:cultivated", "T:distilled", "CCZ:distilled"}
_VAULT_SHYPS_R = 3


def _normalize_code_type(code_type: str) -> str:
    return code_type.lower().replace("-", "_").replace(" ", "_")


def _import_qldpc() -> tuple[Any, Any, Any]:
    try:
        from qldpc import circuits, codes
        from qldpc.objects import Pauli
    except ImportError as ex:  # pragma: no cover - exercised only when qLDPC is absent
        raise ImportError(
            "qLDPC-backed CodePatch objects require the optional `qldpc` package."
        ) from ex
    return circuits, codes, Pauli


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


def _run_without_external_prompt(function: Callable[[], Any]) -> Any:
    """Run qLDPC code that may otherwise request manual GAP/MAGMA input."""
    import builtins

    original_input = builtins.input
    builtins.input = lambda prompt="": "n"
    try:
        return function()
    finally:
        builtins.input = original_input


def _validate_patch_label(patch_label: str) -> PatchLabel:
    if patch_label not in _PATCH_LABELS:
        raise ValueError(f"Patch label must be one of {sorted(_PATCH_LABELS)}, not {patch_label!r}")
    return cast(PatchLabel, patch_label)


def _validate_distillery_label(label: str) -> DistilleryLabel:
    if label not in _DISTILLERY_LABELS:
        raise ValueError(
            f"Distillery label must be one of {sorted(_DISTILLERY_LABELS)}, not {label!r}"
        )
    return cast(DistilleryLabel, label)


def _validate_vault_label(label: str) -> VaultLabel:
    if label not in _VAULT_LABELS:
        raise ValueError(f"Vault label must be one of {sorted(_VAULT_LABELS)}, not {label!r}")
    return cast(VaultLabel, label)


def _default_vault_code_patch() -> CodePatch:
    return CodePatch.from_qldpc_family(
        "shyps",
        _VAULT_SHYPS_R,
        patch_label="memory",
    )


class CodePatch:
    """Metadata for a logical code patch.

    A CodePatch represents a code block together with the resource-estimation metadata needed to
    reason about its size and intended use. qLDPC-backed patches retain the qLDPC code object so
    callers can request logical operators and transversal operations when needed. The patch label
    indicates whether the patch is intended for memory, computation, cultivation, or distillation.
    """

    def __init__(
        self,
        code_type: str = "surface",
        d: int | None = 7,
        *,
        n: int | None = None,
        k: int | None = None,
        num_data_qubits: int | None = None,
        num_measure_qubits: int | None = None,
        patch_label: PatchLabel = "compute",
        qldpc_code: Any | None = None,
        qldpc_family: str | None = None,
        qldpc_args: tuple[object, ...] | None = None,
        qldpc_kwargs: Mapping[str, object] | None = None,
        distance_bound: int | bool | None = None,
        distance_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.code_type = code_type
        self.patch_label = _validate_patch_label(patch_label)
        self.qldpc_family = qldpc_family
        self.qldpc_args = tuple(qldpc_args or ())
        self.qldpc_kwargs = dict(qldpc_kwargs or {})
        self._qldpc_code = qldpc_code

        if qldpc_code is not None:
            metadata = self._metadata_from_qldpc_code(
                qldpc_code,
                d=d,
                distance_bound=distance_bound,
                distance_kwargs=distance_kwargs,
            )
            n = n if n is not None else metadata["n"]
            k = k if k is not None else metadata["k"]
            d = d if d is not None else metadata["d"]
            num_data_qubits = (
                num_data_qubits
                if num_data_qubits is not None
                else metadata["num_data_qubits"]
            )
            num_measure_qubits = (
                num_measure_qubits
                if num_measure_qubits is not None
                else metadata["num_measure_qubits"]
            )

        if n is None and num_data_qubits is not None:
            n = num_data_qubits

        if n is None and _normalize_code_type(code_type) in _SURFACE_CODE_TYPES and d is not None:
            n = d**2
            k = 1 if k is None else k
            num_data_qubits = n if num_data_qubits is None else num_data_qubits
            num_measure_qubits = d**2 - 1 if num_measure_qubits is None else num_measure_qubits

        if n is None:
            raise ValueError(
                "CodePatch requires a qLDPC code/family, an implemented built-in code type, "
                "or explicit n/num_data_qubits metadata."
            )

        if k is None:
            k = 1
        if num_data_qubits is None:
            num_data_qubits = n
        if num_measure_qubits is None:
            num_measure_qubits = 0

        self.d = d
        self.n = n
        self.k = k
        self.num_data_qubits = num_data_qubits
        self.num_measure_qubits = num_measure_qubits

    @classmethod
    def from_qldpc_code(
        cls,
        qldpc_code: Any,
        *,
        code_type: str | None = None,
        d: int | None = None,
        patch_label: PatchLabel = "compute",
        qldpc_family: str | None = None,
        qldpc_args: tuple[object, ...] | None = None,
        qldpc_kwargs: Mapping[str, object] | None = None,
        distance_bound: int | bool | None = None,
        distance_kwargs: Mapping[str, object] | None = None,
    ) -> CodePatch:
        """Build a CodePatch from an already constructed qLDPC code."""
        if code_type is None:
            code_type = type(qldpc_code).__name__
        return cls(
            code_type=code_type,
            d=d,
            patch_label=patch_label,
            qldpc_code=qldpc_code,
            qldpc_family=qldpc_family,
            qldpc_args=qldpc_args,
            qldpc_kwargs=qldpc_kwargs,
            distance_bound=distance_bound,
            distance_kwargs=distance_kwargs,
        )

    @classmethod
    def from_qldpc_family(
        cls,
        code_type: str | Callable[..., object],
        *code_args: object,
        d: int | None = None,
        patch_label: PatchLabel = "compute",
        distance_bound: int | bool | None = None,
        distance_kwargs: Mapping[str, object] | None = None,
        **code_kwargs: object,
    ) -> CodePatch:
        """Build a CodePatch from a qLDPC code family and family-specific constructor data."""
        _, codes, _ = _import_qldpc()

        if callable(code_type):
            qldpc_family = getattr(code_type, "__name__", repr(code_type))
            code_factory = code_type
            patch_code_type = qldpc_family
        else:
            qldpc_family = _resolve_qldpc_family_name(code_type, codes)
            code_factory = getattr(codes, qldpc_family)
            patch_code_type = code_type

        qldpc_args = tuple(code_args)
        if not qldpc_args and d is not None and qldpc_family in _QLDPC_DISTANCE_FAMILIES:
            qldpc_args = (d,)

        qldpc_code = code_factory(*qldpc_args, **code_kwargs)
        return cls.from_qldpc_code(
            qldpc_code,
            code_type=patch_code_type,
            d=d,
            patch_label=patch_label,
            qldpc_family=qldpc_family,
            qldpc_args=qldpc_args,
            qldpc_kwargs=code_kwargs,
            distance_bound=distance_bound,
            distance_kwargs=distance_kwargs,
        )

    @classmethod
    def from_qldpc_factory(
        cls,
        code_type: str,
        code_factory: Callable[..., object],
        *code_args: object,
        d: int | None = None,
        patch_label: PatchLabel = "compute",
        distance_bound: int | bool | None = None,
        distance_kwargs: Mapping[str, object] | None = None,
        **code_kwargs: object,
    ) -> CodePatch:
        """Build a CodePatch from a caller-provided qLDPC code factory."""
        qldpc_code = code_factory(*code_args, **code_kwargs)
        return cls.from_qldpc_code(
            qldpc_code,
            code_type=code_type,
            d=d,
            patch_label=patch_label,
            qldpc_family=getattr(code_factory, "__name__", repr(code_factory)),
            qldpc_args=tuple(code_args),
            qldpc_kwargs=code_kwargs,
            distance_bound=distance_bound,
            distance_kwargs=distance_kwargs,
        )

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def is_qldpc_backed(self) -> bool:
        return self._qldpc_code is not None

    @property
    def qldpc_code(self) -> Any:
        if self._qldpc_code is None:
            raise ValueError("This CodePatch is not backed by a qLDPC code object.")
        return self._qldpc_code

    def logical_ops(self, pauli: str | object | None = None, *, recompute: bool = False) -> Any:
        """Return qLDPC logical Pauli operators for this patch."""
        _, _, Pauli = _import_qldpc()
        if isinstance(pauli, str):
            pauli = Pauli.from_string(pauli.upper())
        return self.qldpc_code.get_logical_ops(pauli, recompute=recompute)

    def transversal_ops(
        self,
        local_gates: Collection[str] = ("S", "H", "SWAP"),
        *,
        deform_code: bool = False,
        remove_redundancies: bool = True,
        with_magma: bool = False,
        allow_external_prompt: bool = False,
    ) -> list[Any]:
        """Return qLDPC logical tableaus and physical circuits for transversal operations."""
        circuits, _, _ = _import_qldpc()

        def _call() -> list[Any]:
            return circuits.get_transversal_ops(
                self.qldpc_code,
                local_gates,
                deform_code=deform_code,
                remove_redundancies=remove_redundancies,
                with_magma=with_magma,
            )

        if allow_external_prompt:
            return _call()
        return _run_without_external_prompt(_call)

    def transversal_circuit(
        self,
        logical_circuit_or_tableau: Any,
        local_gates: Collection[str] = ("S", "H", "SWAP"),
        *,
        deform_code: bool = False,
        with_magma: bool = False,
        allow_external_prompt: bool = False,
    ) -> Any | None:
        """Find a qLDPC transversal physical circuit for a logical Clifford operation."""
        circuits, _, _ = _import_qldpc()

        def _call() -> Any | None:
            return circuits.get_transversal_circuit(
                self.qldpc_code,
                logical_circuit_or_tableau,
                local_gates,
                deform_code=deform_code,
                with_magma=with_magma,
            )

        if allow_external_prompt:
            return _call()
        return _run_without_external_prompt(_call)

    @staticmethod
    def _metadata_from_qldpc_code(
        qldpc_code: Any,
        *,
        d: int | None,
        distance_bound: int | bool | None,
        distance_kwargs: Mapping[str, object] | None,
    ) -> dict[str, Any]:
        n = int(len(qldpc_code))
        k = int(qldpc_code.dimension)
        num_measure_qubits = int(getattr(qldpc_code, "num_checks", 0))
        known_distance = CodePatch._known_qldpc_distance(qldpc_code)

        if d is not None and known_distance is not None and known_distance != d:
            raise ValueError(
                f"Provided distance d={d} does not match qLDPC code distance "
                f"{known_distance}."
            )

        code_distance: int | float | None
        if d is not None:
            code_distance = d
        elif known_distance is not None:
            code_distance = known_distance
        elif distance_bound is not None:
            warnings.warn(
                "Computing qLDPC code distance because it is not known. This may be expensive.",
                RuntimeWarning,
                stacklevel=2,
            )
            code_distance = qldpc_code.get_distance(
                bound=distance_bound, **dict(distance_kwargs or {})
            )
        else:
            warnings.warn(
                "Computing qLDPC code distance because it is not known. This may be expensive.",
                RuntimeWarning,
                stacklevel=2,
            )
            code_distance = qldpc_code.get_distance(**dict(distance_kwargs or {}))

        return {
            "n": n,
            "k": k,
            "d": code_distance,
            "num_data_qubits": n,
            "num_measure_qubits": num_measure_qubits,
        }

    @staticmethod
    def _known_qldpc_distance(qldpc_code: Any) -> int | float | None:
        try:
            known_distance = qldpc_code.get_distance_if_known()
        except AttributeError:
            return None
        if known_distance is None or known_distance != known_distance:
            return None
        if isinstance(known_distance, float) and known_distance.is_integer():
            return int(known_distance)
        return known_distance

    def __repr__(self) -> str:
        args = [
            f"code_type={self.code_type!r}",
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
            f"patch_label={self.patch_label!r}",
        ]
        return f"lsp.CodePatch({', '.join(args)})"


class Farm:
    """Container for cultivation code patches."""

    def __init__(self, code_patches: int | Iterable[CodePatch] = ()) -> None:
        self.code_patches: list[CodePatch] = []
        if isinstance(code_patches, int):
            if code_patches < 0:
                raise ValueError("Farm patch count must be nonnegative.")
            for _ in range(code_patches):
                self.add_patch(CodePatch(code_type="surface", patch_label="cultivate"))
            return
        for patch in code_patches:
            self.add_patch(patch)

    @property
    def num_physical_qubits(self) -> int:
        return sum(patch.n for patch in self.code_patches)

    @property
    def num_logical_qubits(self) -> int:
        return sum(patch.k for patch in self.code_patches)

    @property
    def num_code_patches(self) -> int:
        return len(self.code_patches)

    def add_patch(self, patch: CodePatch) -> None:
        if patch.patch_label != "cultivate":
            raise ValueError(
                "Farm can only contain CodePatch objects with patch_label='cultivate'."
            )
        self.code_patches.append(patch)


class Distillery:
    """Container for distillation code patches."""

    def __init__(self, label: DistilleryLabel) -> None:
        self.label = _validate_distillery_label(label)
        self.code_patches = [
            CodePatch(code_type="surface", patch_label="distil")
            for _ in range(_DISTILLERY_PATCH_COUNTS[self.label])
        ]

    @property
    def num_physical_qubits(self) -> int:
        return sum(patch.n for patch in self.code_patches)

    @property
    def num_logical_qubits(self) -> int:
        return sum(patch.k for patch in self.code_patches)

    @property
    def num_code_patches(self) -> int:
        return len(self.code_patches)


class Factory:
    """Container for CCZ and T distilleries."""

    def __init__(
        self,
        ccz_distilleries: int | Iterable[Distillery] = 0,
        t_distilleries: int | Iterable[Distillery] = 0,
    ) -> None:
        self.ccz_distilleries: list[Distillery] = []
        self.t_distilleries: list[Distillery] = []
        self._add_distilleries(ccz_distilleries, "CCZ")
        self._add_distilleries(t_distilleries, "T")

    @property
    def num_ccz_distilleries(self) -> int:
        return len(self.ccz_distilleries)

    @property
    def num_t_distilleries(self) -> int:
        return len(self.t_distilleries)

    @property
    def num_distilleries(self) -> int:
        return self.num_ccz_distilleries + self.num_t_distilleries

    @property
    def num_physical_qubits(self) -> int:
        return sum(
            distillery.num_physical_qubits
            for distillery in self.ccz_distilleries + self.t_distilleries
        )

    @property
    def num_logical_qubits(self) -> int:
        return sum(
            distillery.num_logical_qubits
            for distillery in self.ccz_distilleries + self.t_distilleries
        )

    def add_ccz_distillery(self, distillery: Distillery) -> None:
        self._add_distillery(distillery, "CCZ")

    def add_t_distillery(self, distillery: Distillery) -> None:
        self._add_distillery(distillery, "T")

    def _add_distilleries(
        self,
        distilleries: int | Iterable[Distillery],
        label: DistilleryLabel,
    ) -> None:
        if isinstance(distilleries, int):
            if distilleries < 0:
                raise ValueError("Factory distillery count must be nonnegative.")
            for _ in range(distilleries):
                self._add_distillery(Distillery(label), label)
            return
        for distillery in distilleries:
            self._add_distillery(distillery, label)

    def _add_distillery(self, distillery: Distillery, label: DistilleryLabel) -> None:
        if not isinstance(distillery, Distillery):
            raise TypeError("Factory can only contain Distillery objects.")
        if distillery.label != label:
            raise ValueError(f"Expected a {label} Distillery, not {distillery.label!r}.")
        if label == "CCZ":
            self.ccz_distilleries.append(distillery)
        else:
            self.t_distilleries.append(distillery)


class Vault:
    """Container for memory patches that store resource states."""

    def __init__(
        self,
        label: VaultLabel,
        code_patches: int | Iterable[CodePatch] = (),
    ) -> None:
        self.label = _validate_vault_label(label)
        self.code_patches: list[CodePatch] = []
        if isinstance(code_patches, int):
            if code_patches < 0:
                raise ValueError("Vault patch count must be nonnegative.")
            for _ in range(code_patches):
                self.add_patch(_default_vault_code_patch())
            return
        for patch in code_patches:
            self.add_patch(patch)

    @property
    def num_physical_qubits(self) -> int:
        return sum(patch.n for patch in self.code_patches)

    @property
    def num_logical_qubits(self) -> int:
        return sum(patch.k for patch in self.code_patches)

    @property
    def num_code_patches(self) -> int:
        return len(self.code_patches)

    def add_patch(self, patch: CodePatch) -> None:
        if not isinstance(patch, CodePatch):
            raise TypeError("Vault can only contain CodePatch objects.")
        if patch.patch_label != "memory":
            raise ValueError(
                "Vault can only contain CodePatch objects with patch_label='memory'."
            )
        self.code_patches.append(patch)


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
