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
import re
import typing
from dataclasses import dataclass, field

import cirq
import numpy as np
import sympy
from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y


def _validate_id(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value

class PhysicalQubit(cirq.Qid):
    """A physical qubit represented as a wire in a Cirq circuit"""
    def __init__(self, patch_id: int, local_coord: tuple[int, int], role: tuple[typing.Literal['data', 'measure'], typing.Literal['X', 'Z'] | None], info: dict[str, int] | None = None) -> None:
        """
        patch_id: the patch id of the CodePatch object to which this physical qubit belongs
        local_coord: local coordinate tuple representing the position of this qubit in the footprint of the CodePatch
        role: Designation of the job of the physical qubit as a data or measure qubit
        """
        self._patch_id = _validate_id(patch_id, "patch_id")
        self._local_coord = local_coord
        self._role = role
        self._info = info

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    def local_coord(self) -> tuple[int, int]:
        return self._local_coord

    @property
    def role(self) -> tuple[typing.Literal['data', 'measure'], typing.Literal['X', 'Z'] | None]:
        return self._role

    @property
    def info(self) -> dict[str, int] | None:
        return self._info
    
    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, int, int, typing.Literal["data", "measure"], typing.Literal["X", "Z"] | None]:
        return self.patch_id, self.local_coord[0], self.local_coord[1], self.role[0], self.role[1]

    def __str__(self) -> str:
        return f"{self.patch_id}:({self.local_coord[0], self.local_coord[1]}, {self.role[1] if self.role[1] else self.role[0]})"

    def __repr__(self) -> str:
        return f"codepatch.PhysicalQubit({self.patch_id!r}, {self.local_coord!r}, {self.role})"    


class LogicalQubit(cirq.Qid):
    """A logical qubit represented as a wire in a Cirq circuit."""

    def __init__(
        self,
        patch_id: int,
        logical_index: int,
        x_support: typing.Iterable[int],
        z_support: typing.Iterable[int],
    ) -> None:
        self._patch_id = _validate_id(patch_id, "patch_id")
        self._logical_index = _validate_id(logical_index, "logical_index")
        self._x_support = self._normalize_support(x_support, "x_support")
        self._z_support = self._normalize_support(z_support, "z_support")
        if not self.x_support | self.z_support:
            raise ValueError("LogicalQubit supports must include at least one physical qubit.")

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    def logical_index(self) -> int:
        return self._logical_index

    @property
    def x_support(self) -> frozenset[int]:
        """Return the physical-qubit indices supporting the logical X operator."""
        return self._x_support

    @property
    def z_support(self) -> frozenset[int]:
        """Return the physical-qubit indices supporting the logical Z operator."""
        return self._z_support

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, int]:
        return self.patch_id, self.logical_index

    @staticmethod
    def _normalize_support(support: typing.Iterable[int], name: str) -> frozenset[int]:
        return frozenset(_validate_id(qubit, f"LogicalQubit {name} entry") for qubit in support)

    def __str__(self) -> str:
        return f"{self.patch_id}:{self.logical_index}"

    def __repr__(self) -> str:
        return f"codepatch.LogicalQubit({self.patch_id!r}, {self.logical_index!r})"


class CodePatch(abc.ABC):
    """Common metadata for an error-correcting code patch."""

    def __init__(
        self,
        patch_id: int,
        n: int,
        k: int,
        d: int,
    ) -> None:
        self._patch_id = _validate_id(patch_id, "patch_id")
        self.n = n
        self.k = k
        self.d = d
        self.num_data_qubits = self.n

    @property
    def patch_id(self) -> int:
        return self._patch_id

    @property
    @abc.abstractmethod
    def num_measure_qubits(self) -> int:
        """Return the number of measurement qubits in the code patch."""

    @property
    @abc.abstractmethod
    def logical_qubits(self) -> tuple[LogicalQubit, ...]:
        """Return the logical qubits encoded by the code patch."""

    @property
    def num_logical_qubits(self) -> int:
        """Return the number of logical qubits encoded by the code patch."""
        return len(self.logical_qubits)

    @property
    def code_params(self) -> tuple[int, int, int | float | None]:
        """The patch's [n, k, d] code parameters."""
        return self.n, self.k, self.d

    @property
    def num_physical_qubits(self) -> int:
        """The total number of data and measurement qubits in the code patch."""
        return self.num_data_qubits + self.num_measure_qubits

    @property
    def footprint(self) -> Rect:
        qubits = self.default_layout()
        rows = [q.local_coord[0] for q in qubits]
        cols = [q.local_coord[1] for q in qubits]

        return Rect(
            rows=max(rows) - min(rows) + 1,
            cols=max(cols) - min(cols) + 1,
        )   


class CSSCodePatch(CodePatch):
    """Metadata shared by qLDPC CSS code patches."""

    @abc.abstractmethod
    def __init__(self, patch_id: int, qldpc_code: codes.CSSCode) -> None:
        self._qldpc_code = qldpc_code
        n, k, code_distance = self._qldpc_code.get_code_params()
        self._logical_qubits = tuple(self._logical_qubits_from_qldpc_code(patch_id))
        super().__init__(
            patch_id=patch_id,
            n=int(n),
            k=int(k),
            d=int(code_distance),
        )

    @property
    def qldpc_code(self) -> codes.CSSCode:
        return self._qldpc_code

    @property
    def logical_qubits(self) -> tuple[LogicalQubit, ...]:
        return self._logical_qubits

    @property
    def num_measure_qubits(self) -> int:
        return int(self.qldpc_code.num_checks)

    def num_x_stabilizers(self) -> int:
        """Return the number of X-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_x)

    def num_z_stabilizers(self) -> int:
        """Return the number of Z-type stabilizer checks."""
        return int(self.qldpc_code.num_checks_z)

    def total_x_check_weight(self) -> int:
        """Return the total weight of all X-type stabilizer checks."""
        return len(self.qldpc_code.matrix_x.nonzero()[0])

    def total_z_check_weight(self) -> int:
        """Return the total weight of all Z-type stabilizer checks."""
        return len(self.qldpc_code.matrix_z.nonzero()[0])

    def _logical_qubits_from_qldpc_code(self, patch_id: int) -> list[LogicalQubit]:
        logical_x = self.qldpc_code.get_logical_ops(Pauli.X)
        logical_z = self.qldpc_code.get_logical_ops(Pauli.Z)
        return [
            LogicalQubit(
                patch_id=patch_id,
                logical_index=index,
                x_support=self._logical_op_support(logical_x[index]),
                z_support=self._logical_op_support(logical_z[index]),
            )
            for index in range(len(logical_x))
        ]

    @staticmethod
    def _logical_op_support(logical_op: typing.Iterable[int]) -> set[int]:
        return {index for index, value in enumerate(logical_op) if value}


class RotatedSurfaceCodePatch(CSSCodePatch):
    """A rotated surface-code patch generated from the qldpc library."""

    def __init__(self, patch_id: int, d: int) -> None:
        if (d - 1) % 2 != 0:
            raise ValueError("CodePatches must be odd distance")
        super().__init__(patch_id=patch_id, qldpc_code=codes.SurfaceCode(d))

    def __repr__(self) -> str:
        args = [
            f"patch_id={self.patch_id!r}",
            f"d={self.d!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
        ]
        return f"codepatch.RotatedSurfaceCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        def transform(x: int, y: int) -> tuple[int, int]:
            return (
                (x + y)//2,
                (x - y + 2 * (self.d - 1))//2,
            )
        qubits = []
        # Data Qubits
        for r in range(self.d):
            for c in range(self.d):
                local_coord = transform(2*r, 2*c)
                role = ("data", None)
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        # Interior weight 4 checks
        for r in range(self.d-1):
            for c in range(self.d-1):
                basis = "X" if (r+c)%2==0 else "Z"
                role = ("measure", basis)
                local_coord = transform(2*r+1, 2*c+1)
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        # Boundary weight 2 checks
        for k in range(self.d - 1):
            if k%2==0:
                # Top Z Boundary
                local_coord = transform(-1, 2*k+1)
                role = ("measure", "Z")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
                # Right X Boundary
                local_coord = transform(2*k+1, 2*self.d-1)
                role = ("measure", "X")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
            else:
                # Bottom Z Boundary
                local_coord = transform(2*self.d-1, 2*k+1)
                role = ("measure", "Z")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
                # Left X Boundary
                local_coord = transform(2*k+1, -1)
                role = ("measure", "X")
                qubits.append(PhysicalQubit(patch_id=self._patch_id, local_coord=local_coord, role=role))
        return qubits

BBTerm = tuple[int, int]
"""An exponent pair (i, j) standing for the monomial x^i y^j."""

BBPolynomial = typing.Union[str, sympy.Expr, typing.Sequence[BBTerm]]
"""A BB polynomial: a string such as "x^3 + y + y^2", a sympy expression, or (i, j) terms."""


def _parse_bb_polynomial(text: str) -> list[BBTerm]:
    """Parse e.g. "1 + y + x + x*y^-2" into exponent pairs (i, j)."""
    terms = []
    for raw in text.replace(" ", "").split("+"):
        if not raw:
            raise ValueError(f"Empty term in polynomial {text!r}.")
        i = j = 0
        if raw != "1":
            for factor in raw.split("*"):
                match = re.fullmatch(r"([xy])(?:\^(-?\d+))?", factor)
                if not match:
                    raise ValueError(f"Cannot parse factor {factor!r} in {text!r}.")
                power = int(match.group(2) or 1)
                if match.group(1) == "x":
                    i += power
                else:
                    j += power
        terms.append((i, j))
    return terms


def _sympy_bb_terms(poly: sympy.Expr) -> list[BBTerm]:
    """Exponent pairs of a sympy polynomial in x and y with unit coefficients."""
    terms = []
    for term in sympy.Add.make_args(sympy.expand(poly)):
        coeff, monomial = term.as_coeff_Mul()
        powers = monomial.as_powers_dict()
        if coeff != 1 or set(powers) - {x, y, sympy.S.One}:
            raise ValueError(f"BB polynomial terms must be monomials in x and y, not {term}.")
        terms.append((int(powers.get(x, 0)), int(powers.get(y, 0))))
    return terms


def _format_bb_term(term: BBTerm) -> str:
    parts = [f"{v}^{p}" if p != 1 else v for v, p in zip("xy", term) if p]
    return "*".join(parts) or "1"


# ----- GF(2) helpers for choosing short logical operators -----


def _gf2_row_reduce(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return (reduced row echelon form, pivot columns) of a 0/1 matrix over GF(2)."""
    reduced = np.array(matrix, dtype=np.uint8, ndmin=2)
    pivots: list[int] = []
    row = 0
    for column in range(reduced.shape[1]):
        if row == reduced.shape[0]:
            break
        hits = np.flatnonzero(reduced[row:, column])
        if not hits.size:
            continue
        pivot = row + int(hits[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        others = np.flatnonzero(reduced[:, column])
        reduced[others[others != row]] ^= reduced[row]
        pivots.append(column)
        row += 1
    return reduced, pivots


def _gf2_rank(matrix: np.ndarray) -> int:
    return len(_gf2_row_reduce(matrix)[1])


def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Rows v spanning {v : matrix @ v = 0} over GF(2)."""
    reduced, pivots = _gf2_row_reduce(matrix)
    free = [c for c in range(reduced.shape[1]) if c not in set(pivots)]
    basis = np.zeros((len(free), reduced.shape[1]), dtype=np.uint8)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for r, p in enumerate(pivots):
            basis[k, p] = reduced[r, f]
    return basis


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    reduced, _ = _gf2_row_reduce(np.hstack([matrix, np.eye(size, dtype=np.uint8)]))
    return reduced[:, size:]


def _quotient_basis(kernel_of: np.ndarray, stabilizers: np.ndarray) -> np.ndarray:
    """Vectors of ker(kernel_of) that are independent modulo rowspace(stabilizers)."""
    rows, current = list(stabilizers), _gf2_rank(stabilizers)
    chosen = []
    for vector in _gf2_nullspace(kernel_of):
        if _gf2_rank(np.vstack([*rows, vector])) > current:
            rows.append(vector)
            chosen.append(vector)
            current += 1
    return np.array(chosen, dtype=np.uint8).reshape(-1, kernel_of.shape[1])


def _low_weight_lz(
    hx: np.ndarray, hz: np.ndarray, trials: int = 100, seed: int = 0
) -> np.ndarray | None:
    """A logical-Z basis of the shortest representatives found, or None if incomplete."""
    rng = np.random.default_rng(seed)
    kernel = _gf2_nullspace(hx)
    n = hx.shape[1]
    found: dict[bytes, np.ndarray] = {}
    for _ in range(trials):
        order = rng.permutation(n)
        for row in _gf2_row_reduce(kernel[:, order])[0]:
            vector = np.zeros(n, dtype=np.uint8)
            vector[order] = row
            found.setdefault(vector.tobytes(), vector)
    rows, current = list(hz), _gf2_rank(hz)
    k = n - _gf2_rank(hx) - current
    chosen: list[np.ndarray] = []
    for vector in sorted(found.values(), key=lambda v: int(v.sum())):
        if len(chosen) == k:
            break
        if _gf2_rank(np.vstack([*rows, vector])) > current:
            rows.append(vector)
            chosen.append(vector)
            current += 1
    if len(chosen) < k:
        return None
    return np.array(chosen, dtype=np.uint8)


def _low_weight_logicals(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A paired logical basis (lz @ lx.T = I) with short Z logicals, then greedily shortened X.

    Z logicals come from random information sets of ker(hx); X logicals are paired to them and
    shortened by adding X checks. Rotation gadgets act on Z supports, so those are shortest.
    """
    hx, hz = np.asarray(hx, dtype=np.uint8), np.asarray(hz, dtype=np.uint8)
    lx = _quotient_basis(hz, hx)
    lz = _low_weight_lz(hx, hz)
    if lz is None:
        lz = _quotient_basis(hx, hz)
    lx = _gf2_inverse(lz @ lx.T % 2).T @ lx % 2
    for _ in range(4):
        improved = False
        for row in lx:
            for stabilizer in hx:
                candidate = row ^ stabilizer
                if candidate.sum() < row.sum():
                    row[:] = candidate
                    improved = True
        if not improved:
            break
    return lx.astype(np.uint8), lz


class BBCodePatch(CSSCodePatch):
    """A bivariate bicycle (BB) code patch generated from the qldpc library.

    The group is Z_l x Z_m, with x of order l and y of order m. Polynomials A and B give
    ``matrix_x = [A | B]`` and ``matrix_z = [B^T | A^T]``. Data qubits 0..lm-1 form the left half
    and lm..2lm-1 the right half; inside a half, group element x^i y^j has index i*m + j. X and Z
    checks use the same group indexing.
    """

    def __init__(
        self,
        patch_id: int,
        l: int,  # noqa: E741
        m: int,
        poly_a: BBPolynomial | None = None,
        poly_b: BBPolynomial | None = None,
        *,
        distance: int | None = None,
        low_weight_logicals: bool = True,
    ) -> None:
        """Builds a BB code patch.

        Args:
            patch_id: The patch id.
            l: The order of x.
            m: The order of y.
            poly_a: Polynomial A as a string ("x^3 + y + y^2"), sympy expression or (i, j) terms.
                Defaults (with poly_b) to the polynomials from the QuEra STAR paper.
            poly_b: Polynomial B, in the same forms as poly_a.
            distance: The known code distance. If omitted, qldpc computes it exactly, which is slow
                for large codes.
            low_weight_logicals: If True (the default), the logical qubits use short logical-Z
                representatives (and paired, greedily shortened logical X) instead of qldpc's
                default basis. Rotation gadgets act on logical-Z supports.
        """
        for value, name in ((l, "l"), (m, "m")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (poly_a is None) != (poly_b is None):
            raise ValueError("Pass both poly_a and poly_b, or neither.")
        if poly_a is None or poly_b is None:
            # Polynomials from the QuEra STAR paper.
            poly_a = 1 + y**3 + x * y**2 + x * y**4
            poly_b = 1 + y**-3 + x**-1 * y**-2 + x**-1 * y**-4
        self._l, self._m = l, m
        terms_a, terms_b = self._terms(poly_a), self._terms(poly_b)
        self._poly_a = self._reduce_terms(terms_a)
        self._poly_b = self._reduce_terms(terms_b)
        # qldpc gets the polynomials as written: reduced ones may lose x or y entirely.
        qldpc_code = codes.BBCode(
            {x: l, y: m},
            sum((x**i * y**j for i, j in terms_a), sympy.S.Zero),
            sum((x**i * y**j for i, j in terms_b), sympy.S.Zero),
        )
        if distance is not None:
            if not isinstance(distance, int) or isinstance(distance, bool) or distance < 1:
                raise ValueError("distance must be a positive integer.")
            # Cache the known distance so qldpc does not search for it.
            qldpc_code._distance = distance
        if low_weight_logicals:
            qldpc_code.set_logical_ops_xz(
                *_low_weight_logicals(qldpc_code.matrix_x, qldpc_code.matrix_z)
            )
        super().__init__(patch_id=patch_id, qldpc_code=qldpc_code)

    @property
    def l(self) -> int:  # noqa: E743
        """The order of x."""
        return self._l

    @property
    def m(self) -> int:
        """The order of y."""
        return self._m

    @property
    def poly_a(self) -> tuple[BBTerm, ...]:
        """Polynomial A as exponent pairs (i, j), reduced mod (l, m)."""
        return self._poly_a

    @property
    def poly_b(self) -> tuple[BBTerm, ...]:
        """Polynomial B as exponent pairs (i, j), reduced mod (l, m)."""
        return self._poly_b

    @property
    def half(self) -> int:
        """Data qubits per half (= lm = number of X checks = number of Z checks)."""
        return self.l * self.m

    def index(self, term: BBTerm) -> int:
        """Index of group element x^i y^j inside one half (also its X/Z check index)."""
        return (term[0] % self.l) * self.m + term[1] % self.m

    def polynomials(self) -> tuple[str, str]:
        """Polynomials A and B as strings."""
        return (
            " + ".join(map(_format_bb_term, self.poly_a)),
            " + ".join(map(_format_bb_term, self.poly_b)),
        )

    @staticmethod
    def _terms(poly: BBPolynomial) -> list[BBTerm]:
        if isinstance(poly, str):
            return _parse_bb_polynomial(poly)
        if isinstance(poly, sympy.Expr):
            return _sympy_bb_terms(poly)
        return [(int(i), int(j)) for i, j in poly]

    def _reduce_terms(self, terms: list[BBTerm]) -> tuple[BBTerm, ...]:
        # Coefficients live in GF(2), so terms that coincide modulo (l, m) cancel in pairs.
        counts = collections.Counter((i % self.l, j % self.m) for i, j in terms)
        normal = tuple(term for term, count in counts.items() if count % 2)
        if not normal:
            raise ValueError("BB polynomials must have at least one term modulo (l, m).")
        return normal

    def __repr__(self) -> str:
        poly_a, poly_b = self.polynomials()
        args = [
            f"patch_id={self.patch_id!r}",
            f"l={self.l!r}",
            f"m={self.m!r}",
            f"poly_a={poly_a!r}",
            f"poly_b={poly_b!r}",
            f"n={self.n!r}",
            f"k={self.k!r}",
            f"d={self.d!r}",
        ]
        return f"codepatch.BBCodePatch({', '.join(args)})"

    def default_layout(self) -> list[PhysicalQubit]:
        """Produces a plaquette-based physical layout for the physical qubits in the code patch"""
        qubits = []
        l, m = self.l, self.m
        for index in range(self.n):
            half, col, row = map(int, np.unravel_index(index, (2, l, m)))
            if half == 0:  # L data + X check
                data_local_coord = (2 * row, 2 * col)
                meas_local_coord = (2 * row, 2 * col + 1)
                basis = "X"
            else:  # R data + Z check
                data_local_coord = (2 * row + 1, 2 * col + 1)
                meas_local_coord = (2 * row + 1, 2 * col)
                basis = "Z"
            data_qubit = PhysicalQubit(patch_id=self.patch_id, local_coord=data_local_coord, role=("data", None))
            meas_qubit = PhysicalQubit(patch_id=self.patch_id, local_coord=meas_local_coord, role=("measure", basis))
            qubits.extend([data_qubit, meas_qubit])
        return qubits


@dataclass(frozen=True)
class Rect:
    rows: int
    cols: int
    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rect dimensions must be positive")

class Placeable(typing.Protocol):
    @property
    def footprint(self) -> Rect:
        ...

@dataclass(frozen=True)
class Placement:
    element: Placeable
    row: int
    col: int

    @property
    def row_end(self) -> int:
        return self.row + self.element.footprint.rows

    @property
    def col_end(self) -> int:
        return self.col + self.element.footprint.cols

    def overlaps(self, placement: Placement):
        return not (
            placement.row_end <= self.row
            or self.row_end <= placement.row
            or placement.col_end <= self.col
            or self.col_end <= placement.col
        )

@dataclass
class Zone:
    rows: int
    cols: int
    role: str

    @property
    def footprint(self) -> Rect:
        return Rect(self.rows, self.cols)

@dataclass
class Layout2:
    rows: int
    cols: int
    spacing: float = 1.0
    placements: dict[int, Placement] = field(default_factory=dict)
    regions: dict[int, Zone] = field(default_factory=dict)

    def add(self, element: Placeable, row, col) -> int:
        """Adds a new proposed placement if possible, returning its unique identifier"""
        candidate_placement = Placement(element=element, row=row, col=col)
        if not self.in_bounds(placement=candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement):
            raise ValueError("This space is already occupied")
        next_id = max(self.placements, default=0) + 1
        self.placements[next_id] = candidate_placement
        return next_id

    def remove(self, placement_id: int):
        """Removes the Placement at placement_id"""
        del self.placements[placement_id]

    def move(self, placement_id: int, row: int, col: int) -> None:
        """Moves the anchor of a Placement with placement_id to the given row, col coordinate"""
        prior_placement = self.placements[placement_id]
        candidate_placement = Placement(element=prior_placement.element, row=row, col=col)
        if not self.in_bounds(candidate_placement):
            raise ValueError("This placement is out of bounds")
        if self.occupied(placement=candidate_placement, exclude=placement_id):
            raise ValueError("This space is already occupied")
        self.placements[placement_id] = candidate_placement

    def global_coord(self, placement_id: int, local_coord: tuple[int, int]) -> tuple[int, int]:
        """Returns global coordinate for a local object within the Placement with the given placement_id"""
        placement = self.placements[placement_id]
        r, c = local_coord
        return placement.row + r, placement.col + c
        
    def occupied(self, placement: Placement, exclude: int | None = None) -> bool:
        """Returns True if the proposed Placement overlaps with any existing placement, subject to excluding one placement"""
        return any(placement.overlaps(prior_placement) for pid, prior_placement in self.placements.items() if pid != exclude)

    def in_bounds(self, placement: Placement):
        """Returns True if the proposed placement is within the bounds of the Layout"""
        return (
            placement.row >= 0
            and placement.col >= 0
            and placement.row_end <= self.rows
            and placement.col_end <= self.cols
        )

    def get_patch(self, patch_id: int):
        for placement in self.placements.values():
            element = placement.element
            if isinstance(element, CodePatch) and element.patch_id == patch_id:
                return element
        raise ValueError(f"Found no CodePatch with patch_id={patch_id} in Layout")
        
