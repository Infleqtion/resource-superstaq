"""The degenerate boundary patch used by inplace logical-Y preparation."""

from __future__ import annotations

from dataclasses import dataclass

from .types import PauliProduct


@dataclass(frozen=True)
class YBoundarySurfaceCode:
    """The full-rank XXZZ patch immediately before the Y-transition round.

    This is the paper's ``ZXXZ`` boundary ordering when listed clockwise from
    the top edge, equivalently ``XXZZ`` when listed from the right edge.  The
    top-left data site is absent, so the patch has ``d**2 - 1`` qubits and the
    same number of independent CSS stabilizers.  It consequently stores no
    logical qubits; its unique stabilizer state is the input to the inverse
    diagonal-twist transition that prepares ``|+i>``.
    """

    distance: int

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be odd and at least 3")

    @property
    def num_data_qubits(self) -> int:
        return self.distance**2 - 1

    def canonical_wire(self, x: int, y: int) -> int:
        """Map coordinates to compact IDs, omitting the top-left site."""
        if (x, y) == (0, 0):
            raise ValueError("the XXZZ boundary patch omits data site (0, 0)")
        return y * self.distance + x - 1

    def data_coordinates(self) -> dict[int, tuple[float, float]]:
        """Return display coordinates with boundary ancillas on non-negative sites."""
        return {
            self.canonical_wire(x, y): (x + 0.5, y + 0.5)
            for y in range(self.distance)
            for x in range(self.distance)
            if (x, y) != (0, 0)
        }

    def initial_basis(self, x: int, y: int) -> str:
        """Return the nearest-boundary basis used in the reversed final round."""
        return "Z" if x + y < self.distance else "X"

    def stabilizers(self) -> list[PauliProduct]:
        """Return the full-rank CSS checks of the missing-corner XXZZ patch."""
        d = self.distance
        data_sites = {(x, y) for y in range(d) for x in range(d) if (x, y) != (0, 0)}
        checks: list[PauliProduct] = []
        for check_y in range(-1, d):
            for check_x in range(-1, d):
                center_x = check_x + 0.5
                center_y = check_y + 0.5
                support_sites = [
                    (x, y)
                    for y in (check_y, check_y + 1)
                    for x in (check_x, check_x + 1)
                    if (x, y) in data_sites
                ]
                if len(support_sites) < 2:
                    continue

                pauli = "Z" if (check_x + check_y) % 2 == 0 else "X"
                x_boundary = center_y == d - 0.5 or center_x == d - 0.5
                z_boundary = center_y == -0.5 or center_x == -0.5
                if (x_boundary and pauli != "X") or (z_boundary and pauli != "Z"):
                    continue
                checks.append(
                    (pauli, tuple(self.canonical_wire(x, y) for x, y in support_sites))
                )
        if len(checks) != self.num_data_qubits:
            raise AssertionError("XXZZ patch must have a full-rank stabilizer group")
        return checks


def y_transition_inverse_schedule(
    distance: int, *, include_ticks: bool = False
) -> tuple[list[str], dict[int, tuple[float, float]]]:
    """Return the reverse-time diagonal-twist transition from Fig. 7.

    The schedule is a coordinate-preserving translation of the paper's
    reference circuit.  It maps the missing-corner, full-rank XXZZ patch into
    the ordinary rotated patch while preparing the restored corner in the Y
    basis.  The returned coordinates include both data and transient checks.
    """
    boundary = YBoundarySurfaceCode(distance)
    d = distance

    def patch_checks(
        *, missing_corner: bool, boundaries: tuple[str, str, str, str]
    ) -> dict[tuple[int, int], str]:
        top, right, bottom, left = boundaries
        data = {
            (x, y)
            for y in range(d)
            for x in range(d)
            if not missing_corner or (x, y) != (0, 0)
        }
        checks: dict[tuple[int, int], str] = {}
        for y in range(-1, d):
            for x in range(-1, d):
                support = {
                    (data_x, data_y)
                    for data_y in (y, y + 1)
                    for data_x in (x, x + 1)
                    if (data_x, data_y) in data
                }
                if len(support) < 2:
                    continue
                basis = "Z" if (x + y) % 2 == 0 else "X"
                required = {
                    top if y == -1 else None,
                    right if x == d - 1 else None,
                    bottom if y == d - 1 else None,
                    left if x == -1 else None,
                } - {None}
                if not required or (len(required) == 1 and basis == required.pop()):
                    checks[(2 * x + 1, 2 * y + 1)] = basis
        return checks

    # The ordinary XZXZ patch and the missing-corner XXZZ patch use different
    # subsets of the same check lattice. Their union is the transition's set
    # of transient check ancillas.
    ordinary = patch_checks(missing_corner=False, boundaries=("X", "Z", "X", "Z"))
    boundary_checks = patch_checks(missing_corner=True, boundaries=("Z", "X", "X", "Z"))
    check_positions = sorted(
        set(ordinary) | set(boundary_checks), key=lambda p: (p[1], p[0])
    )
    wire_at = {
        **{(2 * x, 2 * y): y * d + x for y in range(d) for x in range(d)},
        **{position: d * d + index for index, position in enumerate(check_positions)},
    }
    used = set(wire_at)

    def is_x_check(position: tuple[int, int]) -> bool:
        x, y = position
        return x % 2 and y % 2 and ((x + y) // 2) % 2 == 0

    xs = {position for position in used if is_x_check(position)}
    zs = {
        position
        for position in used
        if position[0] % 2 and position[1] % 2 and position not in xs
    }
    top_row = {position for position in used if position[1] == -1}
    right_col = {position for position in used if position[0] == 2 * d - 1}

    def split_diagonal(
        positions: set[tuple[int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
        down_left: set[tuple[int, int]] = set()
        middle: set[tuple[int, int]] = set()
        up_right: set[tuple[int, int]] = set()
        for x, y in positions:
            if x > y + 2:
                up_right.add((x, y))
            elif x in {y, y + 2}:
                middle.add((x, y))
            else:
                down_left.add((x, y))
        return down_left, middle, up_right

    xs_dl, xs_md, xs_ur = split_diagonal(xs)
    zs_dl, zs_md, zs_ur = split_diagonal(zs)

    def pairs(
        positions: set[tuple[int, int]], delta: tuple[int, int], *, reverse: bool
    ) -> list[int]:
        result: list[int] = []
        for position in sorted(positions, key=lambda p: (p[1], p[0])):
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if neighbor not in used:
                continue
            first, second = (neighbor, position) if reverse else (position, neighbor)
            result.extend((wire_at[first], wire_at[second]))
        return result

    def operation(name: str, targets: list[int]) -> str | None:
        return f"{name} {' '.join(map(str, targets))}" if targets else None

    def append_layer(lines: list[str], *operations: str | None) -> None:
        # Operations in one paper layer have disjoint targets. Render each
        # gate kind once, keeping the DEQ schedule compact without changing
        # its Clifford action.
        grouped_targets: dict[str, list[str]] = {}
        for operation_text in operations:
            if operation_text is None:
                continue
            name, targets = operation_text.split(" ", maxsplit=1)
            grouped_targets.setdefault(name, []).append(targets)
        lines.extend(
            f"{name} {' '.join(targets)}" for name, targets in grouped_targets.items()
        )
        if include_ticks:
            lines.append("TICK")

    def append_inverse_layer(
        lines: list[str], layer: list[tuple[str, list[int]]]
    ) -> None:
        """Append a reversed paper layer using its native Clifford gates."""
        append_layer(lines, *(operation(name, targets) for name, targets in layer))

    dl, dr, ul, ur = (-1, 1), (1, 1), (-1, -1), (1, -1)
    # These are the forward circuit's Clifford layers. The inverse below
    # reverses their order. XCY and CX are self-inverse in this usage.
    forward_layers = [
        [
            ("CX", pairs(xs - right_col, dl, reverse=False)),
            ("CX", pairs(zs - top_row, dl, reverse=True)),
        ],
        [
            ("CX", pairs(xs - right_col, dr, reverse=False)),
            ("CX", pairs(zs - top_row, ul, reverse=True)),
        ],
        [
            ("CX", pairs(xs_ur | xs_md, ul, reverse=True)),
            ("CX", pairs(zs_ur, dr, reverse=False)),
            ("XCY", pairs(zs_md, dr, reverse=False)),
            ("CX", pairs(xs_dl, ul, reverse=False)),
            ("CX", pairs(zs_dl, dr, reverse=True)),
        ],
        [
            ("CX", pairs(xs_ur, dl, reverse=True)),
            ("CX", pairs(zs_ur, dl, reverse=False)),
            ("CX", pairs(xs_dl, ur, reverse=False)),
            ("CX", pairs(zs_dl, ur, reverse=True)),
        ],
        [("XCY", pairs(xs_md - top_row, dl, reverse=True))],
    ]

    x_measurements = (xs - top_row) | right_col
    z_measurements = (zs - right_col) | top_row
    lines: list[str] = [
        operation("R", [wire_at[position] for position in sorted(z_measurements)]),
        "RY 0",
        operation("RX", [wire_at[position] for position in sorted(x_measurements)]),
    ]
    if include_ticks:
        lines.append("TICK")
    # Source Fig. 7 applies H to the upper-right half and sqrt(X) along the
    # diagonal. Its published reverse-time preparation circuit uses SQRT_X
    # with this same convention.
    append_layer(
        lines,
        operation(
            "SQRT_X",
            [
                wire_at[position]
                for position in sorted(used)
                if position[0] == position[1] and position[0] % 2
            ],
        ),
        operation(
            "H",
            [
                wire_at[position]
                for position in sorted(used)
                if position[0] > position[1]
            ],
        ),
    )
    for layer in reversed(forward_layers):
        append_inverse_layer(lines, layer)
    lines.extend(
        operation
        for operation in (
            operation(
                "M",
                [wire_at[position] for position in sorted((zs - top_row) | right_col)],
            ),
            operation(
                "MX",
                [wire_at[position] for position in sorted((xs - right_col) | top_row)],
            ),
        )
        if operation is not None
    )
    # The inverse transition's logical-Y flow is the parity of the terminal
    # Z-check outcomes strictly above the diagonal and the terminal X-check
    # outcomes strictly below it. Each record-controlled X on the restored
    # corner flips the logical Y sign, fixing the output to the +1 eigenstate.
    terminal_measurements = sorted((zs - top_row) | right_col) + sorted(
        (xs - right_col) | top_row
    )
    correction_targets: list[str] = []
    for index, position in enumerate(terminal_measurements):
        if (position in zs and position[0] > position[1]) or (
            position in xs and position[0] < position[1]
        ):
            correction_targets.extend(
                (f"rec[-{len(terminal_measurements) - index}]", "0")
            )
    if correction_targets:
        lines.append("CX " + " ".join(correction_targets))
    coordinates = {
        wire: (position[0] / 2 + 0.5, position[1] / 2 + 0.5)
        for position, wire in wire_at.items()
    }
    # The boundary state's compact code IDs map to physical wires 1..d²-1;
    # wire 0 is restored by the RY in the transition itself.
    if set(boundary.data_coordinates()) != set(range(d * d - 1)):
        raise AssertionError("unexpected compact missing-corner wire layout")
    return [line for line in lines if line is not None], coordinates
