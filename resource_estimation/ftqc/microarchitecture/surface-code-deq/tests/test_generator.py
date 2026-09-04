import importlib.util
from pathlib import Path
import re
import sys

import pytest
import stim

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from surface_code_deq import (
    HadamardDeformationLayout,
    RotatedSurfaceCode,
    YBoundarySurfaceCode,
)
from surface_code_deq.schedules import cnot_syndrome_schedule
from surface_code_deq.gadget_visualizations import gadget_circuits
from surface_code_deq.hadamard_gadgets import (
    _corner_movement_hook_orders,
    _extension_hook_orders,
)
from surface_code_deq.prepare_y_gadgets import _boundary_round, y_boundary_circuits
from surface_code_deq.y_basis import y_transition_inverse_schedule


_generator_path = (
    Path(__file__).parents[1] / "tools" / "generate_rotated_surface_code_deq.py"
)
_spec = importlib.util.spec_from_file_location("generator", _generator_path)
assert _spec and _spec.loader
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def test_y_boundary_code_is_full_rank_with_the_missing_corner_layout() -> (
    None
):
    for distance in (3, 5, 7):
        patch = YBoundarySurfaceCode(distance)
        stabilizers = patch.stabilizers()
        assert patch.num_data_qubits == distance * distance - 1
        assert len(stabilizers) == patch.num_data_qubits
        assert patch.canonical_wire(1, 0) == 0
        assert patch.initial_basis(1, 0) == "Z"
        assert patch.initial_basis(distance - 1, distance - 1) == "X"

        tableau = stim.Tableau.from_stabilizers(
            [
                stim.PauliString(
                    "".join(
                        pauli if qubit in support else "_"
                        for qubit in range(patch.num_data_qubits)
                    )
                )
                for pauli, support in stabilizers
            ],
            allow_underconstrained=False,
        )
        assert len(tableau) == patch.num_data_qubits


def test_y_boundary_preparation_defaults_to_half_distance_boundary_rounds() -> None:
    source = generator.render_prepare_y(5)
    assert source.count("CODE ") == 2
    assert "CODE YBoundaryStateD5 [[24,0]]" in source
    assert "GADGET PrepareYBoundaryD5" in source
    assert "GADGET PrepareYBoundarySED5" in source
    assert "GADGET PrepareYTransitionD5" in source
    assert "COMPOSE PrepareY" in source
    assert "REPEAT 2 {\n        PrepareYBoundarySED5 0\n    }" in source
    assert "PrepareYTransitionD5 IN(0) OUT(1)" in source
    assert "OUTPUT RotatedSurfaceCodeW5H5 1" in source
    assert "REPEAT 2" in source
    assert "CX rec[-" in source
    with pytest.raises(ValueError, match="at least 2"):
        generator.render_prepare_y(5, boundary_rounds=1)


def test_si1000_noise_rejects_invalid_physical_error_rates() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        generator.inject_si1000_noise("GADGET Empty {\\n}\\n", -0.001)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        generator.inject_si1000_noise("GADGET Empty {\\n}\\n", 1.001)


def test_unified_surface_code_library_declares_shared_types_once() -> None:
    source = generator.render_surface_code_library(3)
    assert source.count("CODE RotatedSurfaceCodeW3H3 [[9,1,3]]") == 1
    assert "COMPOSE FaultTolerantCNOTD3" in source
    assert "COMPOSE LogicalHadamardD3" in source
    assert "COMPOSE LogicalSD3" in source
    assert "COMPOSE PrepareY" in source
    assert "YBoundarySurfaceCode" not in source
    assert source.index("GADGET PrepareX") < source.index("GADGET PrepareYBoundaryD3")
    assert source.index("GADGET PrepareYBoundaryD3") < source.index(
        "GADGET SyndromeExtraction"
    )


def test_logical_s_injects_the_prepared_y_state_and_tracks_its_byproduct() -> None:
    source = generator.render_surface_code_library(3)
    logical_s = source[source.index("COMPOSE LogicalSD3 {") :]
    assert "PrepareY 1" in logical_s
    assert "FaultTolerantCNOTD3 0 1" in logical_s
    assert "MeasureZ 1" in logical_s
    assert "CONDITIONAL rec[-1] Z0 0" in logical_s


def test_measurement_readouts_follow_the_declared_logical_strings() -> None:
    source = generator.render_surface_code_library(3)
    measure_z = source[source.index("GADGET MeasureZ {") : source.index("GADGET MeasureX {")]
    measure_x = source[source.index("GADGET MeasureX {") : source.index("GADGET LogicalX {")]
    assert "READOUT rec[-9] rec[-8] rec[-7]" in measure_z
    assert "READOUT rec[-9] rec[-6] rec[-3]" in measure_x


def test_reverse_time_y_transition_prepares_the_positive_logical_y_state() -> None:
    """The transition's record-controlled corner X fixes every logical frame."""
    for distance in (3, 5, 7):
        boundary = YBoundarySurfaceCode(distance)
        boundary_round, _, _ = _boundary_round(boundary)

        transition, coordinates = y_transition_inverse_schedule(distance)
        assert min(value for point in coordinates.values() for value in point) >= 0
        assert "TICK" not in transition
        assert any(line.startswith("XCY ") for line in transition)
        assert any(line.startswith("SQRT_X ") for line in transition)
        correction_lines = [line for line in transition if line.startswith("CX rec[-")]
        assert len(correction_lines) == 1

        z_data = []
        x_data = []
        for wire, (x, y) in boundary.data_coordinates().items():
            (
                z_data if boundary.initial_basis(int(x), int(y)) == "Z" else x_data
            ).append(wire + 1)
        circuit = stim.Circuit(
            "\n".join(
                [
                    "R " + " ".join(map(str, z_data)),
                    "RX " + " ".join(map(str, x_data)),
                    *boundary_round,
                    *transition,
                ]
            )
        )
        logical_y = stim.PauliString(distance * distance)
        logical_y[0] = "Y"
        for index in range(1, distance):
            logical_y[index] = "Z"
            logical_y[index * distance] = "X"

        for _ in range(4):
            simulator = stim.TableauSimulator()
            simulator.do_circuit(circuit)
            assert simulator.peek_observable_expectation(logical_y) == 1

        source = generator.render_prepare_y(distance)
        assert "COMPOSE PrepareY" in source
        assert f"PrepareYTransitionD{distance} IN(0) OUT(1)" in source
        assert "CX rec[-" in source
        assert "TICK" not in source
        visualization = y_boundary_circuits(distance)["YBoundaryToRotated"]
        assert "CX rec[-" not in str(visualization)


def test_prepare_y_boundary_round_uses_gidneys_inverse_hook_order() -> None:
    """The first boundary round is the inverse of Gidney's final readout."""
    schedule, _, _ = _boundary_round(YBoundarySurfaceCode(3))
    cnot_layers = [line for line in schedule if line.startswith("CX ")]
    assert cnot_layers == [
        "CX 1 18 3 19 20 4 22 6 7 23 24 8",
        "CX 20 5 22 7 4 23",
        "CX 2 18 4 19 20 1 6 21 22 3 8 23 24 5 25 7",
        "CX 1 19 20 2 3 21 22 4 5 23 25 8",
    ]


def test_mxx_has_typed_merge_rounds_and_applies_its_z_frame_byproduct() -> None:
    source = generator.render_mxx(3, merged_rounds=3)
    assert "CODE RotatedSurfaceCodeW7H3 [[21,1]]" in source
    assert "GADGET MergeBeginXX" in source
    assert "GADGET MergedSEXX" in source
    assert "GADGET MergeEndXX" in source
    assert "GADGET SyndromeExtraction" in source
    assert "GADGET MeasureX" in source
    assert "COMPOSE RotatedSurfaceCodeMemoryD3" in source
    assert "REPEAT 3" in source
    assert "CONDITIONAL rec[-1] Z0 1" in source
    assert "PROGRAM FaultTolerantMXXD3MemoryX" in source
    assert "GADGET LogicalX" in source
    assert "PROGRAM FaultTolerantMXXD3FrameA" in source
    assert "TICK" not in source
    assert "MPP" not in source
    assert "RX 10 11 13 14" in source
    assert "CX 1 9 10 2 11 4 5 12 14 8 3 15" in source
    assert any(
        line.strip().startswith("R ") and {"18", "19", "20"} <= set(line.split())
        for line in source.splitlines()
    )  # The physical seam shares the first reset layer.
    assert "READOUT M2 M3 M4 M5" in source


def test_syndrome_round_uses_distinct_check_ancillas_and_disjoint_cnot_layers() -> None:
    stabilizers = RotatedSurfaceCode(width=3, height=3).stabilizers()
    schedule, measurement_index = generator._cnot_syndrome_schedule(
        stabilizers, ancilla_offset=9
    )
    assert schedule[0] == "R 9 12 15 16"
    assert schedule[1] == "RX 10 11 13 14"
    assert schedule[-2:] == ["M 9 12 15 16", "MX 10 11 13 14"]
    assert "TICK" not in schedule
    assert measurement_index == {0: 0, 1: 4, 2: 5, 3: 1, 4: 6, 5: 7, 6: 2, 7: 3}
    assert [line for line in schedule if line.startswith("CX ")] == [
        "CX 1 9 10 2 11 4 5 12 14 8 3 15",
        "CX 4 9 10 1 11 3 8 12 14 7 6 15",
        "CX 0 9 10 5 11 7 4 12 13 1 2 16",
        "CX 3 9 10 4 11 6 7 12 13 0 5 16",
    ]

    for line in schedule:
        if not line.startswith("CX "):
            continue
        targets = [int(target) for target in line.split()[1:]]
        assert len(targets) == len(set(targets))


def test_mxx_geometry_is_distance_independent() -> None:
    source = generator.render_mxx(5)
    assert "CODE RotatedSurfaceCodeW5H5 [[25,1,5]]" in source
    assert "CODE RotatedSurfaceCodeW11H5 [[55,1]]" in source
    assert "REPEAT 5" in source
    assert any(
        line.strip().startswith("R ")
        and {"50", "51", "52", "53", "54"} <= set(line.split())
        for line in source.splitlines()
    )


def test_mzz_uses_the_vertical_x_boundary_merge_geometry() -> None:
    source = generator.render_mzz(3, merged_rounds=3)
    assert "CODE RotatedSurfaceCodeW3H7 [[21,1]]" in source
    assert "GADGET MergeBeginZZ" in source
    assert "GADGET MergedSEZZ" in source
    assert "GADGET MergeEndZZ" in source
    assert any(
        line.strip().startswith("RX ") and {"18", "19", "20"} <= set(line.split())
        for line in source.splitlines()
    )
    assert "MX 18 19 20" in source
    assert "READOUT M0 M1 M2 M3" in source
    assert "CONDITIONAL rec[-1] X0 1" in source

    circuits = gadget_circuits(3, "ZZ")
    coordinates = circuits["MergeBeginZZ"].get_final_qubit_coordinates()
    assert coordinates[18] == [0.5, 3.5]
    assert coordinates[20] == [2.5, 3.5]
    assert len({tuple(point) for point in coordinates.values()}) == len(coordinates)


def test_shared_surgery_library_defines_a_mediator_based_cnot() -> None:
    source = generator.render_cnot(3, merged_rounds=3)
    assert "GADGET MergeBeginXX" in source
    assert "GADGET MergeBeginZZ" in source
    assert "COMPOSE FaultTolerantMXXD3" in source
    assert "COMPOSE FaultTolerantMZZD3" in source
    assert "COMPOSE FaultTolerantCNOTD3" in source
    assert "PrepareX 2" in source
    assert "FaultTolerantMZZD3 0 2" in source
    assert "FaultTolerantMXXD3 2 1" in source
    assert "MeasureZ 2" in source
    assert "CONDITIONAL rec[-2] Z0 0" in source
    assert "CONDITIONAL rec[-3] X0 1" in source


def test_logical_hadamard_is_composed_from_typed_deformation_stages() -> None:
    for distance in (3, 5, 7):
        source = generator.render_surgery_library(distance)
        size = distance * distance
        targets = " ".join(map(str, range(size)))
        patch_type = f"RotatedSurfaceCodeW{distance}H{distance}"
        assert f"CODE HadamardFrameD{distance}" in source
        assert f"CODE HadamardExtensionD{distance}" in source
        assert f"CODE HadamardCornerD{distance}" in source
        assert f"CODE HadamardReturnD{distance}" in source
        for stage in (
            "TransversalHadamard",
            "HadamardExtend",
            "HadamardExtensionSE",
            "HadamardCornerMove",
            "HadamardDomainWallSE",
            "HadamardShrink",
            "HadamardReturnExtend",
            "HadamardReturnExtensionSE",
            "HadamardReturnShrink",
            "HadamardSwapQEC",
        ):
            assert f"GADGET {stage}D{distance}" in source
        assert f"COMPOSE LogicalHadamardD{distance}" in source
        assert f"H {targets}" in source
        assert f"INPUT {patch_type} {targets}" in source
        assert f"OUTPUT {patch_type} 7" in source
        assert f"TransversalHadamardD{distance} IN(0) OUT(1)" in source
        assert f"HadamardSwapQECD{distance} IN(6) OUT(7)" in source
        assert (
            f"REPEAT {distance - 1} {{\n"
            f"        HadamardExtensionSED{distance} 2\n"
            "    }"
        ) in source
        assert (
            f"REPEAT {distance - 1} {{\n"
            f"        HadamardDomainWallSED{distance} 3\n"
            "    }"
        ) in source
        assert (
            f"REPEAT {distance - 1} {{\n"
            f"        HadamardReturnExtensionSED{distance} 5\n"
            "    }"
        ) in source
        assert "PROPAGATE" not in source
        assert source.count(f"REPEAT {distance - 1} {{") == 3
        assert "TICK" not in source

    circuit = gadget_circuits(3)["TransversalHadamard"]
    assert "H 0 1 2 3 4 5 6 7 8" in str(circuit)

def test_visualized_merge_circuits_match_rendered_physical_operations() -> None:
    def deq_operations(source: str, gadget_name: str) -> list[str]:
        match = re.search(
            rf"GADGET {gadget_name} \{{(?P<body>.*?)^\}}",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match, gadget_name
        return [
            line.strip()
            for line in match.group("body").splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
            and not line.lstrip().startswith(("INPUT ", "OUTPUT ", "READOUT "))
        ]

    def crumble_operations(circuit: stim.Circuit) -> list[str]:
        return [
            line
            for line in str(circuit).splitlines()
            if not line.startswith("QUBIT_COORDS")
        ]

    def normalized_layers(lines: list[str]) -> list[str]:
        result: list[str] = []
        for line in lines:
            if line == "TICK":
                continue
            operation, *targets = line.split()
            if result and result[-1].split()[0] == operation:
                result[-1] += " " + " ".join(targets)
            else:
                result.append(line)
        return result

    for distance in (3, 5, 7):
        source = generator.render_surgery_library(distance)
        for basis in ("XX", "ZZ"):
            circuits = gadget_circuits(distance, basis)
            for gadget_name in (
                f"MergeBegin{basis}",
                f"MergedSE{basis}",
                f"MergeEnd{basis}",
            ):
                assert normalized_layers(
                    crumble_operations(circuits[gadget_name])
                ) == normalized_layers(deq_operations(source, gadget_name))


def test_rendered_and_visualized_gadgets_omit_identity_operations() -> None:
    source = generator.render_surgery_library(3)
    assert "\n    I " not in source
    assert "\n    CZ " not in source
    for basis in ("XX", "ZZ"):
        for circuit in gadget_circuits(3, basis).values():
            assert "\nI " not in "\n" + str(circuit)
            assert "\nCZ " not in "\n" + str(circuit)


def test_rotated_surface_code_class_supports_square_and_merged_rectangles() -> None:
    square = RotatedSurfaceCode(width=5, height=5)
    merged = RotatedSurfaceCode(width=11, height=5)
    assert square.type_name == "RotatedSurfaceCodeW5H5"
    assert square.num_data_qubits == 25
    assert len(square.stabilizers()) == 24
    assert merged.type_name == "RotatedSurfaceCodeW11H5"
    assert merged.num_data_qubits == 55
    assert len(merged.stabilizers()) == 54


def test_rotated_surface_code_supports_the_even_height_hadamard_extension() -> None:
    extended = RotatedSurfaceCode(width=3, height=6)
    assert extended.type_name == "RotatedSurfaceCodeW3H6"
    assert extended.num_data_qubits == 18
    assert len(extended.stabilizers()) == 17


def test_hadamard_extension_uses_rightward_staggered_geometry() -> None:
    for distance in (3, 5, 7):
        code = HadamardDeformationLayout(distance, "extension")
        checks = code.syndrome_checks()
        syndrome_checks = code.syndrome_checks()
        assert code.num_data_qubits == 2 * distance * distance - 1
        assert len(checks) == code.num_data_qubits - 1
        assert len(syndrome_checks) == code.num_data_qubits - 1
        assert (
            2 * distance - 0.5,
            distance - 0.5,
        ) not in code.data_coordinates().values()
        for index, first in enumerate(checks):
            first_by_qubit = {qubit: pauli for pauli, qubit in first}
            for second in checks[index + 1 :]:
                second_by_qubit = {qubit: pauli for pauli, qubit in second}
                anti_commuting_overlap = sum(
                    first_by_qubit[qubit] != second_by_qubit[qubit]
                    for qubit in first_by_qubit.keys() & second_by_qubit.keys()
                )
                assert anti_commuting_overlap % 2 == 0

        assert all(len({pauli for pauli, _ in check}) == 1 for check in syndrome_checks)
        expected_input = {
            ("Z" if pauli == "X" else "X", tuple(sorted(support)))
            for pauli, support in RotatedSurfaceCode(distance, distance).stabilizers()
            if not all(qubit % distance == distance - 1 for qubit in support)
        }
        actual_input = {
            (
                next(iter({pauli for pauli, _ in check})),
                tuple(sorted(qubit for _, qubit in check)),
            )
            for check in syndrome_checks
            if all(qubit < distance * distance for _, qubit in check)
        }
        assert actual_input == expected_input

        full = HadamardDeformationLayout(distance, "corner_moved")
        assert len(full.syndrome_checks()) == full.num_data_qubits - 1
        assert len(full.syndrome_checks()) == full.num_data_qubits - 1
        assert full.wire(2 * distance - 1, distance - 1) == (full.num_data_qubits - 1)
        full_coordinates = full.data_coordinates()
        bottom_checks = [
            check
            for check in full.syndrome_checks()
            if len(check) == 2
            and all(full_coordinates[qubit][1] == distance - 0.5 for _, qubit in check)
        ]
        assert len(bottom_checks) == distance
        assert all({pauli for pauli, _ in check} == {"X"} for check in bottom_checks)
        assert all(len({pauli for pauli, _ in check}) == 1 for check in full.syndrome_checks())


def test_hadamard_deformation_uses_the_reference_hook_orders() -> None:
    extension_orders = _extension_hook_orders(HadamardDeformationLayout(5, "extension"))
    assert extension_orders[0]["X"] == ((-1, 1), (-1, -1), (1, 1), (1, -1))
    assert extension_orders[6]["X"] == ((-1, 1), (1, 1), (-1, -1), (1, -1))
    assert extension_orders[5]["Z"] == ((-1, 1), (-1, -1), (1, 1), (1, -1))
    assert extension_orders[5]["Z"] == extension_orders[0]["X"]

    corner_orders = _corner_movement_hook_orders(
        HadamardDeformationLayout(5, "corner_moved")
    )
    assert corner_orders[0]["X"] == ((-1, -1), (-1, 1), (1, -1), (1, 1))
    assert corner_orders[4]["X"] == ((-1, -1), (1, -1), (-1, 1), (1, 1))
    assert corner_orders[1]["Z"] == ((-1, -1), (1, -1), (-1, 1), (1, 1))
    assert corner_orders[3]["Z"] == corner_orders[0]["X"]


def test_hadamard_visualizations_follow_extension_and_corner_move() -> None:
    circuits = gadget_circuits(3)
    extend = circuits["HadamardExtend"]
    extend_coordinates = extend.get_final_qubit_coordinates()
    assert extend_coordinates[0] == [0.5, 0.5]
    assert extend_coordinates[6] == [0.5, 2.5]
    assert extend_coordinates[9] == [3.5, 0.5]
    assert extend_coordinates[15] == [3.5, 2.5]
    assert max(extend_coordinates[q][0] for q in range(9)) < min(
        extend_coordinates[q][0] for q in range(9, 17)
    )
    extension = HadamardDeformationLayout(3, "extension")
    all_extension_basis_at = {
        tuple(extend_coordinates[17 + index]): "".join(
            sorted({pauli for pauli, _ in check})
        )
        for index, check in enumerate(extension.syndrome_checks())
    }
    boundary_basis_at = {
        coordinate: basis
        for coordinate, basis in all_extension_basis_at.items()
        if coordinate[0] in (0, 6) or coordinate[1] in (0, 3)
    }
    assert boundary_basis_at == {
        (1.0, 0.0): "Z",
        (4.0, 0.0): "X",
        (6.0, 1.0): "Z",
        (0.0, 2.0): "X",
        (2.0, 3.0): "Z",
        (4.0, 3.0): "Z",
    }
    assert 17 not in {
        qubit
        for qubit, coordinate in extend_coordinates.items()
        if all(value % 1 == 0.5 for value in coordinate)
    }
    for qubit, coordinate in extend_coordinates.items():
        expected_fraction = 0.5 if qubit < 17 else 0
        assert all(value % 1 == expected_fraction for value in coordinate)
    assert len({tuple(point) for point in extend_coordinates.values()}) == len(
        extend_coordinates
    )
    assert "RX 9 10 11 12 13 14 15 16" in str(extend)
    assert str(extend).count("TICK") == 6

    corner = circuits["HadamardCornerMove"]
    corner_coordinates = corner.get_final_qubit_coordinates()
    assert corner_coordinates[17] == [5.5, 2.5]
    for qubit, coordinate in corner_coordinates.items():
        expected_fraction = 0.5 if qubit < 18 else 0
        assert all(value % 1 == expected_fraction for value in coordinate)
    assert "R 17" in str(corner)
    assert str(corner).count("TICK") == 6
    assert len({tuple(point) for point in corner_coordinates.values()}) == len(
        corner_coordinates
    )
    assert {
        qubit: coordinate
        for qubit, coordinate in corner_coordinates.items()
        if qubit < 17
    } == {
        qubit: coordinate
        for qubit, coordinate in extend_coordinates.items()
        if qubit < 17
    }
    full = HadamardDeformationLayout(3, "corner_moved")
    bottom_basis_at = {
        tuple(corner_coordinates[18 + index]): "".join(
            sorted({pauli for pauli, _ in check})
        )
        for index, check in enumerate(full.syndrome_checks())
        if tuple(corner_coordinates[18 + index])[1] == 3
    }
    assert bottom_basis_at == {
        (1.0, 3.0): "X",
        (3.0, 3.0): "X",
        (5.0, 3.0): "X",
    }

    domain_wall_se = circuits["HadamardDomainWallSE"]

    def operations(circuit: stim.Circuit) -> list[str]:
        return [
            line
            for line in str(circuit).splitlines()
            if not line.startswith("QUBIT_COORDS")
        ]

    corner_operations = operations(corner)
    corner_operations[0] = corner_operations[0].replace("R 17 ", "R ")
    assert corner_operations == operations(domain_wall_se)
    assert corner_coordinates == domain_wall_se.get_final_qubit_coordinates()

    shrink = circuits["HadamardShrink"]
    assert "M 0 1 2 3 4 5 6 7 8" in str(shrink)

    for name in (
        "HadamardExtend",
        "HadamardExtensionSE",
        "HadamardCornerMove",
        "HadamardDomainWallSE",
    ):
        # Every local CSS extraction uses four conflict-free CNOT layers.
        body = [
            line
            for line in str(circuits[name]).splitlines()
            if not line.startswith("QUBIT_COORDS")
        ]
        assert sum(line.startswith("CX ") for line in body) == 4
        assert not any(line.startswith("CZ ") for line in body)


def test_post_shrink_hadamard_gadgets_follow_the_horizontal_return_path() -> None:
    circuits = gadget_circuits(3)

    return_extend = circuits["HadamardReturnExtend"]
    coordinates = return_extend.get_final_qubit_coordinates()
    assert coordinates[0] == [1.5, 0.5]
    assert coordinates[4] == [5.5, 0.5]
    assert coordinates[10] == [1.5, 2.5]
    added = {0, 1, 5, 6, 10, 11}
    reset_targets = {
        int(target)
        for line in str(return_extend).splitlines()
        if line.startswith("R ")
        for target in line.split()[1:]
    }
    assert added <= reset_targets

    def operations(circuit: stim.Circuit) -> list[str]:
        return [
            line
            for line in str(circuit).splitlines()
            if not line.startswith("QUBIT_COORDS")
        ]

    initial_round = operations(return_extend)
    initial_round[0] = "R " + " ".join(
        target for target in initial_round[0].split()[1:] if int(target) not in added
    )
    assert initial_round == operations(circuits["HadamardReturnExtensionSE"])

    return_shrink = circuits["HadamardReturnShrink"]
    assert "M 3 4 8 9 13 14" in str(return_shrink)

    northwest = circuits["HadamardSwapQECNW"].get_final_qubit_coordinates()
    southwest = circuits["HadamardSwapQECSW"].get_final_qubit_coordinates()
    assert northwest[0] == [1.5, 0.5]
    assert northwest[9] == [1.0, 1.0]
    assert southwest[0] == [1.0, 1.0]
    assert southwest[9] == [0.5, 0.5]
    assert all(
        value >= 0
        for coordinates in (northwest, southwest)
        for point in coordinates.values()
        for value in point
    )
    for name in ("HadamardSwapQECNW", "HadamardSwapQECSW"):
        schedule = str(circuits[name])
        assert schedule.count("SWAP ") == 1
        assert "SWAP 0 9 1 10 2 11 3 12 4 13 5 14 6 15 7 16 8 17" in schedule
        assert "M 0 1 2 3 4 5 6 7 8" not in schedule


def test_single_patch_geometry_matches_stims_rotated_memory_convention() -> None:
    distance = 3
    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=1
    )
    coordinates = reference.get_final_qubit_coordinates()
    data_by_stim_qubit = {
        qubit: ((int(x) - 1) // 2) + distance * ((int(y) - 1) // 2)
        for qubit, (x, y) in coordinates.items()
        if int(x) % 2 and int(y) % 2
    }
    x_ancillas: set[int] = set()
    interactions: list[tuple[int, int]] = []
    measurement_ancillas: list[int] = []
    for instruction in reference:
        if instruction.name == "H":
            x_ancillas.update(target.value for target in instruction.targets_copy())
        elif instruction.name == "CX":
            targets = [target.value for target in instruction.targets_copy()]
            interactions.extend(zip(targets[::2], targets[1::2]))
        elif instruction.name == "MR":
            measurement_ancillas = [
                target.value for target in instruction.targets_copy()
            ]
            break

    expected: set[tuple[str, tuple[int, ...]]] = set()
    for ancilla in measurement_ancillas:
        support = {
            data_by_stim_qubit[target]
            for control, target in interactions
            if control == ancilla and target in data_by_stim_qubit
        } | {
            data_by_stim_qubit[control]
            for control, target in interactions
            if target == ancilla and control in data_by_stim_qubit
        }
        expected.add(("X" if ancilla in x_ancillas else "Z", tuple(sorted(support))))

    actual = {
        (pauli, tuple(sorted(support)))
        for pauli, support in RotatedSurfaceCode(distance, distance).stabilizers()
    }
    assert actual == expected

    displayed = gadget_circuits(distance)["SyndromeExtraction"]
    displayed_coordinates = displayed.get_final_qubit_coordinates()
    for stim_qubit, data_qubit in data_by_stim_qubit.items():
        x, y = coordinates[stim_qubit]
        assert displayed_coordinates[data_qubit] == [x / 2, y / 2]

    for index, stabilizer in enumerate(
        RotatedSurfaceCode(distance, distance).stabilizers()
    ):
        pauli, support = stabilizer
        matching_ancilla = next(
            ancilla
            for ancilla in measurement_ancillas
            if (
                "X" if ancilla in x_ancillas else "Z",
                tuple(
                    sorted(
                        {
                            data_by_stim_qubit[target]
                            for control, target in interactions
                            if control == ancilla and target in data_by_stim_qubit
                        }
                        | {
                            data_by_stim_qubit[control]
                            for control, target in interactions
                            if target == ancilla and control in data_by_stim_qubit
                        }
                    )
                ),
            )
            == (pauli, tuple(sorted(support)))
        )
        x, y = coordinates[matching_ancilla]
        assert displayed_coordinates[distance * distance + index] == [x / 2, y / 2]


def test_visualization_circuits_cover_the_merge_and_use_compact_coordinates() -> None:
    circuits = gadget_circuits(3)
    assert {
        "MergeBeginXX",
        "MergedSEXX",
        "MergeEndXX",
        "TransversalHadamard",
        "HadamardExtend",
        "HadamardExtensionSE",
        "HadamardCornerMove",
        "HadamardDomainWallSE",
        "HadamardShrink",
        "HadamardReturnExtend",
        "HadamardReturnExtensionSE",
        "HadamardReturnShrink",
        "HadamardSwapQECNW",
        "HadamardSwapQECSW",
    } <= set(circuits)
    patch_coordinates = circuits["SyndromeExtraction"].get_final_qubit_coordinates()
    assert patch_coordinates[0] == [0.5, 0.5]
    assert patch_coordinates[9] == [1.0, 1.0]
    assert patch_coordinates[13] == [1.0, 0.0]
    for qubit, coordinate in patch_coordinates.items():
        if qubit < 9:
            assert all(value % 1 == 0.5 for value in coordinate)
        else:
            assert all(value % 1 == 0 for value in coordinate)

    coordinates = circuits["MergeBeginXX"].get_final_qubit_coordinates()
    assert coordinates[18] == [3.5, 0.5]
    for qubit, coordinate in coordinates.items():
        if qubit < 21:  # Data, including the d=3 merge seam.
            assert all(value % 1 == 0.5 for value in coordinate)
        else:
            assert all(value % 1 == 0 for value in coordinate)
    schedule = str(circuits["SyndromeExtraction"])
    assert "CX" in schedule
    assert "MPP" not in schedule
    assert "TICK" in schedule

    # The two disjoint post-split patch rounds share every time layer.
    merge_end_schedule = str(circuits["MergeEndXX"])
    assert merge_end_schedule.count("TICK") == 6
    assert "R 21 24 27 28 29 32 35 36" in merge_end_schedule


def test_visualized_gadget_wires_have_unique_coordinates() -> None:
    for distance in (3, 5, 7):
        for basis in ("XX", "ZZ"):
            for name, circuit in gadget_circuits(distance, basis).items():
                coordinates = circuit.get_final_qubit_coordinates()
                points = [tuple(point) for point in coordinates.values()]
                assert len(points) == len(set(points)), (distance, basis, name)
