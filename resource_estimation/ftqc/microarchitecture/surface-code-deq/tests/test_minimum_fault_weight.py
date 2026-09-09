import stim
from surface_code_deq.verification import distance as minimum_fault_weight


def test_shortest_error_sat_problem_exports_wdimacs() -> None:
    circuit = stim.Circuit(
        """
        R 0
        X_ERROR(0.001) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    problem = minimum_fault_weight._shortest_error_sat_problem(circuit)

    assert problem.startswith("p wcnf ")
    assert problem.endswith("0\n")


def test_shortest_error_rc2_cost() -> None:
    circuit = stim.Circuit("X_ERROR(0.001) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")

    assert minimum_fault_weight._shortest_error_rc2_cost(circuit) == 1
