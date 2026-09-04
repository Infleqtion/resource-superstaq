"""Tests for the conservative DEQ/PyMatching graphlike adapter."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
DECODER_PATH = ROOT / "pymatching_window_decoder.py"
SPEC = importlib.util.spec_from_file_location("pymatching_window_decoder", DECODER_PATH)
assert SPEC and SPEC.loader
decoder_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = decoder_module
SPEC.loader.exec_module(decoder_module)


class Edge:
    def __init__(self, vertices: list[int], probability: float = 0.1) -> None:
        self.vertices = vertices
        self.probability = probability


class Hypergraph:
    def __init__(self, vertex_num: int, hyperedges: list[Edge]) -> None:
        self.vertex_num = vertex_num
        self.hyperedges = hyperedges


def test_graphlike_correction_uses_original_deq_edge_indices() -> None:
    hypergraph = Hypergraph(
        3,
        [
            Edge([0, 1]),  # DEQ edge 0
            Edge([1, 2]),  # DEQ edge 1
            Edge([2]),  # DEQ edge 2, physical boundary
        ],
    )
    decoder = decoder_module.Decoder(
        hypergraph, {"physical_boundary_vertices": [2]}
    )

    assert decoder.decode([0, 1]) == [0]
    assert decoder.decode([2]) == [2]


@pytest.mark.parametrize("bad_vertices, support", [([], 0), ([0, 1, 2], 3)])
def test_non_graphlike_input_reports_edge_and_support_summary(
    bad_vertices: list[int], support: int
) -> None:
    hypergraph = Hypergraph(4, [Edge([0, 1]), Edge(bad_vertices)])

    with pytest.raises(decoder_module.HypergraphEligibilityError) as error:
        decoder_module.Decoder(hypergraph, {})

    message = str(error.value)
    assert "edge index 1" in message
    assert f"support size {support}" in message
    assert "support 2: 1" in message
    assert f"support {support}: 1" in message


@pytest.mark.parametrize(
    "hypergraph, config",
    [
        (Hypergraph(2, [Edge([0, 1]), Edge([1, 0])]), {}),
        (
            Hypergraph(1, [Edge([0]), Edge([0])]),
            {"physical_boundary_vertices": [0]},
        ),
    ],
)
def test_duplicate_endpoints_report_both_deq_edge_indices(
    hypergraph: Hypergraph, config: dict
) -> None:

    with pytest.raises(decoder_module.HypergraphEligibilityError) as error:
        decoder_module.Decoder(hypergraph, config)

    assert "edge index 1 duplicates edge index 0" in str(error.value)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("DEQ_RUN_INTEGRATION") != "1",
    reason="set DEQ_RUN_INTEGRATION=1 to run the DEQ subprocess integration test",
)
def test_noiseless_existing_d3_window_runs_with_pymatching(tmp_path: Path) -> None:
    """The existing d=3 identity fixture supplies an empty graph at p=0."""
    command = [
        sys.executable,
        "tools/run_logical_clifford_ler.py",
        "--circuit",
        "examples/identity_clifford.txt",
        "--num-logical-qubits",
        "1",
        "--distance",
        "3",
        "--noise-p",
        "0",
        "--shots",
        "2",
        "--errors",
        "1",
        "--ideal-shots",
        "2",
        "--batch-size",
        "1",
        "--jobs",
        "1",
        "--work-dir",
        str(tmp_path),
        "--decoder",
        "black-box-python",
        "--decoder-config",
        '{"file":"pymatching_window_decoder.py","parallel":1}',
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    assert "Logical errors: 0" in result.stdout
