"""Graphlike PyMatching adapter for DEQ's ``black-box-python`` decoder.

DEQ's Python protocol exposes a decoding hypergraph containing only detector
vertices and fault probabilities.  In particular, it does not identify
whether a one-detector edge ends at a physical code boundary or at the window
coordinator's future-time cut.  To avoid incorrectly absorbing a carried
defect at that cut, one-detector edges require explicit classification through
``physical_boundary_vertices`` (or the deliberately unsafe
``assume_all_boundaries_physical`` test switch).

Optional configuration:

``physical_boundary_vertices``
    Iterable of detector vertex ids known to be physical boundaries.
``assume_all_boundaries_physical``
    Permit every one-detector edge.  Intended only for a graph whose boundary
    representation has been independently verified.
``timing``
    Print PyMatching decode latency summary when ``reset`` is called.
``timing_percentiles``
    Percentiles to report (default: ``[50, 95, 99]``).
``diagnostic_path`` or ``DEQ_PYMATCHING_DIAGNOSTIC_PATH``
    Write a compact JSON description of the supplied hypergraph.  This is
    useful for auditing a DEQ window before enabling boundary edges.
"""

from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import pymatching


class HypergraphEligibilityError(ValueError):
    """The DEQ hypergraph cannot be represented without changing its meaning."""


def _support_summary(hyperedges: list[Any]) -> dict[int, int]:
    summary: dict[int, int] = {}
    for hyperedge in hyperedges:
        support = len(hyperedge.vertices)
        summary[support] = summary.get(support, 0) + 1
    return summary


def _format_summary(summary: dict[int, int]) -> str:
    return ", ".join(f"support {size}: {count}" for size, count in sorted(summary.items())) or "empty"


class Decoder:
    """Retained PyMatching graph for one fixed, graphlike DEQ hypergraph."""

    def __init__(self, hypergraph: Any, config: dict) -> None:
        self._config = dict(config or {})
        self._vertex_num = int(hypergraph.vertex_num)
        if self._vertex_num < 0:
            raise HypergraphEligibilityError(f"vertex_num must be non-negative, got {self._vertex_num}")
        self._hyperedges = list(hypergraph.hyperedges)
        self._support_summary = _support_summary(self._hyperedges)
        self._timing_enabled = bool(self._config.get("timing", False))
        self._timing_percentiles = tuple(self._config.get("timing_percentiles", (50, 95, 99)))
        if any(not 0 <= float(percentile) <= 100 for percentile in self._timing_percentiles):
            raise ValueError("timing_percentiles must be numbers in [0, 100]")
        self._latencies_ns: list[int] = []
        self._reported_calls = 0
        if self._timing_enabled:
            atexit.register(self._report_timing)

        boundary_vertices = self._config.get("physical_boundary_vertices", ())
        try:
            self._physical_boundary_vertices = {int(vertex) for vertex in boundary_vertices}
        except TypeError as error:
            raise ValueError("physical_boundary_vertices must be an iterable of vertex ids") from error
        self._assume_all_boundaries_physical = bool(
            self._config.get("assume_all_boundaries_physical", False)
        )
        self._write_diagnostic_if_requested()

        self.matching = pymatching.Matching()
        self._active_fault_ids: set[int] = set()
        seen_endpoints: dict[tuple[int, ...], int] = {}
        for edge_index, hyperedge in enumerate(self._hyperedges):
            vertices = tuple(int(vertex) for vertex in hyperedge.vertices)
            support = len(vertices)
            if support not in (1, 2):
                raise HypergraphEligibilityError(
                    "PyMatching requires graphlike DEQ hyperedges: "
                    f"edge index {edge_index} has support size {support}; "
                    f"support counts: {_format_summary(self._support_summary)}"
                )
            if len(set(vertices)) != support:
                raise HypergraphEligibilityError(
                    f"edge index {edge_index} repeats a detector vertex {vertices}; "
                    f"support size {support}; support counts: {_format_summary(self._support_summary)}"
                )
            for vertex in vertices:
                if not 0 <= vertex < self._vertex_num:
                    raise HypergraphEligibilityError(
                        f"edge index {edge_index} references detector {vertex}, outside "
                        f"0..{self._vertex_num - 1}; support size {support}; "
                        f"support counts: {_format_summary(self._support_summary)}"
                    )
            probability = float(hyperedge.probability)
            if not math.isfinite(probability) or not 0 <= probability < 0.5:
                raise HypergraphEligibilityError(
                    f"edge index {edge_index} has probability {probability!r}; expected p in [0, 0.5); "
                    f"support size {support}; support counts: {_format_summary(self._support_summary)}"
                )
            # Impossible faults need no boundary classification and cannot be
            # returned by PyMatching, so intentionally omit them altogether.
            if probability == 0:
                continue
            if support == 1 and not (
                self._assume_all_boundaries_physical
                or vertices[0] in self._physical_boundary_vertices
            ):
                raise HypergraphEligibilityError(
                    "DEQ does not tag one-detector edges as physical boundaries versus "
                    "the future-time carry interface. Refusing to terminate a possible "
                    f"carried defect: edge index {edge_index}, detector {vertices[0]}, "
                    f"support size {support}; support counts: {_format_summary(self._support_summary)}. "
                    "Pass physical_boundary_vertices only after independently auditing the window."
                )
            endpoint_key = tuple(sorted(vertices))
            previous = seen_endpoints.get(endpoint_key)
            if previous is not None:
                raise HypergraphEligibilityError(
                    "PyMatching would merge repeated endpoint pairs and lose DEQ fault ids: "
                    f"edge index {edge_index} duplicates edge index {previous} at {endpoint_key}; "
                    f"support size {support}; support counts: {_format_summary(self._support_summary)}"
            )
            seen_endpoints[endpoint_key] = edge_index
            weight = math.log((1 - probability) / probability)
            if support == 2:
                self.matching.add_edge(*vertices, fault_ids={edge_index}, weight=weight)
            else:
                self.matching.add_boundary_edge(vertices[0], fault_ids={edge_index}, weight=weight)
            self._active_fault_ids.add(edge_index)

    def _write_diagnostic_if_requested(self) -> None:
        destination = self._config.get("diagnostic_path") or os.environ.get(
            "DEQ_PYMATCHING_DIAGNOSTIC_PATH"
        )
        if not destination:
            return
        data = {
            "vertex_num": self._vertex_num,
            "support_counts": {str(size): count for size, count in self._support_summary.items()},
            "one_detector_edges": [
                {"edge_index": index, "vertex": int(edge.vertices[0]), "probability": float(edge.probability)}
                for index, edge in enumerate(self._hyperedges)
                if len(edge.vertices) == 1
            ],
        }
        Path(destination).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def decode(self, syndrome: list[int]) -> list[int]:
        dense_syndrome = np.zeros(self._vertex_num, dtype=np.uint8)
        seen: set[int] = set()
        for vertex in syndrome:
            vertex = int(vertex)
            if not 0 <= vertex < self._vertex_num:
                raise ValueError(f"syndrome detector {vertex} outside 0..{self._vertex_num - 1}")
            if vertex in seen:
                raise ValueError(f"sparse syndrome contains detector {vertex} more than once")
            seen.add(vertex)
            dense_syndrome[vertex] = 1
        started = perf_counter_ns() if self._timing_enabled else 0
        correction = self.matching.decode(dense_syndrome)
        if self._timing_enabled:
            self._latencies_ns.append(perf_counter_ns() - started)
        return [
            fault_id
            for fault_id in np.flatnonzero(correction).tolist()
            if fault_id in self._active_fault_ids
        ]

    def _report_timing(self) -> None:
        if len(self._latencies_ns) > self._reported_calls:
            samples = np.asarray(self._latencies_ns, dtype=np.float64)
            percentiles = ", ".join(
                f"p{percentile:g}={np.percentile(samples, percentile) / 1e6:.6f}ms"
                for percentile in self._timing_percentiles
            )
            print(
                "PyMatching decode latency: "
                f"count={len(samples)} total={samples.sum() / 1e9:.6f}s "
                f"mean={samples.mean() / 1e6:.6f}ms {percentiles}",
                flush=True,
            )
            self._reported_calls = len(self._latencies_ns)

    def reset(self) -> None:
        if self._timing_enabled:
            self._report_timing()
