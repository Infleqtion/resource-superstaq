# surface-code-deq

A small, generated DEQ library for rotated surface-code gadgets at every odd
distance `d ≥ 3`.

## What the library provides

`surface_code_deq.library.build_library` is the single source of generated DEQ
files. `RotatedSurfaceCode(width, height)` is the construction API, while
each emitted `CODE` uses a concrete compiler-safe name such as
`RotatedSurfaceCodeW3H3`.

The implementation has three layers:

- `rotated_surface_code.py`, `surgery_geometry.py`, `hadamard_geometry.py`, and
  `prepare_y_geometry.py` define code geometry and stabilizers.
- `gadgets/core.py` defines the shared instruction, schedule, typed-port, and
  gadget model. `gadgets/memory.py`, `gadgets/surgery.py`,
  `gadgets/hadamard.py`, and `gadgets/prepare_y.py` are the physical source of
  both DEQ output and Stim diagrams. `library.py` only assembles complete
  libraries.
- `surface_code_deq.verification` contains logical-gate specifications, Choi
  programs, logical-error-rate runners, and fault-distance checks. Verification
  programs are not embedded in production libraries.

For a compact audit path, start with `gadgets/memory.py` for ordinary patch
operations and `gadgets/surgery.py` for lattice surgery. Each `GadgetSpec`
contains the physical schedule, display coordinates, typed inputs and outputs,
and any readout expression. DEQ generation and Stim visualization render that
same object, so there is no second schedule implementation to compare by hand.

Install the package in editable mode once so tests, notebooks, and commands all
use the same imports:

```bash
python -m pip install -e '.[dev]'
```

The unified `surface-code` library contains:

- Ordinary patch operations: `PrepareX`, `PrepareZ`, syndrome extraction,
  logical measurements, and virtual frame updates.
- Fault-tolerant lattice surgery: horizontal `MXX` and vertical `MZZ`, each
  with typed begin, merge-round, and end gadgets.
- Composed Clifford operations: `LogicalCNOTD*`, `LogicalSD*`, and
  `LogicalHadamardD*`.
- `PrepareY`, an encoded `|+i⟩` preparation built from the XXZZ-boundary state
  and its reverse-time diagonal-twist transition.

The physical schedules use fresh check ancillas and conflict-free `CX` layers,
not abstract `MPP` instructions. CNOT uses a `|+⟩` mediator through
`MZZ(control, mediator)`, `MXX(mediator, target)`, and a mediator-Z readout.

## Generate and transpile

```bash
surface-code-deq-generate --distance 3 \
  --merged-rounds 3 \
  --out generated/rotated_surface_code_d3.deq
python -m deq transpile generated/rotated_surface_code_d3.deq \
  --out rotated_surface_code_d3.deq.jit --jobs 1
```

## Validate logical channels

The Choi checker prepares encoded Bell pairs, applies each S, H, or CNOT
gadget, and measures a complete stabilizer description of the expected Choi
state. This checks both the logical Clifford action and every output code-space
stabilizer. Use `--distance` to select the code distance (default: 3).

```bash

surface-code-deq-verify
```

## Check fault distance

The graphlike fault-distance check uses the same Choi construction with ideal
terminal readout. To also export exact, non-graphlike fault-distance MaxSAT
problems, use Stim's circuit-level encoding:

```bash
surface-code-deq-distance --distance 5 \
  --sat-problem-dir d5-fault-distance
```

This writes one WDIMACS `.wcnf` problem per independent logical-Pauli check. Solve
each with a MaxSAT solver; its optimal cost is the corresponding full
circuit-level fault distance. The usual console report remains graphlike.

## Run a logical-error-rate experiment

The experiment takes a text file containing one `H`, `S`, or `CX` gate per
line. It runs the circuit and its inverse as a noisy SI1000 cycle. A zero-noise
window-decoder preflight runs first; it must pass before a noisy LER is
reported (unless `--skip-ideal-check` is explicit).

```bash

surface-code-deq-ler \
  --circuit examples/identity_clifford.txt --num-logical-qubits 1 --distance 3 \
  --noise-p 0.001 --shots 100000 --errors 100
```

`examples/two_qubit_clifford.txt` shows the input format. Replace the identity
input with any H/S/CX gate list once its zero-noise preflight passes.

## PyMatching graphlike reference decoder

`pymatching_window_decoder.py` is DEQ's `black-box-python` adapter for a
strictly graphlike decoding hypergraph. It retains one `pymatching.Matching`
per DEQ hypergraph and returns the original DEQ hyperedge ids, so it rejects
parallel endpoint pairs instead of allowing PyMatching to merge them.

Use the existing experiment with only the decoder selection changed:

```bash
surface-code-deq-ler \
  --circuit examples/identity_clifford.txt --num-logical-qubits 1 --distance 3 \
  --noise-p 0.001 --shots 100000 --errors 100 \
  --decoder black-box-python \
  --decoder-config '{"file":"pymatching_window_decoder.py","parallel":1}'
```

DEQ's Python hypergraph protocol supplies only detector vertices and a fault
probability. It does not tag a one-detector fault as a physical boundary versus
the window coordinator's future-time carry interface. Consequently the adapter
refuses an unclassified one-detector fault. After independently auditing a
window, pass Python-specific options under `py_config`, for example
`"py_config":{"physical_boundary_vertices":[...],"timing":true}`. The
`assume_all_boundaries_physical` switch exists solely for an independently
verified test graph. With `timing`, the adapter prints count, total, mean, and
p50/p95/p99 `Matching.decode` latency at reset/interpreter shutdown.

The current noisy SI1000 d=3 identity window is deliberately rejected: its
first decoder hypergraph has support counts 1:8, 2:7, 3:5, and 4:3. This is a
non-graphlike circuit-level model, so a PyMatching window run would not test
the same decoder problem without an explicit graphification approximation.

For a known identity circuit, `--no-inverse` runs the supplied gates exactly
once. For example, `examples/ten_cnot.txt` applies ten consecutive CNOTs:

```bash

surface-code-deq-ler \
  --circuit examples/ten_cnot.txt --num-logical-qubits 2 --distance 3 \
  --no-inverse --noise-p 0.001 --shots 100000 --errors 100
```

## Add SI1000 noise to a library

```bash

surface-code-deq-generate --distance 3 \
  --noise-model si1000 --noise-p 0.001 \
  --out generated/rotated_surface_code_d3_si1000_p0.001.deq
```

## Hadamard layout

The transversal-H protocol follows Fig. 2 of Gehér et al. The Fig. 2(c)
extension retains the original patch on the left and prepares `d²−1` right-side data
sites, for `2d²−1` data wires. Its geometry is the paper protocol after a
90-degree rotation and a mirror reflection. The left side preserves the exact
H-conjugated `RotatedSurfaceCode` input checkerboard; its top checks are Z,
while the new-side top checks are X and the outer right checks are Z. The
entire lower boundary also uses Z checks. The following Fig. 2(d) corner move
resets the missing bottom-right site in `|0⟩` and replaces those lower Z
half-checks with X half-checks on the complementary stagger, producing the
full `2d²` patch.

After `HadamardShrink` retains the right square, the transformed Fig. 2(f--h)
return deformation grows `d-1` columns to its left and then measures away the
right `d-1` columns. This leaves the patch one data column to the right of its
starting footprint. The two final SWAP-QEC steps move northwest and southwest,
for a net one-column translation back to the left.

## Cultivation reference and manual DEQ gadgets

The cultivation package provides an actual-T d=3 check core, an exact small
state-vector reference, and explicit DEQ check/error models consuming external
raw measurements. It validates the handoff to color-code stabilization and
terminal decoder-selected Pauli corrections. Input preparation and stabilization
are ideal in this regression; physical injection and surface-code escape remain
outside its scope.

```bash
python tools/check_cultivation_handoff.py --coordinator window --rounds 2 \
  --export-prefix generated/cultivation_d3_reference
```

See [cultivation architecture](docs/cultivation-architecture.md) for conventions,
manual model construction, state ownership, and integration tests. The checker
requires the dependencies in `pyproject.toml`; the local `.venv-deq` environment
can be used where the system Python has an older protobuf runtime.

## Explore the gadgets

Open [the cultivation walkthrough](examples/cultivation_flow.html) in a browser
to explore seven recorded scenarios with both DEQ coordinators. Click through
the physical circuit, acceptance checks, stabilization, decoder payload, and
logical fidelity; compare alternative Pauli corrections in the final stage.
This standalone page works offline and displays results from actual runs.

For live simulations, open
[notebooks/explore_cultivation.ipynb](notebooks/explore_cultivation.ipynb) in
Jupyter, run its control cell, and click **Run physical simulation + DEQ**.
Choose the fault, coordinator, stabilization rounds, and seed. The notebook
requires the `notebooks` optional dependencies and uses `.venv-deq/bin/python`
as its simulation worker when available. DEQ's local services must be permitted.
To regenerate the browser page:

```bash
python tools/demo_cultivation.py --export-html examples/cultivation_flow.html
```

Open [notebooks/explore_rotated_surface_code_deq.ipynb](notebooks/explore_rotated_surface_code_deq.ipynb)
in Jupyter for interactive and full-page Crumble views of the ordinary and
lattice-surgery gadget bodies.
