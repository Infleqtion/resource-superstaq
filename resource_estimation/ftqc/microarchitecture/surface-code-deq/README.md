# surface-code-deq

A small, generated DEQ library for rotated surface-code gadgets at every odd
distance `d ≥ 3`.

## What the library provides

`tools/generate_rotated_surface_code_deq.py` is the single source of generated
DEQ files. `RotatedSurfaceCode(width, height)` is the construction API, while
each emitted `CODE` uses a concrete compiler-safe name such as
`RotatedSurfaceCodeW3H3`.

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
.venv-deq/bin/python tools/generate_rotated_surface_code_deq.py --distance 3 \
  --operation surface-code --merged-rounds 3 \
  --out generated/rotated_surface_code_d3.deq
.venv-deq/bin/python -m deq transpile generated/rotated_surface_code_d3.deq \
  --out /tmp/rotated_surface_code_d3.deq.jit --jobs 1
```

## Validate logical channels

The Choi checker transpiles d=3, d=5, and d=7 libraries, then uses DEQ's
frame-aware JIT runtime to require the expected signed logical-Pauli
stabilizers of the composed S, H, and CNOT channels.

```bash

.venv-deq/bin/python tools/check_deq_gadget_semantics.py
```

## Check fault distance

The graphlike fault-distance check uses the same Choi construction with ideal
terminal readout. To also export exact, non-graphlike fault-distance MaxSAT
problems, use Stim's circuit-level encoding:

```bash
.venv-deq/bin/python tools/validate_logical_gadgets.py --distance 5 \
  --sat-problem-dir /tmp/d5-fault-distance
```

This writes one WDIMACS `.wcnf` problem per independent Choi stabilizer. Solve
each with a MaxSAT solver; its optimal cost is the corresponding full
circuit-level fault distance. The usual console report remains graphlike.

## Run a logical-error-rate experiment

The experiment takes a text file containing one `H`, `S`, or `CX` gate per
line. It runs the circuit and its inverse as a noisy SI1000 cycle. A zero-noise
window-decoder preflight runs first; it must pass before a noisy LER is
reported (unless `--skip-ideal-check` is explicit).

```bash

.venv-deq/bin/python tools/run_logical_clifford_ler.py \
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
.venv-deq/bin/python tools/run_logical_clifford_ler.py \
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

.venv-deq/bin/python tools/run_logical_clifford_ler.py \
  --circuit examples/ten_cnot.txt --num-logical-qubits 2 --distance 3 \
  --no-inverse --noise-p 0.001 --shots 100000 --errors 100
```

## Add SI1000 noise to a library

```bash

.venv-deq/bin/python tools/generate_rotated_surface_code_deq.py --distance 3 \
  --operation surface-code --noise-model si1000 --noise-p 0.001 \
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

## Explore the gadgets

Open [notebooks/explore_rotated_surface_code_deq.ipynb](notebooks/explore_rotated_surface_code_deq.ipynb)
in Jupyter for interactive and full-page Crumble views of the ordinary and
lattice-surgery gadget bodies.
