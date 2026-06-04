# QSVT Architecture Problem Statement

Quantum singular value transformation (QSVT) is a unifying primitive
behind many major quantum algorithms.  Classical computers do not
build a new physical circuit for every program. The hardware contains arithmetic units, memory interfaces, and control logic that are
called repeatedly by many different programs. Programs change the instruction
stream and data, while the underlying circuit fabric remains reusable.

Quantum computing is usually described differently. For each algorithm, we often
compile a complete circuit: a full gate-level schedule whose structure is tied
to the specific problem. This makes quantum architecture feel less like a reusable processor and more like a custom circuit. The question is whether QSVT can change that framing.

The core problem is to determine whether QSVT can be elevated from
an algorithmic abstraction into an architecture-level primitive. In particular,
we want to ask whether a quantum computer could expose a programmable
mechanism for running many QSVT instances by changing the block encoding, phase
program, projectors, and measurements.

The architectural hypothesis is that QSVT may provide a reusable quantum
execution skeleton. Instead of recompiling an entire monolithic quantum circuit
for every algorithm, the system might compile or load a smaller set of program
objects: a block encoding, a phase sequence, projector definitions, and a measurement instruction. The hardware or compiler stack would then repeatedly execute the same QSVT control pattern, analogous to how classical programs repeatedly call the same arithmetic circuits.

This does not mean QSVT automatically eliminates compilation. The block encoding
may still be expensive, application-specific, and difficult to synthesize. The
more modest question is whether QSVT moves a large part of the repeated
algorithmic structure out of bespoke circuit compilation and into a reusable
architecture-level interface.

## Algorithm Decomposition

Across the QSVT algorithms in the Chuang grand-unification paper, the
repeatable part is the same architectural skeleton:

```text
QSVT_RUN(block_encoding, projectors, phase_sequence)
  = alternating projector-controlled phase rotations and calls to U or U dagger
```

The parts that change from algorithm to algorithm can be separated into
preprocessing, which prepares the QSVT program objects, and postprocessing,
which extracts the algorithm's answer from the transformed block.

| Algorithm | Preprocessing | Repeatable QSVT Operator | Postprocessing |
|---|---|---|---|
| Unstructured search | Prepare the uniform input state, define the oracle access to the marked state, choose the search polynomial, and identify the simple block encoding whose singular value is the marked-state overlap. | Run QSVT with the search phase sequence to transform the marked-state overlap so the target state has high amplitude. | Measure the final register in the computational basis to read out the marked item. |
| Eigenvalue threshold | Prepare or assume access to a block encoding of `H / alpha`, choose the threshold polynomial, prepare the witness state with promised overlap, and set the threshold parameters. | Run QSVT with the threshold/filter phase sequence to separate low-eigenvalue and high-eigenvalue cases. | Measure the ancilla over repeated trials and compare the observed frequency to the decision threshold. |
| Phase estimation | Prepare the eigenstate, initialize the classical feedback variable `theta`, and at each bit position construct a block encoding of `A_j(theta) = (I + exp(-2 pi i theta) U^(2^j)) / 2`. | Run the same QSVT discriminator pattern on each `A_j(theta)` to extract one phase bit. | Measure the ancilla, feed the bit back into `theta`, and output the final phase estimate after the loop. |
| Hamiltonian simulation | Prepare a block encoding of `H / alpha`, choose approximation degree from `t` and `epsilon`, and synthesize the cosine and sine phase sequences. | Run QSVT twice on the same block encoding: once for the cosine polynomial and once for the sine polynomial. | Combine the two transformed blocks to obtain a block encoding of `exp(-i H t)`. |
| Matrix inversion | Prepare a block encoding of `A dagger`, estimate or assume the condition number `kappa`, and choose the polynomial approximating `1 / x` on the promised singular-value interval. | Run QSVT with the matrix-inversion phase sequence to transform singular values by the approximate reciprocal polynomial. | Use the resulting block encoding as an inverse operator; for linear systems, apply it to `|b>`, then postselect, amplify, and normalize as needed. |

This decomposition suggests the architectural proposal more clearly: the
preprocessing stage builds a QSVT instruction packet, the QSVT operator is the
candidate reusable quantum architecture, and the postprocessing stage converts
the transformed block into the algorithm-specific answer.

## Implementation MVP

One possible way to investigate this hypothesis is to implement a general QSVT
`Bloq` in Qualtran. The purpose would not be to claim that QSVT is already a
complete architecture, but to create a concrete object that lets us ask better
resource-estimation and compilation questions.

The experimental `Bloq` might look conceptually like:

```text
QSVTProgramBloq(
    block_encoding,
    projector,
    phase_program,
    degree,
    convention,
)
```

This object could be:

- As a compact resource model, it counts the repeated structure without
  expanding every step.
- A circuit-generation object, it unrolls the QSVT loop into an ordinary
  acyclic quantum circuit when a concrete circuit is needed.

Questions worth exploring in such an implementation:

- Should the phase program be an explicit list of angles, a callable `phi(k)`,
  or a symbolic polynomial descriptor?
- Should projectors be represented as generic Qualtran `Bloq`s, reflections
  around prepared states, or a smaller set of architecture-native projector
  operations?
- Should `U` be any `Bloq`, only a Qualtran `BlockEncoding`, or specifically a
  block encoding with known `alpha`, `epsilon`, signal state, and register
  widths?
- Should the `Bloq` support both one-projector and two-projector QSVT
  conventions?
- How much of the loop can be counted symbolically before the block encoding
  must be decomposed?
- What is the smallest benchmark where this `Bloq` says something useful:
  search, Hamiltonian simulation, phase estimation, or matrix inversion?

The implementation experiment would give us a clean place to separate the
architectural loop from the algorithm-specific preprocessing and
postprocessing. If most costs still live inside the block encoding, that would
be important evidence against treating QSVT as a hardware primitive. If the
projector-phase and `U`/`U dagger` loop dominates across many examples, that
would strengthen the case for a QSVT-native architecture.
