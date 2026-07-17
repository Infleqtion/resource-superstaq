Gate-set Compilation and Synthesis
==================================

Resource estimation requires a circuit to be expressed using operations that
the downstream compiler and architecture models understand. An arbitrary
input circuit may contain rotations, matrix gates, or other high-level
operations that do not have directly defined resource costs.

resource-superstaq addresses this with a two-stage gate-set compilation
pipeline:

#. Compile the input circuit into Clifford operations and arbitrary-angle Rz
   rotations.
#. Approximate the remaining Rz rotations using Clifford and T gates.

The resulting circuit can then be passed to the architecture-specific
compiler.

Compilation pipeline
--------------------

The current compilation flow is:

.. code-block:: text

   Input Cirq circuit
           |
           v
   Clifford+Rz compilation
           |
           v
   Clifford+T synthesis
           |
           v
   Architecture and layout selection
           |
           v
   Compilation to logical primitives
           |
           v
   Resource estimation

The gate-set compilation stages affect the final estimate. In particular,
rotation-synthesis accuracy affects the number of T gates, and T gates require
magic-state resources in the architecture model.

Target gate sets
----------------

A target gate set defines the operations that should remain after a compilation
stage. Operations outside the target are decomposed or synthesized into
supported operations.

The main target-gate-set constructors are:

``clifford_rz_gateset``
   Constructs the target used for the first compilation stage.

``clifford_t_gateset``
   Constructs the target used for Clifford+T synthesis.

Both targets are passed to ``compile_gateset``:

.. code-block:: python

   compiled_circuit = res.compile_gateset.compile_gateset(
       circuit,
       gateset=target_gateset,
   )

For target gate sets other than the repository's Clifford+T target,
``compile_gateset`` delegates to Cirq's
``optimize_for_target_gateset`` function. The Clifford+T target is handled by
the repository's custom Rz-synthesis implementation.

Compile to Clifford+Rz
----------------------

The first stage converts a general Cirq circuit into an intermediate
Clifford+Rz representation:

.. code-block:: python

   clifford_rz_circuit = res.compile_gateset.compile_gateset(
       circuit,
       gateset=res.compile_gateset.clifford_rz_gateset(),
   )

This stage uses ``CliffRzGateset``, a Cirq two-qubit compilation target.

Its compilation and postprocessing steps include:

* Removing negligible operations.
* Decomposing unsupported two-qubit operations.
* Expressing CZ in terms of CNOT and Hadamard when encountered directly by the
  two-qubit decomposer.
* Merging compatible single-qubit operations.
* Converting phased-X operations into Z rotations, Hadamards, and X gates.
* Moving Z operations through the circuit where commutation permits.
* Recognizing special Z rotations as exact Clifford operations.
* Leaving other Z rotations as arbitrary-angle ``cirq.Rz`` operations.
* Aligning the resulting circuit and synchronizing terminal measurements.

For the computational portion of the circuit, the resulting representation is
built from operations such as:

* H
* S
* X
* Z
* CNOT
* Rz

Measurements and supported non-unitary operations can also be retained.

Why use an intermediate representation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separating general decomposition from rotation synthesis makes the two
problems easier to reason about.

The first stage normalizes high-level circuit operations into a small basis
containing arbitrary Z rotations. The second stage can then focus exclusively
on approximating those rotations.

This also makes it possible to inspect the Clifford+Rz circuit before choosing
a rotation-synthesis tolerance.

Clifford+Rz tolerance
~~~~~~~~~~~~~~~~~~~~~

The Clifford+Rz target accepts an absolute tolerance:

.. code-block:: python

   target = res.compile_gateset.clifford_rz_gateset(atol=1e-8)

This tolerance is used when decomposing, simplifying, and dropping negligible
operations. It is distinct from the approximation tolerance used during
Clifford+T synthesis.

Compile to Clifford+T
---------------------

The second stage replaces each remaining arbitrary-angle Rz operation with a
sequence of Clifford and T gates:

.. code-block:: python

   clifford_t_circuit = res.compile_gateset.compile_gateset(
       clifford_rz_circuit,
       gateset=res.compile_gateset.clifford_t_gateset(atol=1e-3),
   )

The current Clifford+T target contains:

* H
* S
* X
* Z
* CNOT
* T

Identity, measurement, and reset operations are also permitted.

Rz synthesis
~~~~~~~~~~~~

The synthesis implementation first checks whether an Rz angle is close to a
rotation with an exact representation. These cases are represented directly
using operations such as I, Z, S, and T.

Other angles are passed to ``pygridsynth``, which produces an approximate
single-qubit sequence over Clifford and T operations.

The generated sequence may contain:

* H
* S
* T
* X
* Z

Global-phase operations returned by the synthesizer do not need to be added to
the circuit.

Approximation tolerance
~~~~~~~~~~~~~~~~~~~~~~~

The ``atol`` passed to ``clifford_t_gateset`` is the maximum approximation
error for each synthesized Rz rotation:

.. code-block:: python

   target = res.compile_gateset.clifford_t_gateset(atol=1e-3)

This is a per-rotation tolerance, not a bound on the error of the complete
circuit.

A smaller tolerance generally requires longer synthesis sequences and more T
gates. Because T gates consume magic states, this can increase:

* The number of magic-state preparations.
* Factory demand.
* Circuit execution time.
* Physical operation counts.

A larger tolerance generally produces shorter sequences but less accurate
rotation approximations.

When selecting a tolerance for a complete program, the number of synthesized
rotations should also be considered. The tutorial demonstrates one way to
derive a per-rotation tolerance from a target program fidelity.

Complete example
----------------

The two compilation stages can be used together as follows:

.. code-block:: python

   import cirq
   import resource_estimation as res

   q0, q1 = cirq.LineQubit.range(2)

   circuit = cirq.Circuit(
       cirq.rx(0.2).on(q0),
       cirq.CNOT(q0, q1),
       cirq.rz(0.3).on(q1),
   )

   clifford_rz_circuit = res.compile_gateset.compile_gateset(
       circuit,
       gateset=res.compile_gateset.clifford_rz_gateset(),
   )

   clifford_t_circuit = res.compile_gateset.compile_gateset(
       clifford_rz_circuit,
       gateset=res.compile_gateset.clifford_t_gateset(atol=1e-3),
       verbose=False,
   )

The resulting ``clifford_t_circuit`` is suitable for the standard
architecture-and-layout workflow.

Toffoli decomposition
---------------------

The repository contains a ``toffoli_decompose`` helper that replaces
``cirq.TOFFOLI`` operations with a fixed Clifford+T decomposition:

.. code-block:: python

   qubits = cirq.LineQubit.range(3)
   circuit = cirq.Circuit(cirq.TOFFOLI.on(*qubits))

   decomposed_circuit = res.compile_gateset.toffoli_decompose(circuit)

The decomposition uses H, CNOT, S, T, and Z operations. Inverse T operations
are represented using combinations of Z, S, and T.

This helper is implemented and tested, but it has important limitations:

* It is not automatically called by ``compile_gateset``.
* It does not optimize the resulting Clifford+T circuit.
* It only replaces operations recognized as ``cirq.TOFFOLI``.
* ``CliffRzGateset`` does not currently have a specialized Toffoli-preserving
  decomposition.

Therefore, ``toffoli_decompose`` should currently be treated as a separate,
explicit transformation rather than part of the default compilation pipeline.

Toffoli and CCZ
---------------

Toffoli and CCZ are related by Hadamards on the target qubit:

.. code-block:: text

   Toffoli = H(target) * CCZ * H(target)

This relationship allows a compiler to choose between decomposing a Toffoli
into Clifford+T gates or preserving an equivalent CCZ operation for a
CCZ-specific resource model.

The downstream ``ft_compile`` function currently validates circuits over:

* Clifford operations
* T
* CCZ
* Measurement
* Reset

It does not accept Toffoli directly.

CCZ resource teleportation and CCZ-distillation logic are present in the
fault-tolerant compiler. However, direct end-to-end CCZ support is not yet
available through the standard gate-set and public-layout workflow:

* ``clifford_t_gateset`` does not preserve CCZ.
* The standard ``MovementLayout`` does not currently generate CCZ-factory
  nodes.
* The CCZ distillation path requires a compatible distillation layout.
* The internal movement-distillation layout is not currently exported as part
  of the public ``resource_estimation.ftqc`` interface.

.. note::

   A public compilation target that preserves Toffoli or CCZ operations is not
   currently implemented. Documentation should distinguish this planned
   direction from the existing Clifford+T pipeline.

Choosing whether to preserve Toffoli or CCZ
-------------------------------------------

The choice between decomposition and preservation can have a significant
effect on a resource estimate.

Decomposing Toffoli into Clifford+T:

* Uses the existing T-state resource path.
* Increases the number of T gates.
* May increase T-factory demand and execution time.

Preserving an equivalent CCZ operation:

* Requires a CCZ-aware compiler and layout.
* Requires a compatible CCZ-state preparation or distillation model.
* May provide a different space-time tradeoff than a T-only decomposition.

Until the public CCZ workflow is complete, the default supported approach is
to compile or explicitly decompose the circuit into Clifford+T.

Effect on resource estimates
----------------------------

Gate-set compilation is part of the resource-estimation model, not merely an
input-format conversion.

The final estimate can depend on:

* The decomposition chosen for high-level gates.
* Whether multi-qubit non-Clifford gates are preserved or decomposed.
* The Clifford+Rz simplification tolerance.
* The Rz-synthesis tolerance.
* The number of arbitrary rotations in the circuit.
* The T count and T depth of the synthesized circuit.
* The magic-state resources available in the selected layout.

When comparing architecture or layout models, the same gate-set compilation
policy should be used for each estimate. Otherwise, differences in the
compiled circuits may be mistaken for differences caused by the resource
models.

Current limitations
-------------------

The current implementation has the following limitations:

* The standard public pipeline targets Clifford+T rather than
  Clifford+T+CCZ.
* Toffoli decomposition is available only as a separate helper.
* The Toffoli helper applies a fixed decomposition and does not perform
  follow-up optimization.
* The Clifford+Rz target has no specialized Toffoli decomposition.
* The Clifford+T approximation tolerance applies separately to each Rz
  operation.
* The implementation does not automatically select a synthesis tolerance from
  a complete-program error budget.
* Different decompositions of the same input circuit may produce different
  resource estimates.