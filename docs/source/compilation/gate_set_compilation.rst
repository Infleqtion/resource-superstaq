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


Compile to Clifford+Rz
----------------------

The first stage converts a general Cirq circuit into an intermediate
Clifford+Rz representation using ``CliffRzGateset``, a Cirq two-qubit compilation target.

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
built from the following operations:

* H
* S
* X
* Z
* CNOT
* Rz

Measurements and supported non-unitary operations can also be retained.

NOTE: The Clifford+Rz target accepts an absolute tolerance:

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

NOTE: Work is being done to consider a clifford+T+CCZ pass which keeps CCZ gates in
the circuit and turns Toffolis into CCZs by wrapping the target in hadimard gates.

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