Resource-estimation workflow
============================

resource-superstaq estimates the resources needed to execute a quantum circuit
on a modeled fault-tolerant architecture. The estimate is produced through
several compilation and modeling stages.

The overall workflow is:

#. Prepare an input Cirq circuit.
#. Compile the circuit into the Clifford+Rz gate set.
#. Synthesize the remaining rotations into Clifford+T operations.
#. Select a fault-tolerant architecture and logical layout.
#. Compile the circuit into fault-tolerant primitives.
#. Calculate resource estimates from the primitive circuit.

Each stage affects the resulting estimate.

Prepare an input circuit
------------------------

The workflow begins with a :class:`cirq.Circuit`. The circuit may contain
high-level gates that are not directly represented by the fault-tolerant
architecture.

For example:

.. code-block:: python

   import cirq
   import resource_estimation as res

   q0, q1 = cirq.LineQubit.range(2)

   circuit = cirq.Circuit(
       cirq.H(q0),
       cirq.CNOT(q0, q1),
       cirq.rz(0.2).on(q1),
   )

Compile to Clifford+Rz
----------------------

The first compilation stage converts the input circuit into Clifford
operations and arbitrary-angle Z rotations.

.. code-block:: python

   clifford_rz_circuit = res.compile_gateset.compile_gateset(
       circuit,
       gateset=res.compile_gateset.clifford_rz_gateset(),
   )

Using an intermediate Clifford+Rz representation separates general
gate decomposition from rotation synthesis.

Compile to Clifford+T
---------------------

Arbitrary-angle Z rotations are then approximated using Clifford and T gates:

.. code-block:: python

   clifford_t_circuit = res.compile_gateset.compile_gateset(
       clifford_rz_circuit,
       gateset=res.compile_gateset.clifford_t_gateset(atol=0.001),
   )

The ``atol`` parameter controls the maximum approximation error for each
synthesized Z rotation. Smaller tolerances generally produce more accurate
approximations but may require more T gates.

Because T gates require expensive fault-tolerant resources, the selected
tolerance can significantly affect the final estimate.

Select an architecture
----------------------

An architecture describes the physical assumptions used by the estimator,
including:

* The error-correcting code and code distance.
* The fault-tolerant primitives supported by the architecture.
* The physical cost and duration of those primitives.
* Whether movement, idling correction, or post-operation syndrome extraction
  is required.

For example:

.. code-block:: python

   architecture = res.ftqc.DefaultMovement(d=11)

Here, ``d=11`` selects a code distance of 11. ``DefaultMovement`` is one
available architecture model and is not necessarily appropriate for every
application.

Select a layout
---------------

A layout describes how logical data qubits and supporting resources are
arranged. It also provides information needed for routing and magic-state
consumption.

.. code-block:: python

   layout = res.ftqc.MovementLayout(
       input_circuit=clifford_t_circuit,
       num_t_factories=5,
   )

The number of factories affects the space-time tradeoff. Additional factories
may allow more T states to be prepared concurrently, but they also require
additional physical qubits.

Layouts and architectures must be chosen together. For example, movement-based
architectures generally use a movement-compatible layout.

Compile to fault-tolerant primitives
------------------------------------

The fault-tolerant compiler lowers the Clifford+T circuit into the primitive
operations supported by the selected architecture:

.. code-block:: python

   primitive_circuit = res.ftqc.ft_compile(
       layout=layout,
       arc=architecture,
       verbose=True,
   )

The resulting circuit may contain operations representing syndrome
extraction, logical movement, magic-state use, lattice-surgery operations, or
other architecture-specific primitives.

Estimate resources
------------------

Create a resource estimator using the same architecture:

.. code-block:: python

   estimator = res.ftqc.ResourceEstimator(arc=architecture)

The primitive circuit can then be used to estimate several quantities:

.. code-block:: python

   parallel_cost = estimator.parallel_circuit_cost(
       primitive_circuit,
       pretty=True,
   )
   circuit_time = estimator.parallel_circuit_time(primitive_circuit)
   physical_qubits = estimator.physical_qubits(primitive_circuit)

``parallel_circuit_cost``
   Estimates the physical operations along the circuit's critical path.

``parallel_circuit_time``
   Estimates execution time while accounting for operations that can occur in
   parallel.

``physical_qubits``
   Estimates the physical qubits required by the logical patches represented
   in the compiled circuit.

For total operation counts rather than critical-path costs, use
``serial_circuit_cost``.

Interpreting an estimate
------------------------

A resource estimate is conditional on the selected model. It should not be
interpreted as a hardware-independent prediction.

Important inputs include:

* The input circuit and its initial decomposition.
* The rotation-synthesis tolerance.
* The selected architecture.
* The error-correcting-code distance.
* The logical layout.
* The number and type of magic-state factories.
* Idling and syndrome-extraction assumptions.

When comparing two estimates, change one modeling assumption at a time where
possible. This makes it easier to identify which choice caused the difference.