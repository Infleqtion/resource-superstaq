Magic-State Resources
=====================

The current resource-estimation workflow supports magic-state resources
for:

* :math:`T` gates,
* :math:`S` gates when they are not implemented natively by the selected
  architecture, and
* :math:`CCZ` gates.

These resource states are prepared using either cultivation or
distillation and are then consumed through gate teleportation.

For information about how circuits are compiled into the supported logical
gate set, see :doc:`gate_set_compilation`.


Resource-state selection
------------------------

The compiler determines which type of resource state is needed from the
logical operation and the selected layout.

.. list-table::
   :header-rows: 1
   :widths: 15 30 25 30

   * - Logical operation
     - When a resource is required
     - Resource
     - Current preparation method
   * - :math:`T`
     - Always
     - :math:`T` state
     - Cultivation or 15-to-1 distillation
   * - :math:`S`
     - When the architecture does not support the operation directly
     - :math:`Y` state
     - Cultivation
   * - :math:`CCZ`
     - Always
     - :math:`CCZ` state
     - 8-to-1 distillation

A :math:`T` resource state is represented by

.. math::

   \lvert T \rangle =
   \frac{\lvert 0 \rangle + e^{i\pi/4}\lvert 1 \rangle}{\sqrt{2}}.

The cultivated resource used for an :math:`S` operation is represented by

.. math::

   \lvert Y \rangle =
   \frac{\lvert 0 \rangle + e^{i\pi/2}\lvert 1 \rangle}{\sqrt{2}}.


Cultivation
-----------

Cultivation prepares a single magic state on a logical code patch. In the
lattice-surgery representation, cultivation is described by a
``Cultivate(theta)`` operation:

* ``Cultivate(pi / 4)`` prepares a :math:`T` state.
* ``Cultivate(pi / 2)`` prepares a :math:`Y` state.

The physical cost of cultivation is determined by the selected architecture.
Relevant architecture parameters include:

``cultivation_fault_distance``
   Controls the fault distance used by the cultivation cost model.

``cultivation_repetition``
   Multiplies the estimated cultivation cost to account for repeated
   preparation attempts.

``fold_cultiv``
   Selects the folded form of the :math:`T`-state cultivation protocol.

The movement cost associated with cultivation depends on the architecture.
For example, some architectures require patches to be moved to measurement
zones, while others allow the required operations to be performed in place.

The current estimator treats ``cultivation_repetition`` as a direct cost
multiplier. It does not randomly simulate successful and failed cultivation
attempts.


Distillation
------------

Distillation consumes several lower-quality input magic states to produce a
higher-quality output resource. Resource Superstaq currently contains two
distillation circuits.

15-to-1 T-state distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 15-to-1 protocol consumes fifteen cultivated :math:`T` states and
produces one output :math:`T` state.

The implemented logical layout uses:

* 15 cultivated :math:`T`-state inputs,
* 15 data-line patches,
* one output patch, and
* 31 logical patches in total.

The distillation circuit is constructed by
``resource_estimation.ftqc.distil.distil_15_to_1``.

8-to-1 CCZ-state distillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 8-to-1 protocol consumes eight cultivated :math:`T` states and produces
one three-qubit :math:`CCZ` resource state.

The implemented logical layout uses:

* eight cultivated :math:`T`-state inputs,
* fifteen main circuit patches,
* three output patches contained within the main circuit, and
* 23 logical patches in total.

The distillation circuit is constructed by
``resource_estimation.ftqc.distil.ccz_8_to_1``.

For movement-based architectures, the estimator expands the selected
distillation circuit, inserts the required movement operations, and
calculates its serial or parallel resource cost. The result is then
multiplied by the architecture's ``distillation_repetition`` parameter.

As with cultivation, the repetition parameter is currently a deterministic
cost multiplier rather than a stochastic simulation of distillation
success.


Factories
---------

Magic states are supplied to the logical circuit through factory locations
defined by the selected layout.

The layout separately tracks factories for:

* :math:`T` states,
* :math:`Y` states used for :math:`S` operations, and
* :math:`CCZ` states.

When a resource state is requested, the compiler performs the following
steps:

#. Check whether a prepared factory output of the required type is
   available.
#. If none are available, prepare and reload the applicable factories.
#. Select the factory nearest to the operation's logical qubits.
#. Route or teleport the factory output to the logical operation.
#. Mark that factory output as consumed.

For layouts with explicit connectivity, factory selection uses routing
distance. Otherwise, the implementation uses a movement-distance heuristic.


Consuming a magic state
-----------------------

After a resource has been prepared, the compiler consumes it through a
teleportation circuit. At a high level, this consists of:

#. coupling the factory output to the program qubit,
#. measuring the factory output,
#. resetting the factory patch, and
#. applying the required correction operation.

The correction depends on the resource being consumed:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Resource
     - Target operation
     - Correction modeled by the compiler
   * - :math:`T` state
     - :math:`T`
     - :math:`S`
   * - :math:`Y` state
     - :math:`S`
     - :math:`Z`
   * - :math:`CCZ` state
     - :math:`CCZ`
     - A sequence of Hadamard, Pauli-X, and CNOT operations across the
       three operation qubits

Whether the correction is physically required depends on the measurement
outcome. The current logical resource model includes the correction
operations directly so that their cost is represented.


Current limitations
-------------------

Magic-state support is still developing. The following limitations are
important when interpreting current estimates:

* The estimator calculates logical resource costs but does not propagate
  state fidelities or simulate logical error probabilities.
* Cultivation and distillation repetition counts are modeled as fixed cost
  multipliers.
* Measurement-conditioned corrections inside the implemented distillation
  circuits are currently represented unconditionally.
* Cultivation costs are currently implemented only for
  :math:`\theta=\pi/4` and :math:`\theta=\pi/2` resource states.
* There is no cultivated :math:`CCZ` preparation path; :math:`CCZ`
  resources use distillation.
* The layout used for distillation is currently an internal implementation
  rather than part of the main exported layout API.
* End-to-end :math:`CCZ` compilation and factory placement are not yet
  available through every standard compilation and layout path.

These restrictions should be considered when comparing estimates produced
using different gate sets, layouts, or architectures.