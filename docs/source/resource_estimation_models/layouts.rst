Logical Layout Models
=====================

Layout models describe the placement and connectivity of logical code patches.
They map input-circuit qubits onto ``cirq.GridQubit`` locations and track the
logical resources used during compilation.

A layout may contain:

* Data patches representing input-circuit qubits.
* Ancilla patches used for lattice-surgery routing.
* T-state factories.
* S-state factories.
* CCZ-state factories or supporting distillation blocks.
* Graph edges describing allowed routing paths.

Layouts operate at the logical-patch level. They do not describe the
individual physical qubits inside each code patch.

Common layout behavior
----------------------

Every layout produces:

``mapped_circuit``
   A copy of the input circuit whose qubits have been mapped to
   ``cirq.GridQubit`` locations.

``layout_graph``
   A NetworkX graph containing data, ancilla, factory, and supporting block
   nodes.

Layouts also track which factories are available. During compilation, a
factory can be selected using a routing-distance or Manhattan-distance
heuristic and then marked as used. Factories can be reloaded when another
round of magic states becomes available.

For lattice-style layouts, CNOT routing uses the shortest path through the
layout graph. The route must pass through at least one ancilla patch and cannot
use unrelated data or factory patches as intermediate routing space.

The current routing method processes operations independently. It does not
optimize multiple simultaneous CNOT routes for global parallelism.

Available layouts
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 21 17 20 20

   * - Class
     - Intended model
     - Connectivity
     - Factory configuration
     - CNOT routing
   * - ``MovementLayout``
     - Movement architectures
     - All-to-all model
     - Configurable T factories
     - Handled by movement primitives
   * - ``Column``
     - Lattice-surgery architectures
     - Nearest-neighbor graph
     - Derived from circuit size
     - Ancilla paths
   * - ``FactorySandwich``
     - Lattice-surgery architectures
     - Nearest-neighbor graph
     - Configurable S and T factories
     - Ancilla rows
   * - ``Embedded``
     - Lattice-surgery architectures
     - Nearest-neighbor graph
     - Derived from layout boundary
     - Interleaved ancilla paths

Typical pairings
----------------

Movement-based architecture models generally use ``MovementLayout``:

* ``DefaultMovement``
* ``DualSpeciesMovement``
* ``MeasureZonesOnly``

Lattice-surgery architecture models generally use one of the lattice layouts:

* ``DefaultLattice``
* ``Superconductor``

with:

* ``Column``
* ``FactorySandwich``
* ``Embedded``

These are intended pairings based on the primitives and routing behavior in
the current implementation. The code does not provide a comprehensive
architecture-layout compatibility validator.

MovementLayout
--------------

``MovementLayout`` is intended for architectures that support logical
movement and transversal CNOT operations.

The layout:

* Maps data patches and T factories onto a compact logical grid.
* Uses no S factories because movement architectures support logical S
  directly.
* Allows the number of T factories to be selected by the user.
* Uses a fully connected layout graph rather than explicit ancilla routing.
* Selects nearby factories using Manhattan distance.
* Does not implement lattice-surgery CNOT routing.
* Relies on the architecture and compiler to assign movement costs.

The current placement is not optimized to minimize total travel distance.

.. code-block:: python

   layout = res.ftqc.MovementLayout(
       input_circuit=clifford_t_circuit,
       num_t_factories=5,
   )

.. autoclass:: resource_estimation.ftqc.MovementLayout

Column
------

``Column`` is a lattice-surgery layout with two columns of data patches.
Factory patches appear at the left and right boundaries, with ancilla patches
providing routes between data and factories.

Its repeating structure is:

.. code-block:: text

   S | a | q | a | q | a | S
   T | a | a | a | a | a | T
   S | a | q | a | q | a | S
   T | a | a | a | a | a | T

where:

``q``
   Data patch.

``a``
   Ancilla patch.

``S``
   S-state factory.

``T``
   T-state factory.

For ``n`` input-circuit qubits, the implementation creates
``2 * ceil(n / 2)`` factories of each type. If the circuit contains an odd
number of qubits, the unused data position becomes an ancilla.

Graph edges connect patches with Manhattan distance one. CNOTs are routed
through the shortest valid ancilla path.

.. code-block:: python

   layout = res.ftqc.Column(
       input_circuit=clifford_t_circuit,
   )

.. autoclass:: resource_estimation.ftqc.Column

FactorySandwich
---------------

``FactorySandwich`` places a row of data patches between two ancilla rows.
S-state factories are placed above the upper ancilla row and T-state
factories are placed below the lower ancilla row.

.. code-block:: text

   S  S  S  S
   a  a  a  a
   q  q  q  q
   a  a  a  a
   T  T  T  T

The number of S and T factories is configurable. The width of the layout is
the maximum of:

* The number of data patches.
* The number of S factories.
* The number of T factories.

Graph edges connect nearest-neighbor patches. CNOT operations and factory
access are routed through the two ancilla rows.

.. code-block:: python

   layout = res.ftqc.FactorySandwich(
       input_circuit=clifford_t_circuit,
       num_s_factories=2,
       num_t_factories=4,
   )

.. autoclass:: resource_estimation.ftqc.FactorySandwich

Embedded
--------

``Embedded`` packs data patches into an approximately square array and inserts
ancilla rows and columns between them. Alternating S- and T-state factories
surround the resulting array.

The layout:

* Derives its data-array dimensions from the number of input qubits.
* Fills unused positions in the initial square with ancilla space.
* Inserts ancilla rows and columns between data patches.
* Places alternating S and T factories around the boundary.
* Connects nearest-neighbor patches.
* Routes CNOT operations through the inserted ancilla network.
* Computes the number of S and T factories from the generated boundary.

The number of factories cannot currently be configured directly.

.. code-block:: python

   layout = res.ftqc.Embedded(
       input_circuit=clifford_t_circuit,
   )

.. autoclass:: resource_estimation.ftqc.Embedded

Visualizing a layout
--------------------

Every layout provides a ``draw`` method that displays its graph:

.. code-block:: python

   layout.draw()

The current color mapping is:

* Green for data patches.
* Blue for ancilla patches.
* Red for T factories.
* Yellow for S factories.
* Orange for CCZ factories.
* Pink for supporting distillation blocks.