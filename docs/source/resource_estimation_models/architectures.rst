Architecture Models
===================

Architecture models describe the hardware capabilities and physical-cost
assumptions used by resource-superstaq. They define the logical primitives
available to the compiler and assign physical gate counts and execution times
to those primitives.

The models also provide assumptions used when estimating error-corrected
operations, including code distance, syndrome-extraction rounds, logical-patch
size, movement costs, and magic-state preparation costs.

What an architecture controls
-----------------------------

Every architecture model provides:

* A rotated-code logical patch with code distance ``d``.
* A set of logical primitives accepted by the architecture.
* Physical gate counts for each primitive.
* Parallel-operation counts used to calculate critical-path time.
* Physical gate-duration assumptions.
* Syndrome-extraction costs.
* Magic-state cultivation costs.
* Policies for idling and post-operation correction.
* Movement assumptions, where applicable.

NOTE: The architecture must be used consistently for both compilation and resource
estimation

Common configuration
--------------------

``d``
   The distance of the rotated-code patch. Increasing ``d`` increases the
   physical-qubit count and the cost of logical operations.

``idling``
   Whether the fault-tolerant compiler adds correction operations for idle
   logical qubits.

``post_op_correction``
   Whether syndrome extraction is added after applicable logical operations.

``syndrome_rounds``
   The number of syndrome-extraction rounds used by the cost model. When this
   value is ``None``, the implementation uses ``d`` rounds.

``cultivation_repetition``
   Multiplier applied to T-state cultivation costs.

``cultivation_fault_distance``
   Fault distance passed to the cultivation cost model.

``fold_cultiv``
   Selects the folded cultivation model for movement-based architectures.

``distillation_repetition``
   Multiplier applied to distillation costs. This option is available on the
   movement-based architecture classes.

Available models
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 14 20 20

   * - Class
     - Modeled system
     - Movement
     - Logical two-qubit style
     - Physical timing model
   * - ``DefaultLattice``
     - Dual species without movement
     - No
     - Lattice surgery
     - Neutral-atom assumptions
   * - ``DefaultMovement``
     - Single-species zoned architecture
     - Yes
     - Transversal CNOT
     - Neutral-atom assumptions
   * - ``DualSpeciesMovement``
     - Dual species with movement
     - Yes
     - Transversal CNOT
     - Neutral-atom assumptions
   * - ``MeasureZonesOnly``
     - Movement with measurement zones
     - Yes
     - Transversal CNOT
     - Neutral-atom assumptions
   * - ``Superconductor``
     - Superconducting proxy
     - No
     - Lattice surgery
     - Superconducting assumptions

DefaultLattice
--------------

``DefaultLattice`` models a dual-species system without logical-qubit
movement. Logical CNOT operations are decomposed into lattice-surgery
``Merge`` and ``Split`` primitives.

The model:

* Uses lattice-surgery merge and split operations.
* Does not include ``Move`` as a supported primitive.
* Includes cultivation, syndrome extraction, measurement, reset, and logical
  single-qubit primitives.
* Uses the neutral-atom physical gate-duration table, excluding movement.
* Defaults to idling correction and post-operation correction.
* Uses ``d`` syndrome-extraction rounds when ``syndrome_rounds`` is not set.
* Applies no movement penalty to T-state cultivation.

The implementation identifies this model internally as
``DualSpeciesNoMovement``.

.. code-block:: python

   architecture = res.ftqc.DefaultLattice(
       d=11,
       idling=True,
       post_op_correction=True,
   )

.. autoclass:: resource_estimation.ftqc.DefaultLattice

DefaultMovement
---------------

``DefaultMovement`` models a zoned, single-species neutral-atom system with
movement available for transversal operations.

The model:

* Supports logical CNOT, S, H, cultivation, distillation, movement, syndrome
  extraction, measurement, and reset primitives.
* Models logical CNOT as a transversal physical CZ layer with single-qubit
  basis changes.
* Models H as a transversal single-qubit operation followed by a qubit
  permutation.
* Models S using a fold-transversal construction.
* Adds movement around CNOT and measurement operations assigned to interaction
  or measurement zones.
* Adds movement penalties to syndrome extraction and cultivation.
* Supports T and CCZ distillation cost models.
* Defaults to one syndrome-extraction round.
* Defaults to no idling correction and enables post-operation correction.

Logical movement between two layout locations is assigned a
Manhattan-distance-based cost. The current implementation caps the modeled
movement time at the architecture's maximum movement duration.

The implementation identifies this model internally as
``SingleSpeciesMovement``.

.. code-block:: python

   architecture = res.ftqc.DefaultMovement(
       d=11,
       fold_cultiv=False,
       cultivation_repetition=1,
       distillation_repetition=1,
   )

.. autoclass:: resource_estimation.ftqc.DefaultMovement

DualSpeciesMovement
-------------------

``DualSpeciesMovement`` inherits the primitive set and most operation costs
from ``DefaultMovement``. It changes the assumptions for readout,
syndrome extraction, cultivation, and where movement is inserted.

The model:

* Allows atoms of different species to move along alleyways.
* Uses the same logical S and CNOT cost models as ``DefaultMovement``.
* Adds alley movement around logical CNOT operations.
* Does not add zone movement around measurement.
* Uses syndrome-extraction costs without movement penalties.
* Uses Y-state cultivation without a movement penalty.
* Adds a CZ-related movement penalty to T-state cultivation only when folded
  cultivation is selected.
* Uses the neutral-atom physical gate-duration table.

.. code-block:: python

   architecture = res.ftqc.DualSpeciesMovement(
       d=11,
       fold_cultiv=False,
   )

.. autoclass:: resource_estimation.ftqc.DualSpeciesMovement

MeasureZonesOnly
----------------

``MeasureZonesOnly`` inherits from ``DefaultMovement`` but models a system in
which measurement requires transport to a measurement zone while entangling
operations can occur outside a dedicated interaction zone.

The model:

* Adds movement before and after measurement-zone operations.
* Adds alley movement around logical CNOT operations.
* Uses the same logical S and CNOT costs as ``DefaultMovement``.
* Adds measurement-related movement penalties to syndrome extraction.
* Adds measurement-related movement penalties to T- and Y-state cultivation.
* Adds CZ-related cultivation movement when folded cultivation is selected.
* Uses the neutral-atom physical gate-duration table.
* Defaults to the movement architecture's single syndrome-extraction round.

The current implementation uses the fold-transversal S cost inherited from
``DefaultMovement``. The source identifies alternative S-gate implementations
as future work.

The implementation identifies this model internally as
``ReadoutZonesOnly``.

.. code-block:: python

   architecture = res.ftqc.MeasureZonesOnly(
       d=11,
       fold_cultiv=False,
   )

.. autoclass:: resource_estimation.ftqc.MeasureZonesOnly

Superconductor
--------------

``Superconductor`` is a proxy model for superconducting hardware. It inherits
the lattice-surgery primitives and operation-cost formulas from
``DefaultLattice`` but replaces the physical gate-duration assumptions.

The model:

* Uses lattice-surgery merge and split operations.
* Does not support logical movement.
* Uses physical CZ and single-qubit gates as its primary gate assumptions.
* Uses faster physical gate, measurement, and reset times than the
  neutral-atom models.
* Defaults to idling correction and post-operation correction.
* Uses ``d`` syndrome-extraction rounds when ``syndrome_rounds`` is not set.

This class is an approximate resource model, not a model of a particular
superconducting processor.

.. code-block:: python

   architecture = res.ftqc.Superconductor(
       d=11,
       idling=True,
       post_op_correction=True,
   )

.. autoclass:: resource_estimation.ftqc.Superconductor

Physical timing assumptions
---------------------------

Physical operation times are represented in microseconds.

The current neutral-atom defaults are:

.. list-table::
   :header-rows: 1

   * - Physical operation
     - Duration
   * - CZ
     - 0.27 microseconds
   * - Single-qubit operation
     - 5 microseconds
   * - Reset
     - 400 microseconds
   * - Measurement
     - 1000 microseconds
   * - Maximum movement operation
     - 500 microseconds

The current superconducting defaults are:

.. list-table::
   :header-rows: 1

   * - Physical operation
     - Duration
   * - CZ
     - 0.040 microseconds
   * - Single-qubit operation
     - 0.020 microseconds
   * - Reset
     - 1 microsecond
   * - Measurement
     - 0.5 microseconds

These values are model inputs and may change as the implementation is refined.
They should not be interpreted as guarantees for a particular device.

Implementation limitations
--------------------------

The architecture models contain approximations and provisional assumptions:

* They estimate operation cost and duration but do not directly calculate a
  logical failure probability for the complete circuit.
* Several cultivation costs are based on precomputed or approximate circuit
  models.
* Correlated decoding is represented through architecture assumptions such as
  reduced syndrome-round counts rather than a decoding simulation.
* Movement costs use simplified zone and Manhattan-distance models.
* Some logical-operation implementations are marked as future work in the
  source.
* ``Superconductor`` currently accepts ``cultivation_fault_distance`` in its
  constructor but does not forward that value to ``DefaultLattice``; the
  inherited default is therefore used.