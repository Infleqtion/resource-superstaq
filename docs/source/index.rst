.. resource-superstaq documentation master file, created by
   sphinx-quickstart on Fri Jul 17 11:34:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

resource-superstaq
================================


Welcome to the resource-superstaq documentation.

resource-superstaq is a Python toolkit for estimating the resources required
to execute quantum algorithms on fault-tolerant quantum computers. It provides
an end-to-end workflow for compiling quantum circuits, mapping them onto a
fault-tolerant architecture, and estimating quantities such as execution time,
physical-qubit requirements, and logical-operation costs.

What can resource-superstaq do?
-------------------------------

* Compile Cirq circuits into resource-relevant gate sets, including
  Clifford+Rz and Clifford+T.
* Approximate arbitrary rotations using Clifford and T gates.
* Represent fault-tolerant architectures and logical-qubit layouts.
* Compile logical circuits into fault-tolerant primitives.
* Estimate circuit execution time and physical-qubit requirements.
* Compare resource tradeoffs across architecture and layout choices.

Typical workflow
----------------

A resource-estimation workflow generally consists of four stages:

#. **Gate-set compilation:** Compile the input circuit into Clifford+T or
   another supported gate set.
#. **Architecture and layout selection:** Choose the physical architecture,
   code distance, logical-qubit layout, and magic-state resources.
#. **Fault-tolerant compilation:** Lower the logical circuit into operations
   supported by the selected architecture.
#. **Resource estimation:** Calculate execution time, operation costs, and
   physical-qubit requirements.

Getting started
---------------

New users should begin with the installation guide and quickstart tutorial.
The quickstart will demonstrate the complete workflow from an input circuit
to a fault-tolerant resource estimate.

For additional context, see the paper
`Resource Estimation via Efficient Compilation of Key Quantum Primitives
<https://doi.org/10.48550/arXiv.2604.01376>`_.



NOTE: This documentation is built with ``reStructuredText`` syntax. If you would like to build upon this documentation,see the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Getting Started:

   getting_started/installation
   getting_started/workflow
   getting_started/tutorial

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Resource Estimation Models

   resource_estimation_models/architectures
   resource_estimation_models/layouts

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Compilation

   compilation/gate_set_compilation
   compilation/magic_state_resources


