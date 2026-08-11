Installation
============

Requirements
------------
Python 3.14 is recommended because it is the version used by the project's
continuous integration tests.

Clone the repository
--------------------

Clone the repository and enter its root directory:

.. code-block:: console

   git clone git@github.com:Infleqtion/resource-superstaq.git
   cd resource-superstaq

Create an environment
---------------------

Create a Python 3.14 virtual environment.

On macOS or Linux:

.. code-block:: console

   python3.14 -m venv .venv
   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1


Install the repository
----------------------

From the repository root, install the project and its dependencies in editable
mode:

.. code-block:: console

   python -m pip install --upgrade pip
   python -m pip install -e .

An editable installation allows Python to use the code directly from your
local checkout. Changes made to the source are available without reinstalling
the project.

Verify the setup
----------------

Verify that the project can be imported:

.. code-block:: console

   python -c "import resource_estimation; print(resource_estimation.__file__)"

You can also run the test suite:

.. code-block:: console

   python checks/pytest_.py

Build the documentation
-----------------------

Documentation contributors should install the additional documentation
dependencies:

.. code-block:: console

   python -m pip install -r docs/requirements.txt

Then build the documentation from the repository root:

.. code-block:: console

   python checks/build_docs.py