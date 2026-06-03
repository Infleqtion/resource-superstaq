# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cirq
import pytest
import resource_estimation.architecture as arch
import resource_estimation.estimate as est
import resource_estimation.lattice_surgery_primitives as lsp
from numpy import isclose


# x = msd.msd_15_to_1()
# ssm = res.architecture.DefaultMovement(
#     d=11,  # Rotated Surface Code code distance
#     idling=False,  # Include Syndrome Extraction on idling qubits in compiled circuit
#     post_op_correction=True,  # Turn on or off Syndrome Extraction after transversal operations
#     syndrome_rounds=1,  # Rounds of Syndrome Extraction after transversal operations
#     cultivation_repetition=5,  # Expected repetitions of the cultivation circuit to get a successful T state
# )
ssm = res.architecture.DefaultLattice()
# x = distil(ssm, state_type='H')
# x = _distil_cost(ssm, 'H')
print(x)
