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
"""Shared interface for physical circuits supplied by external PL packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import cirq

from resource_estimation.ftqc.codepatch import CSSCodePatch


class PhysicalImplementation(Protocol):
    """Realize a primitive using the existing code-patch encoding and wire order.

    Inputs are already encoded. Output register bindings remain unchanged.
    """

    def build_circuit(
        self,
        primitive: cirq.Gate,
        encoding: CSSCodePatch,
        registers: Mapping[str, Sequence[cirq.Qid]],
    ) -> cirq.Circuit:
        """Return a physical circuit or reject an unsupported request."""
        ...
