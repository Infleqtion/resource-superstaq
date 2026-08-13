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
from __future__ import annotations


class C:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"


def boxed_header(title: str, width: int = 40) -> str:
    pad = width - len(title) - 2
    left = pad // 2
    right = pad - left
    return f"{'=' * left} {title} {'=' * right}"


def hr(width: int = 40) -> str:  # pragma: no cover
    return "=" * width


def make_pretty(obj: object) -> str:  # pragma: no cover
    """
    Pulling out the pretty functionality from the ResourceEstimator class to avoid doubling resource calls
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)
