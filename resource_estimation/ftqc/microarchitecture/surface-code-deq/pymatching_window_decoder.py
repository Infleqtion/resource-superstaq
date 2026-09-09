"""Compatibility file loaded by DEQ's ``black-box-python`` decoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from surface_code_deq.decoders.pymatching import Decoder, HypergraphEligibilityError

__all__ = ["Decoder", "HypergraphEligibilityError"]
