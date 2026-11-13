"""The marker carried by everything derived from the evaluator's synthetic generator.

No real outcome data exists yet, so every model, report, figure and model card made from
``steproute.eval.synthetic`` says so, in the artifact itself and in several places each. The
decision is driven by the data (the grid header names the generator), not by a flag a caller
could forget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["STAMP", "SYNTHETIC_GENERATOR", "is_synthetic"]

STAMP = "SYNTHETIC DATA. NOT A RESULT."
SYNTHETIC_GENERATOR = "steproute.eval.synthetic"


def is_synthetic(header: Mapping[str, Any]) -> bool:
    """Whether a grid header says the grid came from the synthetic generator."""
    return str(header.get("generator", "")).startswith(SYNTHETIC_GENERATOR)
