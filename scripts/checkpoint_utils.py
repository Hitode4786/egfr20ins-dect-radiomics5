"""
Compatibility helpers for loading locally generated PyTorch checkpoints.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


try:
    _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters
except (TypeError, ValueError):
    _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = False


def load_torch_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load project checkpoints without triggering PyTorch's future warning."""
    load_kwargs = {"map_location": map_location}
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        # These checkpoints intentionally store optimizer/config/history metadata.
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)
