"""Fail fast unless the archived EGFR20ins replay runtime is active."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Dict


FROZEN_PYTHON = (3, 10, 14)
FROZEN_PACKAGES: Dict[str, str] = {
    "numpy": "1.26.4",
    "scikit-learn": "1.7.2",
    "xgboost": "3.2.0",
    "shap": "0.48.0",
}


def assert_frozen_runtime() -> None:
    """Prevent a replay under a dependency set known to alter RFE selection."""
    observed_python = sys.version_info[:3]
    discrepancies = []
    if observed_python != FROZEN_PYTHON:
        discrepancies.append(
            f"Python={'.'.join(map(str, observed_python))} (required {'.'.join(map(str, FROZEN_PYTHON))})"
        )
    for distribution, expected in FROZEN_PACKAGES.items():
        try:
            observed = version(distribution)
        except PackageNotFoundError:
            observed = "not installed"
        if observed != expected:
            discrepancies.append(f"{distribution}={observed} (required {expected})")
    if discrepancies:
        raise RuntimeError(
            "Frozen EGFR20ins replay environment check failed; refusing to run because RFE selection is "
            "version-sensitive. Activate the environment defined in environment.yml. "
            "Observed discrepancies: " + "; ".join(discrepancies)
        )
