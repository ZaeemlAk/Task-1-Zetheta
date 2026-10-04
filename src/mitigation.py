"""Mitigation strategy scaffolding for fairness remediation."""

from __future__ import annotations

from typing import Any


def mitigation_strategy_matrix() -> dict[str, dict[str, str]]:
    """Return a summary of mitigation categories for the audit report."""
    return {
        "reweighting": {
            "category": "pre-processing",
            "summary": "Adjust sample weights to balance outcomes across groups.",
        },
        "disparate_impact_remover": {
            "category": "pre-processing",
            "summary": "Modify feature distributions to reduce group-level differences.",
        },
        "fairness_regularisation": {
            "category": "in-processing",
            "summary": "Train a model with fairness constraints in the objective function.",
        },
        "equalised_odds_thresholding": {
            "category": "post-processing",
            "summary": "Apply decision thresholds to better align group error rates.",
        },
    }
