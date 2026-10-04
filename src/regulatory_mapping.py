"""Regulatory mapping scaffold for the fairness audit."""

from __future__ import annotations


REGULATORY_MAP = {
    "india": {
        "RBI Fair Practices": "Ensure fair and transparent credit decisions and grievance redressal.",
        "DPDP Act 2023": "Use personal data lawfully, fairly, and with documented purpose limitation.",
    },
    "us": {
        "ECOA": "Credit decisions should not discriminate on the basis of protected characteristics.",
        "Regulation B": "Implement rules for credit-evaluation criteria and adverse-action notices.",
    },
    "eu": {
        "EU AI Act": "High-risk AI system obligations for creditworthiness assessment.",
    },
}


def get_regulatory_summary() -> dict[str, dict[str, str]]:
    """Return a structured overview of relevant regulatory areas."""
    return REGULATORY_MAP
