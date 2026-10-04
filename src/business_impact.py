"""Business-impact placeholder calculations for the audit."""

from __future__ import annotations

from typing import Any


def estimate_business_impact(
    revenue_per_approval: float,
    rejected_applicants: int,
    approval_rate_gap: float,
    conservative_multiplier: float = 0.5,
    base_multiplier: float = 1.0,
    aggressive_multiplier: float = 1.5,
) -> dict[str, float]:
    """Estimate a rough business impact from potentially missed approvals.

    This is a conservative placeholder and should be replaced with actual business assumptions from
    the project brief or finance stakeholders when available.
    """
    potential_revenue = revenue_per_approval * rejected_applicants
    return {
        "conservative": potential_revenue * conservative_multiplier * (1 - approval_rate_gap),
        "base": potential_revenue * base_multiplier * (1 - approval_rate_gap),
        "aggressive": potential_revenue * aggressive_multiplier * (1 - approval_rate_gap),
    }
