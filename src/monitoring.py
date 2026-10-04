"""Monitoring framework definitions for ongoing fairness checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonitoringThreshold:
    """Threshold values for audit monitoring."""
    green: float
    amber: float
    red: float


def build_monitoring_thresholds() -> dict[str, MonitoringThreshold]:
    """Define default alert thresholds for fairness and performance metrics."""
    return {
        "approval_rate_gap": MonitoringThreshold(0.00, 0.03, 0.05),
        "default_rate_gap": MonitoringThreshold(0.00, 0.02, 0.04),
        "auc_gap": MonitoringThreshold(0.00, 0.02, 0.04),
        "demographic_parity_ratio": MonitoringThreshold(0.95, 0.90, 0.80),
    }
