"""Intersectional fairness analysis for group combinations."""

from __future__ import annotations

import pandas as pd


def intersectional_approval_rates(df: pd.DataFrame, group_cols: list[str], decision_col: str) -> pd.DataFrame:
    """Compute approval rates for intersecting categories.

    Args:
        df: DataFrame containing the relevant grouping columns and treatment outcome.
        group_cols: One or more group columns such as gender and city tier.
        decision_col: Binary approval indicator.

    Returns:
        A DataFrame with intersection counts and approval rates.
    """
    if not group_cols:
        raise ValueError("At least one group column is required.")
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for intersectional analysis: {missing}")
    if decision_col not in df.columns:
        raise KeyError(f"Decision column '{decision_col}' not found in DataFrame.")

    grouped = df.groupby(group_cols, dropna=False)[decision_col].agg(["count", "mean"])
    grouped = grouped.rename(columns={"mean": "approval_rate"})
    grouped["approval_count"] = df.groupby(group_cols, dropna=False)[decision_col].sum()
    grouped["rejection_count"] = grouped["count"] - grouped["approval_count"]
    return grouped.reset_index()
