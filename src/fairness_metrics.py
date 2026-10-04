"""Fairness metric calculations used in the audit."""

from __future__ import annotations

from typing import Any

import pandas as pd


def approval_rate(df: pd.DataFrame, group_col: str, decision_col: str) -> pd.DataFrame:
    """Compute approval rate by group.

    Args:
        df: DataFrame containing the decision column.
        group_col: Grouping column, such as gender or city tier.
        decision_col: Binary decision variable where 1 means approved.

    Returns:
        A DataFrame with group counts and rates.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if decision_col not in df.columns:
        raise KeyError(f"Column '{decision_col}' not found in DataFrame.")

    grouped = df.groupby(group_col, dropna=False)[decision_col].agg(["count", "mean"])
    grouped = grouped.rename(columns={"mean": "approval_rate"})
    grouped["approval_count"] = df.groupby(group_col, dropna=False)[decision_col].sum()
    grouped["rejection_count"] = grouped["count"] - grouped["approval_count"]
    return grouped.reset_index()


def demographic_parity_ratio(df: pd.DataFrame, group_col: str, decision_col: str, reference_group: Any) -> float:
    """Compute demographic parity ratio for a chosen reference group."""
    rates = approval_rate(df, group_col, decision_col)
    ref_rate = rates.loc[rates[group_col] == reference_group, "approval_rate"].iloc[0]
    if ref_rate == 0:
        return float("inf")
    return float(rates["approval_rate"].div(ref_rate).mean())


def false_discovery_rate(df: pd.DataFrame, group_col: str, score_col: str, decision_col: str, label_col: str | None = None) -> pd.DataFrame:
    """Compute false discovery rate by group as a placeholder for the audit pipeline."""
    if decision_col not in df.columns:
        raise KeyError(f"Column '{decision_col}' not found in DataFrame.")
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")

    result = []
    for group, group_df in df.groupby(group_col, dropna=False):
        approvals = group_df[group_df[decision_col] == 1]
        if approvals.empty:
            fdr = float("nan")
        else:
            default_count = int((approvals[label_col] == 1).sum()) if label_col and label_col in approvals.columns else 0
            fdr = default_count / len(approvals)
        result.append({group_col: group, "fdr": fdr, "approved_count": len(approvals)})
    return pd.DataFrame(result)
