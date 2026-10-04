"""Exploratory data analysis utilities for the fairness audit."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def describe_group_summary(df: pd.DataFrame, group_col: str, decision_col: str) -> pd.DataFrame:
    """Return a summary table for approval-rate comparisons by group."""
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if decision_col not in df.columns:
        raise KeyError(f"Column '{decision_col}' not found in DataFrame.")

    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            applicants=(decision_col, "size"),
            approvals=(decision_col, "sum"),
            rejections=(decision_col, lambda s: (s == 0).sum()),
            approval_rate=(decision_col, "mean"),
        )
        .reset_index()
    )
    summary["approval_rate"] = summary["approval_rate"].round(4)
    return summary


def plot_approval_rate_by_group(df: pd.DataFrame, group_col: str, decision_col: str, title: str) -> Any:
    """Plot approval rate by group for quick audit review."""
    summary = describe_group_summary(df, group_col, decision_col)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x=group_col, y="approval_rate")
    plt.title(title)
    plt.ylabel("Approval rate")
    plt.xlabel(group_col.replace("_", " ").title())
    plt.tight_layout()
    return plt.gcf()
