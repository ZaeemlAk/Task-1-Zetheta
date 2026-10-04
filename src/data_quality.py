"""Data quality inspection for the audit dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a structured data-quality summary for a dataset.

    Returns a dictionary containing counts, dtypes, missing-value summaries, duplicates, class
    balance, and rough outlier indications. The function is intentionally conservative: it reports
    what the data supports without making assumptions about labels or protected groups.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    summary: dict[str, Any] = {
        "n_records": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if not df.empty:
        sample_numeric = df.select_dtypes(include=[np.number])
        if not sample_numeric.empty:
            numeric_summary: dict[str, Any] = {}
            for column in sample_numeric.columns:
                s = sample_numeric[column]
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((s < lower) | (s > upper)).sum())
                numeric_summary[column] = {
                    "min": float(s.min()) if not s.isna().all() else None,
                    "max": float(s.max()) if not s.isna().all() else None,
                    "mean": float(s.mean()) if not s.isna().all() else None,
                    "median": float(s.median()) if not s.isna().all() else None,
                    "outlier_count": outlier_count,
                }
            summary["numeric_summary"] = numeric_summary

    summary["class_balance"] = {}
    for column in df.columns:
        s = df[column]
        if pd.api.types.is_numeric_dtype(s):
            unique_count = s.nunique(dropna=True)
            if unique_count <= 10:
                summary["class_balance"][column] = s.value_counts(dropna=False).to_dict()
        else:
            unique_count = s.nunique(dropna=True)
            if unique_count <= 10:
                summary["class_balance"][column] = s.value_counts(dropna=False).to_dict()

    summary["potential_leakage"] = detect_potential_leakage(df)
    summary["data_quality_issues"] = summarise_quality_issues(df)
    return summary


def detect_potential_leakage(df: pd.DataFrame) -> list[str]:
    """Return a short list of suspected leakage indicators based on column names."""
    leakage_indicators = [
        "credit_score",
        "bureau_score",
        "default",
        "decision",
        "approved",
        "outcome",
        "risk_band",
        "credit_limit",
    ]
    matches = []
    lowered = {col.lower(): col for col in df.columns}
    for key in leakage_indicators:
        if key in lowered:
            matches.append(lowered[key])
    return matches


def summarise_quality_issues(df: pd.DataFrame) -> list[str]:
    """Summarise obvious data-quality concerns."""
    issues: list[str] = []
    if df.duplicated().any():
        issues.append(f"Duplicate rows detected: {int(df.duplicated().sum())}")

    missing_cols = df.columns[df.isna().all()]
    if len(missing_cols):
        issues.append(f"Columns with all missing values: {list(missing_cols)}")

    for column in df.columns:
        series = df[column]
        if series.dtype == object:
            if series.astype(str).str.contains(r"\s+", regex=True).any():
                issues.append(f"Whitespace inconsistencies detected in column '{column}'")

    return issues
