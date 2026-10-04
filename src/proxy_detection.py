"""Proxy variable detection utilities for the fairness audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def mutual_information_score(df: pd.DataFrame, protected_col: str, feature_col: str) -> float:
    """Return a lightweight mutual-information-like score for a categorical pair.

    This function avoids strong assumptions about the underlying distribution and remains useful for
    auditing suspected proxies when the dataset is still being prepared.
    """
    if protected_col not in df.columns or feature_col not in df.columns:
        raise KeyError("Protected and feature columns must both exist in the DataFrame.")

    p = df[protected_col].dropna()
    f = df[feature_col].dropna()
    if p.empty or f.empty:
        return 0.0

    joint = pd.crosstab(p, f)
    joint = joint / joint.to_numpy().sum()
    p_marginal = joint.sum(axis=1)
    f_marginal = joint.sum(axis=0)
    eps = 1e-12
    mi = 0.0
    for i, row in joint.iterrows():
        for j, value in row.items():
            if value <= 0:
                continue
            prob = value
            mi += prob * np.log((prob + eps) / ((p_marginal[i] * f_marginal[j]) + eps))
    return float(mi)


def cramers_v(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Compute Cramér's V for two categorical variables."""
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError("Both columns must exist in the DataFrame.")

    x = df[x_col].dropna()
    y = df[y_col].dropna()
    if x.empty or y.empty:
        return 0.0

    contingency = pd.crosstab(x, y)
    chi2 = float(((contingency.values - contingency.values.sum() * contingency.div(contingency.sum(axis=1), axis=0).values * contingency.div(contingency.sum(axis=0), axis=1).values) ** 2).sum())
    n = contingency.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    min_dim = min(r, k)
    return float(np.sqrt(phi2 / (min_dim - 1))) if min_dim > 1 else 0.0


def find_proxy_candidates(df: pd.DataFrame, protected_cols: list[str], threshold: float = 0.05) -> dict[str, list[str]]:
    """Find features with unusually strong association to protected attributes."""
    results: dict[str, list[str]] = {}
    for protected in protected_cols:
        if protected not in df.columns:
            continue
        flagged = []
        for feature in df.columns:
            if feature == protected:
                continue
            value = mutual_information_score(df, protected, feature)
            if value >= threshold:
                flagged.append(feature)
        results[protected] = flagged
    return results
