"""SHAP analysis scaffolding for model explainability."""

from __future__ import annotations

from typing import Any

import pandas as pd
import shap


def explain_model(model: Any, X: pd.DataFrame) -> dict[str, Any]:
    """Return a lightweight SHAP explainer structure for a fitted model."""
    if X.empty:
        raise ValueError("Input feature matrix cannot be empty.")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return {
        "explainer": explainer,
        "shap_values": shap_values,
    }
