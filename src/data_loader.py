"""Dataset loading utilities for the credit-scoring bias audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".feather"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a supported file format.

    Args:
        path: Path to the source dataset.

    Returns:
        A pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    extension = dataset_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported dataset format: {extension}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".csv":
        df = pd.read_csv(dataset_path)
    elif extension == ".json":
        df = pd.read_json(dataset_path)
    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    elif extension == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif extension == ".feather":
        df = pd.read_feather(dataset_path)
    else:
        raise ValueError(f"No loader configured for extension: {extension}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Loaded object is not a pandas.DataFrame.")

    return df


def find_dataset(root: str | Path) -> list[Path]:
    """Search a directory for likely data files.

    Args:
        root: Directory to scan.

    Returns:
        A list of candidate dataset files.
    """
    search_root = Path(root)
    if not search_root.exists():
        return []

    matches: list[Path] = []
    for path in search_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            matches.append(path)
    return sorted(matches)


def normalise_decision_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...] = ()) -> str | None:
    """Return the first matching decision column name.

    The audit brief uses a few different label naming conventions, so this helper makes a best
    effort without inventing schema assumptions.
    """
    candidate_names = list(candidates)
    if not candidate_names:
        candidate_names = [
            "credit_decision",
            "decision",
            "loan_decision",
            "approval_status",
            "approved",
            "is_approved",
            "approval",
        ]

    for name in candidate_names:
        if name in df.columns:
            return name
    return None


def normalise_default_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...] = ()) -> str | None:
    """Return the first matching default outcome column name."""
    candidate_names = list(candidates)
    if not candidate_names:
        candidate_names = [
            "default_outcome",
            "six_month_default",
            "default",
            "default_flag",
            "is_default",
            "loan_default",
            "delinquent",
        ]

    for name in candidate_names:
        if name in df.columns:
            return name
    return None
