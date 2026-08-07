from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_numeric_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CSV into column-oriented numpy arrays."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = reader.fieldnames or []
    data: dict[str, np.ndarray] = {}
    for column in columns:
        data[column] = np.array([float(row[column]) for row in rows], dtype=float)
    return data


def filter_rows(data: dict[str, np.ndarray], **filters: float) -> dict[str, np.ndarray]:
    """Return rows matching all exact-value filters."""
    if not filters:
        return data

    mask = np.ones_like(next(iter(data.values())), dtype=bool)
    for key, value in filters.items():
        mask &= np.isclose(data[key], value)

    return {key: values[mask] for key, values in data.items()}
