"""Probability and submission contracts.

Every route writes the same two probability tables:

* ``oof_proba.parquet``  - one out-of-fold row per training id;
* ``test_proba.parquet`` - one row per test id.

Both have the columns ``id`` plus the 18 model names. These helpers validate
that contract and turn probabilities into a checked
``submission.parquet``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ID_COLUMN,
    LABEL_COLUMN,
    MODEL_LABELS,
    MODEL_LABEL_SET,
)


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{name} contains duplicated columns: {duplicated}")
    missing = [column for column in required if column not in df.columns]
    extra = [column for column in df.columns if column not in required]
    if missing or extra:
        raise ValueError(
            f"{name} must have exactly these columns: {list(required)}; "
            f"missing={missing}, extra={extra}"
        )


def _checked_ids(ids: Sequence[str] | pd.Index, name: str) -> pd.Index:
    index = pd.Index(list(ids)) if not isinstance(ids, pd.Index) else ids
    if index.hasnans:
        raise ValueError(f"{name} contains null ids")
    if index.has_duplicates:
        duplicated = index[index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"{name} contains duplicated ids, e.g. {duplicated}")
    return index


def _compare_id_sets(actual: pd.Index, expected: pd.Index, where: str) -> None:
    if set(actual) == set(expected):
        return
    missing = list(set(expected) - set(actual))[:5]
    extra = list(set(actual) - set(expected))[:5]
    raise ValueError(
        f"{where} id set does not match: {len(actual)} rows vs {len(expected)} expected; "
        f"missing={missing}, extra={extra}"
    )


# -------------------------------------------------------------------- proba io
def validate_proba(
    proba: pd.DataFrame,
    *,
    ids: Sequence[str] | pd.Index | None = None,
    atol: float = 1e-4,
) -> pd.DataFrame:
    """Validate the probability-table contract and return ``proba``.

    Checks: exact columns, unique/non-null ids, finite values in [0, 1], and
    each row summing to 1 within ``atol``. If ``ids`` is given, the id set must
    match exactly (order is free).
    """
    required = [ID_COLUMN, *MODEL_LABELS]
    _require_columns(proba, required, "probability table")

    actual_ids = _checked_ids(proba[ID_COLUMN], "probability table")
    if ids is not None:
        expected_ids = _checked_ids(ids, "expected ids")
        _compare_id_sets(actual_ids, expected_ids, "probability table")

    try:
        values = proba[list(MODEL_LABELS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability columns must all be numeric") from exc

    if not np.isfinite(values).all():
        raise ValueError("probability table contains NaN or infinite values")
    if (values < -atol).any() or (values > 1.0 + atol).any():
        raise ValueError("probabilities must lie in [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=atol))[:5]
        raise ValueError(
            "each probability row must sum to 1; "
            f"bad rows (positions)={bad.tolist()}, sums={row_sums[bad].tolist()}"
        )
    return proba


def read_proba(
    path: str | Path,
    *,
    ids: Sequence[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """Read a probability parquet and validate it."""
    proba = pd.read_parquet(path)
    return validate_proba(proba, ids=ids)


def write_proba(
    proba: pd.DataFrame,
    path: str | Path,
    *,
    ids: Sequence[str] | pd.Index | None = None,
) -> Path:
    """Validate a probability table and write it as parquet."""
    validate_proba(proba, ids=ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proba.to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------ submission
def argmax_proba(
    proba: pd.DataFrame,
    *,
    ids: Sequence[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """Turn a probability table into ``id, model`` predictions."""
    validate_proba(proba, ids=ids)
    values = proba[list(MODEL_LABELS)].to_numpy(dtype=float)
    chosen = np.asarray(list(MODEL_LABELS), dtype=object)[values.argmax(axis=1)]
    return pd.DataFrame({ID_COLUMN: proba[ID_COLUMN].to_numpy(), LABEL_COLUMN: chosen})


def validate_submission(
    predictions: pd.DataFrame,
    *,
    test_ids: Sequence[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """Validate a final ``id, model`` submission and return it."""
    _require_columns(predictions, [ID_COLUMN, LABEL_COLUMN], "predictions")
    actual_ids = _checked_ids(predictions[ID_COLUMN], "predictions")
    if predictions[LABEL_COLUMN].isna().any():
        raise ValueError("predictions contain null model labels")

    unknown = sorted(set(predictions[LABEL_COLUMN].astype(str)) - MODEL_LABEL_SET)
    if unknown:
        raise ValueError(f"predictions contain labels outside the 18 classes: {unknown}")

    if test_ids is not None:
        expected_ids = _checked_ids(test_ids, "test ids")
        _compare_id_sets(actual_ids, expected_ids, "predictions")

    return predictions


def write_submission(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    test_ids: Sequence[str] | pd.Index | None = None,
) -> Path:
    """Validate and write ``submission.parquet``.

    When ``test_ids`` is provided the output is reordered to match the test set.
    """
    validate_submission(predictions, test_ids=test_ids)
    if test_ids is not None:
        target_ids = _checked_ids(test_ids, "test ids")
        predictions = (
            predictions.set_index(ID_COLUMN)
            .loc[target_ids, [LABEL_COLUMN]]
            .rename_axis(ID_COLUMN)
            .reset_index()
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path
