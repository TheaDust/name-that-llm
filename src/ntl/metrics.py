"""Evaluation metrics for the competition.

The official metric is balanced accuracy: mean per-class recall over the 18
candidate models. Per-class recall is reported alongside it so a route cannot
hide a weak model behind a strong average.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from .constants import MODEL_LABELS


def _check_inputs(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(y_true)} "
            f"and {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty prediction")

    known = set(labels)
    unknown = sorted((set(np.unique(y_true)) | set(np.unique(y_pred))) - known)
    if unknown:
        raise ValueError(f"predictions contain labels outside {len(labels)} classes: {unknown}")
    return y_true, y_pred


def per_class_recall(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
) -> pd.Series:
    """Return recall for every candidate label; absent classes are ``NaN``."""
    y_true, y_pred = _check_inputs(y_true, y_pred, labels)
    values = []
    for label in labels:
        mask = y_true == label
        support = int(mask.sum())
        values.append(float((y_pred[mask] == label).mean()) if support else np.nan)
    return pd.Series(
        values,
        index=pd.Index(list(labels), name="label"),
        name="recall",
        dtype=float,
    )


def balanced_accuracy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
) -> float:
    """Mean recall over the classes present in ``y_true``.

    This matches ``sklearn.metrics.balanced_accuracy_score``. Classes with no
    support are ignored rather than counted as zero.
    """
    recalls = per_class_recall(y_true, y_pred, labels).dropna()
    if recalls.empty:
        raise ValueError("no class in y_true has support")
    return float(recalls.mean())


def accuracy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
) -> float:
    """Plain top-1 accuracy, useful as a secondary diagnostic."""
    y_true, y_pred = _check_inputs(y_true, y_pred, labels)
    return float((y_true == y_pred).mean())


def class_support(
    y_true: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
) -> pd.Series:
    """Number of true examples per label."""
    y_true = np.asarray(y_true)
    known = set(labels)
    unknown = sorted(set(np.unique(y_true)) - known)
    if unknown:
        raise ValueError(f"y_true contains labels outside {len(labels)} classes: {unknown}")
    return pd.Series(
        [int((y_true == label).sum()) for label in labels],
        index=pd.Index(list(labels), name="label"),
        name="support",
        dtype=int,
    )


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
) -> dict:
    """Return a JSON-friendly metric report for one prediction set."""
    y_true, y_pred = _check_inputs(y_true, y_pred, labels)
    recalls = per_class_recall(y_true, y_pred, labels)
    support = class_support(y_true, labels)
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels),
        "accuracy": accuracy(y_true, y_pred, labels),
        "per_class_recall": {
            label: (None if pd.isna(value) else float(value))
            for label, value in recalls.items()
        },
        "support": {label: int(value) for label, value in support.items()},
    }


def confusion_matrix_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = MODEL_LABELS,
    *,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame.

    ``normalize`` can be ``"true"`` (row-normalised recall view), ``"pred"``
    (column-normalised precision view), or ``None`` for raw counts.
    """
    if normalize not in (None, "true", "pred"):
        raise ValueError("normalize must be one of None, 'true', 'pred'")
    y_true, y_pred = _check_inputs(y_true, y_pred, labels)
    matrix = sk_confusion_matrix(y_true, y_pred, labels=list(labels))
    if normalize is not None:
        axis = 1 if normalize == "true" else 0
        totals = matrix.sum(axis=axis, keepdims=True)
        matrix = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix, dtype=float),
            where=totals != 0,
        )
    return pd.DataFrame(
        matrix,
        index=pd.Index(list(labels), name="true"),
        columns=pd.Index(list(labels), name="pred"),
    )
