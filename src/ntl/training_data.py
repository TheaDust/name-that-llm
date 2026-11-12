"""Assemble shared competition CV folds with auxiliary training samples."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from .constants import DEFAULT_SEED
from .data import grouped_cv_splits
from .open_pool import normalize_text, pool_sample_weights, validate_clean_pool


def sample_training_pool(
    pool: pd.DataFrame,
    *,
    max_rows_per_class: int = 1000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Select a deterministic, per-class sample shared by all CV folds."""
    validate_clean_pool(pool)
    if max_rows_per_class <= 0:
        raise ValueError("max_rows_per_class must be positive")
    return pd.concat(
        [
            group.sort_values("id").sample(min(len(group), max_rows_per_class), random_state=seed)
            for _, group in pool.groupby("model", sort=True)
        ],
        ignore_index=True,
    ).sort_values("id").reset_index(drop=True)


def training_folds(
    train: pd.DataFrame,
    folds: pd.DataFrame,
    pool: pd.DataFrame | None = None,
    *,
    pool_weight_ratio: float = 1.0,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, training rows, validation rows) using a preselected pool.

    Training rows include ``data_source`` and ``sample_weight``. Validation
    contains competition rows only, with its original columns. Only ``text``
    is model input; ids, groups and data_source are metadata.
    """
    splits = list(grouped_cv_splits(train, folds))
    if pool is not None:
        validate_clean_pool(pool)
        if set(pool["id"]) & set(train["id"]):
            raise ValueError("pool ids overlap competition train ids")
        if set(normalize_text(pool["text"])) & set(normalize_text(train["text"])):
            raise ValueError("pool texts overlap competition train texts")

    for fold, (train_idx, val_idx) in enumerate(splits):
        competition = train.iloc[train_idx][["id", "text", "model", "prompt_id"]].copy()
        competition["data_source"] = "competition"
        competition["sample_weight"] = 1.0
        if pool is not None:
            auxiliary = pool[["id", "text", "model", "prompt_group"]].copy()
            auxiliary["data_source"] = "pool"
            auxiliary["sample_weight"] = pool_sample_weights(
                pool, train_rows=len(competition), pool_weight_ratio=pool_weight_ratio
            )
            assembled = pd.concat([competition, auxiliary], ignore_index=True)
        else:
            assembled = competition.reset_index(drop=True)
        yield fold, assembled, train.iloc[val_idx].reset_index(drop=True)
