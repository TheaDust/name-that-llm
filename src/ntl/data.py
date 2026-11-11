"""Data loading, validation and prompt-grouped cross-validation.

Why grouped CV: all passages sharing a ``prompt_id`` answer the same question.
A random row split would put sibling answers in both train and validation and
massively inflate the score. Every splitter here keeps a prompt in exactly one
fold.

All indices returned by the CV helpers are **positional** (use ``df.iloc``),
matching scikit-learn's splitter API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import (
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    ID_COLUMN,
    LABEL_COLUMN,
    MODEL_LABEL_SET,
    N_MODELS,
    PROMPT_COLUMN,
    TEXT_COLUMN,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("NTL_DATA_DIR", _REPO_ROOT / "data" / "raw"))

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.parquet"


# --------------------------------------------------------------------- loading
def _resolve(path: str | Path | None, default_name: str) -> Path:
    return Path(path) if path is not None else DEFAULT_DATA_DIR / default_name


def load_train(path: str | Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load ``train.csv``, optionally validating its contract."""
    df = pd.read_csv(_resolve(path, TRAIN_FILE))
    return validate_train(df) if validate else df


def load_test(path: str | Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load ``test.csv`` (only ``id`` and ``text`` are guaranteed)."""
    df = pd.read_csv(_resolve(path, TEST_FILE))
    return validate_test(df) if validate else df


def load_sample_submission(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``sample_submission.parquet`` and check its basic shape."""
    df = pd.read_parquet(_resolve(path, SAMPLE_SUBMISSION_FILE))
    return validate_sample_submission(df)


# ------------------------------------------------------------------ validation
def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{name} contains duplicated columns: {duplicated}")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique_ids(df: pd.DataFrame, name: str) -> None:
    if df[ID_COLUMN].isna().any():
        raise ValueError(f"{name} contains null ids")
    duplicated = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].unique()
    if len(duplicated):
        preview = ", ".join(map(str, duplicated[:5]))
        raise ValueError(f"{name} contains {len(duplicated)} duplicated ids: {preview}")


def _require_text(df: pd.DataFrame, name: str) -> None:
    empty = df[TEXT_COLUMN].isna() | df[TEXT_COLUMN].astype(str).str.strip().eq("")
    if empty.any():
        raise ValueError(f"{name} contains {int(empty.sum())} null/empty texts")


def validate_train(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the training contract and return ``df`` unchanged."""
    _require_columns(
        df, [ID_COLUMN, PROMPT_COLUMN, TEXT_COLUMN, LABEL_COLUMN], "train"
    )
    if df.empty:
        raise ValueError("train is empty")

    _require_unique_ids(df, "train")
    _require_text(df, "train")

    if df[PROMPT_COLUMN].isna().any():
        raise ValueError("train contains null prompt_id values")
    empty_prompts = df[PROMPT_COLUMN].astype(str).str.strip().eq("")
    if empty_prompts.any():
        raise ValueError("train contains empty prompt_id values")

    unknown = sorted(set(df[LABEL_COLUMN].astype(str)) - MODEL_LABEL_SET)
    if unknown:
        raise ValueError(f"train contains labels outside the 18 valid models: {unknown}")

    duplicated_pairs = df.duplicated([PROMPT_COLUMN, LABEL_COLUMN], keep=False)
    if duplicated_pairs.any():
        sample = (
            df.loc[duplicated_pairs, [PROMPT_COLUMN, LABEL_COLUMN]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "each (prompt_id, model) pair must be unique; "
            f"{int(duplicated_pairs.sum())} duplicate rows, e.g. {sample}"
        )

    sizes = df.groupby(PROMPT_COLUMN, sort=False).size()
    oversized = sizes[sizes > N_MODELS]
    if len(oversized):
        raise ValueError(
            f"prompt groups cannot exceed {N_MODELS} rows (one per model); "
            f"found {len(oversized)} oversized groups, max={int(sizes.max())}"
        )

    return df


def validate_test(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the test contract and return ``df`` unchanged."""
    _require_columns(df, [ID_COLUMN, TEXT_COLUMN], "test")
    if df.empty:
        raise ValueError("test is empty")
    _require_unique_ids(df, "test")
    _require_text(df, "test")
    return df


def validate_sample_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the sample submission shape and label values."""
    _require_columns(df, [ID_COLUMN, LABEL_COLUMN], "sample submission")
    _require_unique_ids(df, "sample submission")
    if df[LABEL_COLUMN].isna().any():
        raise ValueError("sample submission contains null model labels")
    unknown = sorted(set(df[LABEL_COLUMN].astype(str)) - MODEL_LABEL_SET)
    if unknown:
        raise ValueError(
            f"sample submission contains labels outside the 18 valid models: {unknown}"
        )
    return df


# ------------------------------------------------------------------------ folds
def assign_folds(
    train: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Return one fold id per training row (0..n_splits-1).

    Splitting is stratified on ``model`` and grouped on ``prompt_id``, so every
    prompt lands in exactly one validation fold.
    """
    validate_train(train)
    n_prompts = train[PROMPT_COLUMN].nunique()
    if not 2 <= n_splits <= n_prompts:
        raise ValueError(
            f"n_splits must be between 2 and the number of prompts ({n_prompts}), "
            f"got {n_splits}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = np.full(len(train), -1, dtype=int)
    y = train[LABEL_COLUMN].to_numpy()
    groups = train[PROMPT_COLUMN].to_numpy()
    for fold, (_, val_idx) in enumerate(splitter.split(train, y, groups)):
        folds[val_idx] = fold

    if (folds < 0).any():
        raise RuntimeError("internal error: some rows were not assigned to a fold")
    return folds


def thin_validation_indices(
    train: pd.DataFrame,
    indices: np.ndarray,
    passages_per_prompt: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Keep at most ``passages_per_prompt`` rows per prompt from ``indices``.

    The hidden test does not contain all 18 answers for every question, so
    scoring on full validation folds is optimistic. Thinning validation to
    1-3 rows per prompt makes the offline estimate closer to the real task.
    """
    if passages_per_prompt < 1:
        raise ValueError("passages_per_prompt must be >= 1")
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return indices

    prompt_values = train[PROMPT_COLUMN].to_numpy()[indices]
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for prompt in pd.unique(prompt_values):
        positions = indices[prompt_values == prompt]
        take = min(passages_per_prompt, len(positions))
        keep.extend(rng.choice(positions, size=take, replace=False).tolist())
    return np.sort(np.asarray(keep, dtype=int))


def _iter_folds(
    train: pd.DataFrame,
    folds: np.ndarray,
    n_splits: int,
    passages_per_prompt: int | None,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for fold in range(n_splits):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        if passages_per_prompt is not None:
            val_idx = thin_validation_indices(
                train, val_idx, passages_per_prompt, seed=seed + fold
            )
        yield train_idx, val_idx


def grouped_cv_splits(
    train: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    passages_per_prompt: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx)`` positional indices for prompt-grouped CV.

    ``passages_per_prompt=None`` keeps full validation folds (use these to build
    OOF probabilities). An integer thins each validation fold to the hidden-test
    shape (use this for the headline score).
    """
    folds = assign_folds(train, n_splits=n_splits, seed=seed)
    return _iter_folds(train, folds, n_splits, passages_per_prompt, seed)


def fold_table(
    train: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a cacheable ``id, prompt_id, model, fold`` table."""
    folds = assign_folds(train, n_splits=n_splits, seed=seed)
    return pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            PROMPT_COLUMN: train[PROMPT_COLUMN].to_numpy(),
            LABEL_COLUMN: train[LABEL_COLUMN].to_numpy(),
            "fold": folds,
        }
    )
