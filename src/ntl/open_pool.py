"""Cleaning for the public Name That LLM open pool.

The Kaggle dataset ``crtproduction/name-that-llm-open-pool`` aggregates 61
public passage collections. It is useful auxiliary training data, but it is
not a drop-in replacement for ``train.csv``:

* about 38% of rows are proxy generations, so their labels can be wrong;
* most prompts only contain one or two models, which turns prompt identity
  into a shortcut;
* passage length and typography drift away from the competition data.

``clean_open_pool`` applies a deterministic, documented filter and returns a
frame that routes can concatenate with their training folds. Competition
validation folds are never touched, and the pool must never be used as a
validation set.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
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
    TEXT_COLUMN,
)

POOL_PROMPT_COLUMN = "prompt_group"
FOLD_COLUMN = "pool_fold"

REQUIRED_POOL_COLUMNS = (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, POOL_PROMPT_COLUMN)

# Columns kept in the cleaned artifact. Everything else is provenance that the
# routes do not need at training time.
CLEAN_COLUMNS = (
    ID_COLUMN,
    TEXT_COLUMN,
    LABEL_COLUMN,
    POOL_PROMPT_COLUMN,
    "n_words",
    "source_dataset",
    "source_license",
    "style_score",
    "match_weight",
    FOLD_COLUMN,
)


@dataclass(frozen=True)
class PoolCleanConfig:
    """Filter thresholds. Defaults define the shared cleaning policy."""

    min_words: int = 40
    max_words: int = 250
    # A core prompt must contain at least this many distinct models in the pool.
    min_prompt_models: int = 8
    max_rows_per_pair: int = 3
    n_splits: int = DEFAULT_N_SPLITS
    seed: int = DEFAULT_SEED


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _optional(df: pd.DataFrame, column: str, default):
    return df[column] if column in df.columns else pd.Series(default, index=df.index)


def normalize_text(text: pd.Series) -> pd.Series:
    """Normalize whitespace and unicode so duplicate passages hash equally."""
    return text.astype(str).str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()


def _step(steps: list[dict], name: str, delta: int, remaining: int) -> None:
    """Record a signed row delta; negative means rows were removed."""
    steps.append({"step": name, "delta": int(delta), "remaining": int(remaining)})


def _prompt_model_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(POOL_PROMPT_COLUMN)[LABEL_COLUMN].transform("nunique")


def _canonical_prompt_groups(df: pd.DataFrame) -> pd.Series:
    """Use exact nonempty canonical prompts across sources, never fuzzy matches.

    Hash namespaces keep canonical values distinct from source-local IDs.
    Missing canonical metadata falls back to the existing prompt group.
    """
    original = df[POOL_PROMPT_COLUMN]
    if original.isna().any() or original.astype(str).str.strip().eq("").any():
        raise ValueError("open pool contains null/empty prompt groups")
    if "prompt_canon" not in df.columns:
        return original
    canonical = normalize_text(df["prompt_canon"].fillna(""))
    groups = "source:" + original.astype(str)
    present = canonical.ne("")
    groups.loc[present] = "canonical:" + canonical.loc[present].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return groups


def _cap_rows_per_pair(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    ordered = df.sort_values(
        ["match_weight", ID_COLUMN], ascending=[False, True], na_position="last"
    )
    return ordered.groupby([POOL_PROMPT_COLUMN, LABEL_COLUMN], sort=False).head(limit)


def _drop_duplicates_and_ambiguity(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    labels_per_text = df.groupby(TEXT_COLUMN)[LABEL_COLUMN].nunique()
    ambiguous = labels_per_text[labels_per_text > 1].index
    before = len(df)
    df = df[~df[TEXT_COLUMN].isin(ambiguous)]
    _step(steps, "ambiguous_text_label", len(df) - before, len(df))

    before = len(df)
    ordered = df.sort_values(
        ["match_weight", ID_COLUMN], ascending=[False, True], na_position="last"
    )
    df = ordered.drop_duplicates(TEXT_COLUMN, keep="first")
    _step(steps, "duplicate_text", len(df) - before, len(df))
    return df


def assign_pool_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Assign prompt-grouped folds used only for pool-internal diagnostics."""
    counts = df[LABEL_COLUMN].value_counts()
    if len(counts) < 2 or int(counts.min()) < n_splits:
        raise ValueError("every label needs at least n_splits rows to build pool folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(
        splitter.split(df, df[LABEL_COLUMN], df[POOL_PROMPT_COLUMN])
    ):
        folds[val_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("internal error: some pool rows were not assigned to a fold")
    return folds


def clean_open_pool(
    pool: pd.DataFrame,
    config: PoolCleanConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(cleaned, steps)`` for a raw open-pool frame.

    ``cleaned`` follows :data:`CLEAN_COLUMNS` and is sorted by id. ``steps`` is
    a small audit table with one row per filter.
    """
    config = config or PoolCleanConfig()
    _require_columns(pool, REQUIRED_POOL_COLUMNS, "open pool")

    steps: list[dict] = []
    df = pool.copy()
    _step(steps, "start", 0, len(df))

    before = len(df)
    df = df[df[TEXT_COLUMN].notna() & df[TEXT_COLUMN].astype(str).str.strip().ne("")]
    _step(steps, "empty_text", len(df) - before, len(df))

    before = len(df)
    df = df[df[LABEL_COLUMN].isin(MODEL_LABEL_SET)]
    _step(steps, "unknown_label", len(df) - before, len(df))

    df[TEXT_COLUMN] = normalize_text(df[TEXT_COLUMN])
    df["n_words"] = df[TEXT_COLUMN].str.split().str.len()

    before = len(df)
    df = df[df["n_words"].between(config.min_words, config.max_words)]
    _step(steps, "length", len(df) - before, len(df))

    before = len(df)
    proxy = pd.Series(False, index=df.index)
    for column in ("flag_kb_proxy", "flag_kb_proxy_after_ruling"):
        if column in df.columns:
            proxy |= df[column].fillna(False).astype(bool)
    df = df[~proxy]
    _step(steps, "proxy_generation", len(df) - before, len(df))

    df["match_weight"] = _optional(df, "match_weight", 1.0).fillna(1.0).astype(float)
    df["style_score"] = _optional(df, "style_score", np.nan)
    df["source_dataset"] = _optional(df, "source_dataset", "unknown").astype(str)
    df["source_license"] = _optional(df, "source_license", "unknown").astype(str)

    df = _drop_duplicates_and_ambiguity(df, steps)

    # Coverage, pair caps and diagnostic folds must all share the same grouping.
    df[POOL_PROMPT_COLUMN] = _canonical_prompt_groups(df)
    df["prompt_models"] = _prompt_model_counts(df)
    before = len(df)
    core = df[df["prompt_models"] >= config.min_prompt_models]
    _step(steps, "prompt_too_few_models", len(core) - before, len(core))

    before = len(core)
    core = _cap_rows_per_pair(core, config.max_rows_per_pair)
    _step(steps, "pair_cap", len(core) - before, len(core))

    cleaned = core.sort_values(ID_COLUMN).reset_index(drop=True)

    cleaned[FOLD_COLUMN] = assign_pool_folds(cleaned, n_splits=config.n_splits, seed=config.seed)

    for column in CLEAN_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
    cleaned = cleaned[list(CLEAN_COLUMNS)]
    _step(steps, "final", 0, len(cleaned))
    return cleaned, pd.DataFrame(steps)


def validate_clean_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the cleaned artifact before a route consumes it."""
    _require_columns(df, CLEAN_COLUMNS, "clean open pool")
    if df.empty:
        raise ValueError("clean open pool is empty")

    duplicated = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].unique()
    if len(duplicated):
        raise ValueError(f"clean open pool contains duplicated ids: {duplicated[:5]}")
    if df[TEXT_COLUMN].isna().any() or df[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError("clean open pool contains null/empty texts")
    unknown = sorted(set(df[LABEL_COLUMN]) - MODEL_LABEL_SET)
    if unknown:
        raise ValueError(f"clean open pool contains unknown labels: {unknown}")
    if df.groupby(POOL_PROMPT_COLUMN)[FOLD_COLUMN].nunique().gt(1).any():
        raise ValueError("each pool prompt must belong to exactly one fold")
    return df


def load_clean_pool(path: str | Path) -> pd.DataFrame:
    """Read and validate a cleaned pool parquet."""
    return validate_clean_pool(pd.read_parquet(path))


def pool_sample_weights(
    pool: pd.DataFrame,
    *,
    train_rows: int,
    pool_weight_ratio: float = 1.0,
) -> np.ndarray:
    """Weight the selected pool rows in their current order.

    Each represented class receives equal total weight. The pool's total
    weight is ``train_rows * pool_weight_ratio``, assuming competition rows
    each have weight 1. Call after every sampling/filtering operation and for
    each training fold; weights are deliberately not cached in the artifact.
    """
    if pool.empty:
        raise ValueError("selected pool is empty")
    if train_rows <= 0:
        raise ValueError("train_rows must be positive")
    if not np.isfinite(pool_weight_ratio) or pool_weight_ratio <= 0:
        raise ValueError("pool_weight_ratio must be finite and positive")
    labels = pool[LABEL_COLUMN]
    if labels.isna().any() or not labels.isin(MODEL_LABEL_SET).all():
        raise ValueError("selected pool contains invalid labels")
    counts = labels.value_counts()
    return (train_rows * pool_weight_ratio / (len(counts) * labels.map(counts))).to_numpy(
        dtype=float
    )


def config_dict(config: PoolCleanConfig) -> dict:
    """JSON-friendly view of a config."""
    return asdict(config)
