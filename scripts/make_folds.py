"""Create the shared prompt-grouped CV assignment.

Usage (from the repository root, after ``pip install -e .``):

    python scripts/make_folds.py
    python scripts/make_folds.py --n-splits 5 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ntl.constants import (
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    ID_COLUMN,
    LABEL_COLUMN,
    PROMPT_COLUMN,
)
from ntl.data import fold_table, load_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "folds.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=None, help="path to train.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output parquet")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train)
    table = fold_table(train, n_splits=args.n_splits, seed=args.seed)

    per_prompt = table.groupby(PROMPT_COLUMN)["fold"].nunique()
    if (per_prompt > 1).any():
        raise RuntimeError("some prompts landed in more than one fold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    summary = table.groupby("fold").agg(
        prompts=(PROMPT_COLUMN, "nunique"),
        rows=(ID_COLUMN, "size"),
    )
    label_counts = pd.crosstab(table["fold"], table[LABEL_COLUMN])
    print(f"wrote {len(table)} rows to {args.out}")
    print(summary.to_string())
    print(
        "rows per model and fold: "
        f"min={int(label_counts.to_numpy().min())}, "
        f"max={int(label_counts.to_numpy().max())}"
    )


if __name__ == "__main__":
    main()
