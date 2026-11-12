"""Check that the local data layout matches what the scripts expect.

Usage (from the repository root, after ``pip install -e .``):

    python scripts/check_data.py
    python scripts/check_data.py --skip-pool
    python scripts/check_data.py --data-dir D:/ntl-data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ntl.data import (
    DEFAULT_DATA_DIR,
    grouped_cv_splits,
    load_sample_submission,
    load_test,
    load_train,
)
from ntl.open_pool import load_clean_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory with train.csv, test.csv and sample_submission.parquet",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--skip-pool", action="store_true", help="do not require the open pool")
    return parser.parse_args()


def size_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    print(f"raw dir:       {args.data_dir}")
    print(f"processed dir: {args.processed_dir}")

    print("\n[raw]")
    train = test = sample = None
    for name, loader in (
        ("train.csv", load_train),
        ("test.csv", load_test),
        ("sample_submission.parquet", load_sample_submission),
    ):
        path = args.data_dir / name
        if not path.exists():
            errors.append(f"missing {path}")
            print(f"MISSING  {name}")
            continue
        try:
            frame = loader(path)
        except ValueError as exc:
            errors.append(f"invalid {path}: {exc}")
            print(f"INVALID  {name}: {exc}")
            continue
        print(f"OK       {name:28s} {len(frame):>6d} rows  {size_mb(path)}")
        if name == "train.csv":
            train = frame
        elif name == "test.csv":
            test = frame
        else:
            sample = frame

    if test is not None and sample is not None and set(sample["id"]) != set(test["id"]):
        print("WARN     sample_submission ids do not match test ids")

    print("\n[processed]")
    folds_path = args.processed_dir / "folds.parquet"
    if not folds_path.exists():
        errors.append(f"missing {folds_path}")
        print("MISSING  folds.parquet")
        print("         run: python scripts/make_folds.py")
    elif train is None:
        print("SKIP     folds.parquet (train.csv is missing)")
    else:
        try:
            folds = pd.read_parquet(folds_path)
            splits = list(grouped_cv_splits(train, folds))
        except (ValueError, KeyError) as exc:
            errors.append(f"invalid {folds_path}: {exc}")
            print(f"INVALID  folds.parquet: {exc}")
        else:
            sizes = ", ".join(
                f"fold{k}: {len(val)} val / {len(tr)} train"
                for k, (tr, val) in enumerate(splits)
            )
            print(f"OK       folds.parquet{'':<14} {len(folds):>6d} rows  {size_mb(folds_path)}")
            print(f"         {sizes}")

    pool_path = args.processed_dir / "open_pool_clean.parquet"
    if args.skip_pool:
        print("SKIP     open_pool_clean.parquet (--skip-pool)")
    elif not pool_path.exists():
        print("MISSING  open_pool_clean.parquet (optional)")
        print("         run: python scripts/prepare_open_pool.py")
    else:
        try:
            pool = load_clean_pool(pool_path)
        except ValueError as exc:
            errors.append(f"invalid {pool_path}: {exc}")
            print(f"INVALID  open_pool_clean.parquet: {exc}")
        else:
            counts = pool["model"].value_counts()
            print(
                f"OK       open_pool_clean.parquet   {len(pool):>6d} rows  {size_mb(pool_path)}"
            )
            print(
                f"         {pool['model'].nunique()} classes, "
                f"{counts.min()}..{counts.max()} rows/class"
            )

    print("\n[result]")
    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("OK: data layout is ready")


if __name__ == "__main__":
    main()
