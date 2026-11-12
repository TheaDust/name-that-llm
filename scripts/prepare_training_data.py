"""Export ready-to-train CV folds from competition data and a cleaned pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ntl.constants import DEFAULT_SEED
from ntl.data import DEFAULT_DATA_DIR, load_train
from ntl.open_pool import load_clean_pool
from ntl.training_data import sample_training_pool, training_folds

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA_DIR / "train.csv")
    parser.add_argument("--folds", type=Path, default=PROCESSED / "folds.parquet")
    parser.add_argument("--pool", type=Path, default=PROCESSED / "open_pool_clean.parquet")
    parser.add_argument("--competition-only", action="store_true")
    parser.add_argument("--max-pool-rows-per-class", type=int, default=1000)
    parser.add_argument("--pool-weight-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=PROCESSED / "training_splits")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    train = load_train(args.train)
    folds = pd.read_parquet(args.folds)
    pool = None
    inputs = {"train": args.train, "folds": args.folds}
    if not args.competition_only:
        pool = sample_training_pool(
            load_clean_pool(args.pool),
            max_rows_per_class=args.max_pool_rows_per_class,
            seed=args.seed,
        )
        inputs["pool"] = args.pool
    prepared = list(training_folds(train, folds, pool, pool_weight_ratio=args.pool_weight_ratio))
    # A fresh directory keeps different sampling/weight configurations separate.
    args.out.mkdir(parents=True, exist_ok=False)
    if pool is not None:
        pool.to_parquet(args.out / "sampled_pool.parquet", index=False)
    report = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "competition_only": args.competition_only,
        "seed": args.seed,
        "max_pool_rows_per_class": args.max_pool_rows_per_class,
        "pool_weight_ratio": args.pool_weight_ratio,
        "pool_rows": 0 if pool is None else len(pool),
        "folds": [],
    }
    for fold, training, validation in prepared:
        directory = args.out / f"fold_{fold}"
        directory.mkdir()
        training.to_parquet(directory / "train.parquet", index=False)
        validation.to_parquet(directory / "validation.parquet", index=False)
        weights = training.groupby("data_source")["sample_weight"].sum().to_dict()
        report["folds"].append({
            "fold": fold,
            "train_rows": len(training),
            "validation_rows": len(validation),
            "weight_totals": weights,
        })
        print(f"fold {fold}: {len(training)} train / {len(validation)} validation; {weights}")
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
