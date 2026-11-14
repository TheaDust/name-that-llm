"""Clean the public Name That LLM open pool into a training-ready parquet.

Usage (from the repository root, after ``pip install -e .``):

    python scripts/prepare_open_pool.py
    python scripts/prepare_open_pool.py --max-rows-per-pair 2 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ntl.constants import LABEL_COLUMN, MODEL_LABELS
from ntl.open_pool import (
    POOL_PROMPT_COLUMN,
    PoolCleanConfig,
    clean_open_pool,
    config_dict,
    normalize_text,
    validate_clean_pool,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "archive" / "ntl_open_pool.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "open_pool_clean.parquet"
DEFAULT_REPORT = REPO_ROOT / "data" / "processed" / "open_pool_clean.json"
DEFAULT_TRAIN = REPO_ROOT / "data" / "raw" / "train.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--skip-train-check", action="store_true")
    parser.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Explicitly allow a partial-class pool; default rejects missing model labels.",
    )
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--min-prompt-models", type=int, default=8)
    parser.add_argument("--max-rows-per-pair", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def check_no_train_overlap(cleaned: pd.DataFrame, train_path: Path) -> None:
    """Fail loudly if any pool row leaks a competition training row."""
    if not train_path.exists():
        print(f"train file not found, skipping overlap check: {train_path}")
        return
    train = pd.read_csv(train_path)
    id_overlap = set(cleaned["id"]) & set(train["id"])
    if id_overlap:
        raise SystemExit(f"pool ids overlap train ids: {sorted(id_overlap)[:5]}")

    train_text = set(normalize_text(train["text"]))
    text_overlap = set(cleaned["text"]) & train_text
    if text_overlap:
        raise SystemExit(f"pool texts overlap train texts: {len(text_overlap)} rows")
    print(f"overlap check: 0 shared ids, 0 shared texts with {train_path.name}")


def main() -> None:
    args = parse_args()
    config = PoolCleanConfig(
        min_words=args.min_words,
        max_words=args.max_words,
        min_prompt_models=args.min_prompt_models,
        max_rows_per_pair=args.max_rows_per_pair,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    print(f"reading {args.input}")
    raw = pd.read_parquet(args.input)
    print(f"raw rows: {len(raw):,}")

    cleaned, steps = clean_open_pool(raw, config)
    cleaned = validate_clean_pool(cleaned)  # contract check before writing

    class_counts = cleaned[LABEL_COLUMN].value_counts().reindex(MODEL_LABELS).fillna(0)
    missing_classes = class_counts[class_counts.eq(0)].index.tolist()
    if missing_classes and not args.allow_missing_classes:
        raise SystemExit(
            f"cleaning removed all rows for classes: {missing_classes}. "
            "Inspect prompt grouping and filters; use --allow-missing-classes "
            "only for an intentional partial-class pool. No output was written."
        )

    check_no_train_overlap(cleaned, args.train)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.out, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.out),
        "output_sha256": sha256(args.out),
        "config": config_dict(config),
        "prompt_group_policy": "exact_normalized_canonical_else_source_group_v1",
        "missing_classes": missing_classes,
        "allow_missing_classes": args.allow_missing_classes,
        "rows": len(cleaned),
        "prompts": int(cleaned[POOL_PROMPT_COLUMN].nunique()),
        "sources": int(cleaned["source_dataset"].nunique()),
        "classes": {str(k): int(v) for k, v in class_counts.items()},
        "steps": steps.to_dict("records"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\ncleaning steps (delta < 0 means rows removed):")
    print(steps.to_string(index=False))
    print(f"\nclean rows: {len(cleaned):,} | prompts: {cleaned[POOL_PROMPT_COLUMN].nunique():,}")
    print("\nrows per class:")
    print(class_counts.astype(int).to_string())
    print(f"\nwrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
