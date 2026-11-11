"""Turn a test probability table into a validated submission.

Usage:

    python scripts/make_submission.py \
        --proba outputs/ensemble/test_proba.parquet \
        --out submissions/submission.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ntl.constants import ID_COLUMN, LABEL_COLUMN
from ntl.data import load_test
from ntl.submission import argmax_proba, read_proba, write_submission

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBA = REPO_ROOT / "outputs" / "ensemble" / "test_proba.parquet"
DEFAULT_OUT = REPO_ROOT / "submissions" / "submission.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proba", type=Path, default=DEFAULT_PROBA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test", type=Path, default=None, help="path to test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test = load_test(args.test)
    test_ids = test[ID_COLUMN]
    proba = read_proba(args.proba, ids=test_ids)
    predictions = argmax_proba(proba, ids=test_ids)
    write_submission(predictions, args.out, test_ids=test_ids)

    counts = predictions[LABEL_COLUMN].value_counts()
    print(f"wrote {len(predictions)} predictions to {args.out}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
