"""Validate a final submission against the test ids and the label set.

Usage:

    python scripts/validate_submission.py
    python scripts/validate_submission.py --submission submissions/submission.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ntl.constants import ID_COLUMN, LABEL_COLUMN
from ntl.data import load_test
from ntl.submission import validate_submission

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = REPO_ROOT / "submissions" / "submission.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--test", type=Path, default=None, help="path to test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test = load_test(args.test)
    submission = pd.read_parquet(args.submission)
    validate_submission(submission, test_ids=test[ID_COLUMN])

    counts = submission[LABEL_COLUMN].value_counts()
    print(f"OK: {len(submission)} rows, {counts.size} distinct labels")
    print(counts.to_string())


if __name__ == "__main__":
    main()
