import numpy as np
import pandas as pd
import pytest

from ntl.constants import ID_COLUMN, LABEL_COLUMN, MODEL_LABELS, N_MODELS
from ntl.submission import (
    argmax_proba,
    read_proba,
    validate_proba,
    validate_submission,
    write_proba,
    write_submission,
)

A, B, C = MODEL_LABELS[:3]


def proba_frame(ids, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.dirichlet(np.ones(N_MODELS), size=len(ids))
    frame = pd.DataFrame(values, columns=list(MODEL_LABELS))
    frame.insert(0, ID_COLUMN, list(ids))
    return frame


def test_valid_probability_table_passes():
    frame = proba_frame(["a", "b", "c"])
    assert validate_proba(frame) is frame


def test_extra_column_is_rejected():
    frame = proba_frame(["a"])
    frame["extra"] = 1.0
    with pytest.raises(ValueError, match="exactly these columns"):
        validate_proba(frame)


def test_duplicate_ids_are_rejected():
    frame = proba_frame(["a", "a"])
    with pytest.raises(ValueError, match="duplicated ids"):
        validate_proba(frame)


def test_bad_row_sum_is_rejected():
    frame = proba_frame(["a"])
    frame.loc[0, MODEL_LABELS[0]] += 0.1
    with pytest.raises(ValueError, match="sum to 1"):
        validate_proba(frame)


def test_out_of_range_probability_is_rejected():
    frame = proba_frame(["a"])
    frame.loc[0, MODEL_LABELS[0]] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_proba(frame)


def test_nan_probability_is_rejected():
    frame = proba_frame(["a"])
    frame.loc[0, MODEL_LABELS[0]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_proba(frame)


def test_id_set_must_match_when_requested():
    frame = proba_frame(["a", "b"])
    with pytest.raises(ValueError, match="id set does not match"):
        validate_proba(frame, ids=["a", "c"])


def test_argmax_proba_returns_id_and_label():
    frame = proba_frame(["a", "b"])
    predictions = argmax_proba(frame)
    assert list(predictions.columns) == [ID_COLUMN, LABEL_COLUMN]
    expected = np.asarray(MODEL_LABELS)[
        frame[list(MODEL_LABELS)].to_numpy().argmax(axis=1)
    ]
    assert predictions[LABEL_COLUMN].tolist() == expected.tolist()


def test_write_and_read_proba_roundtrip(tmp_path):
    frame = proba_frame(["a", "b"], seed=3)
    path = write_proba(frame, tmp_path / "proba.parquet", ids=["a", "b"])
    loaded = read_proba(path, ids=["a", "b"])
    pd.testing.assert_frame_equal(loaded, frame)


def test_validate_submission_accepts_valid_frame():
    predictions = pd.DataFrame({ID_COLUMN: ["a", "b"], LABEL_COLUMN: [A, B]})
    assert validate_submission(predictions, test_ids=["a", "b"]) is predictions


def test_validate_submission_rejects_invalid_label():
    predictions = pd.DataFrame({ID_COLUMN: ["a"], LABEL_COLUMN: ["unknown"]})
    with pytest.raises(ValueError, match="outside the 18"):
        validate_submission(predictions)


def test_validate_submission_rejects_missing_test_id():
    predictions = pd.DataFrame({ID_COLUMN: ["a"], LABEL_COLUMN: [A]})
    with pytest.raises(ValueError, match="id set does not match"):
        validate_submission(predictions, test_ids=["a", "b"])


def test_write_submission_reorders_to_test_order(tmp_path):
    predictions = pd.DataFrame(
        {ID_COLUMN: ["t1", "t2", "t3"], LABEL_COLUMN: [A, B, C]}
    )
    path = write_submission(
        predictions, tmp_path / "submission.parquet", test_ids=["t3", "t1", "t2"]
    )
    loaded = pd.read_parquet(path)
    assert loaded[ID_COLUMN].tolist() == ["t3", "t1", "t2"]
    assert loaded[LABEL_COLUMN].tolist() == [C, A, B]
