import numpy as np
import pandas as pd
import pytest

from ntl.constants import ID_COLUMN, LABEL_COLUMN, MODEL_LABELS, PROMPT_COLUMN, TEXT_COLUMN
from ntl.data import (
    assign_folds,
    fold_table,
    grouped_cv_splits,
    validate_test,
    validate_train,
)


def make_train(n_prompts: int = 12) -> pd.DataFrame:
    rows = []
    for prompt in range(n_prompts):
        for model in MODEL_LABELS:
            rows.append(
                {
                    ID_COLUMN: f"p{prompt}-{model}",
                    PROMPT_COLUMN: f"p{prompt:03d}",
                    TEXT_COLUMN: f"text {prompt} {model}",
                    LABEL_COLUMN: model,
                }
            )
    return pd.DataFrame(rows)


def test_validate_train_accepts_expected_shape():
    train = make_train()
    assert validate_train(train) is train


def test_validate_train_rejects_duplicate_ids():
    train = make_train()
    train.loc[train.index[1], ID_COLUMN] = train.loc[train.index[0], ID_COLUMN]
    with pytest.raises(ValueError, match="duplicated ids"):
        validate_train(train)


def test_validate_train_rejects_duplicate_prompt_model_pairs():
    train = make_train()
    train.loc[train.index[1], LABEL_COLUMN] = train.loc[train.index[0], LABEL_COLUMN]
    with pytest.raises(ValueError, match="pair must be unique"):
        validate_train(train)


def test_validate_train_rejects_unknown_label():
    train = make_train()
    train.loc[train.index[0], LABEL_COLUMN] = "not-a-model"
    with pytest.raises(ValueError, match="outside the 18"):
        validate_train(train)


def test_validate_train_rejects_empty_text():
    train = make_train()
    train.loc[train.index[0], TEXT_COLUMN] = "  "
    with pytest.raises(ValueError, match="null/empty"):
        validate_train(train)


def test_validate_test_accepts_minimal_columns():
    test = pd.DataFrame({ID_COLUMN: ["a", "b"], TEXT_COLUMN: ["x", "y"]})
    assert validate_test(test) is test


def test_assign_folds_keeps_each_prompt_in_one_fold():
    train = make_train()
    folds = assign_folds(train, n_splits=4, seed=7)

    assert folds.shape == (len(train),)
    assert set(folds) == {0, 1, 2, 3}
    per_prompt = pd.crosstab(train[PROMPT_COLUMN], folds)
    assert per_prompt.shape[1] == 4
    assert ((per_prompt > 0).sum(axis=1) == 1).all()
    assert (per_prompt.max(axis=1) == len(MODEL_LABELS)).all()


def test_grouped_cv_splits_have_no_prompt_overlap_and_full_coverage():
    train = make_train()
    seen = np.full(len(train), -1)
    for fold, (train_idx, val_idx) in enumerate(
        grouped_cv_splits(train, fold_table(train, n_splits=4, seed=3))
    ):
        train_prompts = set(train.iloc[train_idx][PROMPT_COLUMN])
        val_prompts = set(train.iloc[val_idx][PROMPT_COLUMN])
        assert train_prompts.isdisjoint(val_prompts)
        seen[val_idx] = fold

    assert (seen >= 0).all()


def test_fold_table_has_expected_columns():
    train = make_train()
    table = fold_table(train, n_splits=4, seed=1)
    assert list(table.columns) == [ID_COLUMN, PROMPT_COLUMN, LABEL_COLUMN, "fold"]
    assert len(table) == len(train)


def test_shared_folds_align_after_training_rows_are_reordered():
    train = make_train()
    folds = fold_table(train, n_splits=4, seed=3)
    reordered = train.sample(frac=1, random_state=8)
    for fold, (_, val_idx) in enumerate(grouped_cv_splits(reordered, folds)):
        assert set(reordered.iloc[val_idx][ID_COLUMN]) == set(
            folds.loc[folds["fold"] == fold, ID_COLUMN]
        )


@pytest.mark.parametrize("damage", ["missing_id", "label", "prompt", "leak", "null", "fractional"])
def test_shared_folds_reject_mismatches_and_leakage(damage):
    train = make_train()
    folds = fold_table(train, n_splits=4)
    if damage == "missing_id":
        folds = folds.iloc[1:]
    elif damage == "label":
        folds.loc[0, LABEL_COLUMN] = MODEL_LABELS[1]
    elif damage == "prompt":
        folds.loc[0, PROMPT_COLUMN] = "another-prompt"
    elif damage == "leak":
        folds.loc[0, "fold"] = (folds.loc[0, "fold"] + 1) % 4
    elif damage == "null":
        folds.loc[0, "fold"] = np.nan
    else:
        folds["fold"] = folds["fold"].astype(float) + 0.5
    with pytest.raises(ValueError):
        list(grouped_cv_splits(train, folds))
