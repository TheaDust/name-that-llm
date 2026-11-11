import numpy as np
import pandas as pd
import pytest

from ntl.constants import ID_COLUMN, LABEL_COLUMN, MODEL_LABELS, PROMPT_COLUMN, TEXT_COLUMN
from ntl.data import (
    assign_folds,
    fold_table,
    grouped_cv_splits,
    thin_validation_indices,
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
        grouped_cv_splits(train, n_splits=4, seed=3)
    ):
        train_prompts = set(train.iloc[train_idx][PROMPT_COLUMN])
        val_prompts = set(train.iloc[val_idx][PROMPT_COLUMN])
        assert train_prompts.isdisjoint(val_prompts)
        seen[val_idx] = fold

    assert (seen >= 0).all()


def test_thinned_validation_matches_hidden_test_shape():
    train = make_train()
    for _, val_idx in grouped_cv_splits(
        train, n_splits=4, seed=3, passages_per_prompt=2
    ):
        val = train.iloc[val_idx]
        assert val.groupby(PROMPT_COLUMN).size().max() <= 2


def test_thinning_is_deterministic_and_respects_cap():
    train = make_train()
    indices = np.arange(len(train))
    first = thin_validation_indices(train, indices, 3, seed=11)
    second = thin_validation_indices(train, indices, 3, seed=11)
    assert np.array_equal(first, second)

    val = train.iloc[first]
    assert val.groupby(PROMPT_COLUMN).size().max() <= 3


def test_thinning_with_large_cap_keeps_everything():
    train = make_train()
    indices = np.arange(len(train))
    kept = thin_validation_indices(train, indices, len(MODEL_LABELS), seed=0)
    assert np.array_equal(kept, indices)


def test_fold_table_has_expected_columns():
    train = make_train()
    table = fold_table(train, n_splits=4, seed=1)
    assert list(table.columns) == [ID_COLUMN, PROMPT_COLUMN, LABEL_COLUMN, "fold"]
    assert len(table) == len(train)
