import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from ntl.constants import MODEL_LABELS
from ntl.metrics import (
    accuracy,
    balanced_accuracy,
    class_support,
    confusion_matrix_df,
    evaluate,
    per_class_recall,
)

A, B, C = MODEL_LABELS[:3]


def test_per_class_recall_and_balanced_accuracy():
    y_true = [A, B, A, B]
    y_pred = [A, B, B, B]

    recalls = per_class_recall(y_true, y_pred, labels=[A, B])
    assert recalls.loc[A] == pytest.approx(0.5)
    assert recalls.loc[B] == pytest.approx(1.0)
    assert balanced_accuracy(y_true, y_pred, labels=[A, B]) == pytest.approx(0.75)
    assert balanced_accuracy(y_true, y_pred, labels=[A, B]) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


def test_absent_class_is_nan_and_ignored():
    y_true = [A, B, A]
    y_pred = [A, B, B]

    recalls = per_class_recall(y_true, y_pred, labels=[A, B, C])
    assert np.isnan(recalls.loc[C])
    assert balanced_accuracy(y_true, y_pred, labels=[A, B, C]) == pytest.approx(0.75)


def test_accuracy_and_support():
    y_true = [A, A, B]
    y_pred = [A, B, B]
    assert accuracy(y_true, y_pred, labels=[A, B]) == pytest.approx(2 / 3)
    support = class_support(y_true, labels=[A, B, C])
    assert support.loc[A] == 2
    assert support.loc[B] == 1
    assert support.loc[C] == 0


def test_evaluate_is_json_friendly():
    report = evaluate([A, B, A], [A, B, B], labels=[A, B, C])
    assert report["balanced_accuracy"] == pytest.approx(0.75)
    assert report["per_class_recall"][C] is None
    assert report["support"][C] == 0


def test_confusion_matrix_is_labelled_and_normalisable():
    y_true = [A, A, B]
    y_pred = [A, B, B]
    matrix = confusion_matrix_df(y_true, y_pred, labels=[A, B])
    assert list(matrix.index) == [A, B]
    assert list(matrix.columns) == [A, B]
    assert matrix.to_numpy().sum() == 3

    normalised = confusion_matrix_df(y_true, y_pred, labels=[A, B], normalize="true")
    assert np.allclose(normalised.sum(axis=1), 1.0)


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        balanced_accuracy([A], ["unknown"], labels=[A, B])
