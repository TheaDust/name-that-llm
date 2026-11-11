import json

import pytest
from sklearn.metrics import balanced_accuracy_score

from ntl.constants import MODEL_LABELS
from ntl.metrics import evaluate

A, B, C = MODEL_LABELS[:3]


def test_report_matches_sklearn_and_preserves_absent_classes():
    truth = [A, B, A, B]
    predicted = [A, B, B, B]
    report = evaluate(truth, predicted)
    assert report["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(truth, predicted)
    )
    assert report["per_class_recall"][A] == pytest.approx(0.5)
    assert report["per_class_recall"][B] == pytest.approx(1.0)
    assert report["per_class_recall"][C] is None
    assert report["support"][C] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("truth,predicted", [([], []), ([A], []), ([A], ["unknown"])])
def test_invalid_metric_inputs_are_rejected(truth, predicted):
    with pytest.raises(ValueError):
        evaluate(truth, predicted)
