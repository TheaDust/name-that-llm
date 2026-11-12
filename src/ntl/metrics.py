"""Shared metric report for full grouped out-of-fold predictions."""

from collections.abc import Sequence

from sklearn.metrics import balanced_accuracy_score, classification_report

from .constants import MODEL_LABEL_SET, MODEL_LABELS


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Return JSON-friendly balanced accuracy, per-class recall and support.

    Inputs must follow the same row order. Score complete OOF predictions for
    comparisons between routes; classes absent from a diagnostic slice have
    null recall and are excluded from balanced accuracy.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    unknown = (set(y_true) | set(y_pred)) - MODEL_LABEL_SET
    if unknown:
        raise ValueError(f"labels outside the 18 valid models: {sorted(unknown)}")
    report = classification_report(
        y_true, y_pred, labels=list(MODEL_LABELS), output_dict=True, zero_division=0
    )
    return {
        "n": len(y_true),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {
            label: report[label]["recall"] if report[label]["support"] else None
            for label in MODEL_LABELS
        },
        "support": {label: int(report[label]["support"]) for label in MODEL_LABELS},
    }
