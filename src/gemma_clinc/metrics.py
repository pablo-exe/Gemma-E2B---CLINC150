"""Benchmark metrics and error analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from gemma_clinc.data import OOS_LABEL

INVALID_LABEL = "__invalid__"


def compute_metrics(y_true: list[str], y_pred: list[str | None]) -> dict[str, Any]:
    """Compute CLINC150 metrics while retaining invalid generations as errors."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and have equal length")

    predictions = [prediction if prediction is not None else INVALID_LABEL for prediction in y_pred]
    labels = sorted(set(y_true))
    in_scope_indexes = [index for index, label in enumerate(y_true) if label != OOS_LABEL]
    in_scope_true = [y_true[index] for index in in_scope_indexes]
    in_scope_pred = [predictions[index] for index in in_scope_indexes]

    oos_true = [label == OOS_LABEL for label in y_true]
    oos_pred = [label == OOS_LABEL for label in predictions]
    oos_precision, oos_recall, oos_f1, _ = precision_recall_fscore_support(
        oos_true, oos_pred, average="binary", zero_division=0
    )

    confusions = Counter(
        (truth, prediction)
        for truth, prediction in zip(y_true, predictions, strict=True)
        if truth != prediction
    )

    return {
        "n_examples": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0),
        "in_scope_accuracy": accuracy_score(in_scope_true, in_scope_pred),
        "oos_precision": oos_precision,
        "oos_recall": oos_recall,
        "oos_f1": oos_f1,
        "invalid_output_rate": predictions.count(INVALID_LABEL) / len(predictions),
        "top_confusions": [
            {"true": truth, "predicted": prediction, "count": count}
            for (truth, prediction), count in confusions.most_common(20)
        ],
    }
