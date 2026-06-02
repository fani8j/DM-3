"""Metrics computation for HAR validation.

Provides per-fold metric computation (Macro F1, per-class F1/precision/recall,
confusion matrix) and cross-fold aggregation (mean ± std of Macro F1).

Requirements: R7.6, R7.7
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from har.data.types import FoldMetrics


def compute_fold_metrics(y_true: list[int], y_pred: list[int]) -> FoldMetrics:
    """Compute all metrics for one validation fold.

    Uses sklearn with labels=[0,1,2,3,4,5] so absent classes get explicit zeros.

    Returns FoldMetrics with:
    - macro_f1: unweighted mean of per-class F1
    - per_class_f1: list of 6 F1 scores
    - per_class_precision: list of 6 precision scores
    - per_class_recall: list of 6 recall scores
    - confusion_matrix: 6x6 as nested lists
    """
    labels = [0, 1, 2, 3, 4, 5]
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return FoldMetrics(
        macro_f1=float(macro_f1),
        per_class_f1=f1.tolist(),
        per_class_precision=precision.tolist(),
        per_class_recall=recall.tolist(),
        confusion_matrix=cm.tolist(),
    )


def aggregate_fold_metrics(fold_metrics_list: list[FoldMetrics]) -> tuple[float, float]:
    """Aggregate Macro F1 across K folds: return (mean, std).

    R7.7: When all GroupKFold folds have completed evaluation, report the
    mean and standard deviation of Macro F1 across folds.
    """
    f1_scores = [fm.macro_f1 for fm in fold_metrics_list]
    return float(np.mean(f1_scores)), float(np.std(f1_scores))
