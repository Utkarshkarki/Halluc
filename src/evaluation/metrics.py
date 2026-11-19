"""Evaluation metrics for hallucination detection.

Provides pure, standalone metric functions for binary and multi-class
evaluation of hallucination detection models.

Taxonomy mapping:
    Multi-class (3 classes):
        0: SUPPORTED
        1: CONTRADICTED
        2: INSUFFICIENT_EVIDENCE
    Binary (2 classes):
        0: SUPPORTED (non-hallucinated)
        1: HALLUCINATED (CONTRADICTED or INSUFFICIENT_EVIDENCE)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy(y_true: list[int] | NDArray, y_pred: list[int] | NDArray) -> float:
    """Compute overall classification accuracy.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Fraction of correctly classified samples in [0.0, 1.0].

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Cannot compute accuracy on empty inputs")
    return float(np.mean(yt == yp))


def precision(
    y_true: list[int] | NDArray,
    y_pred: list[int] | NDArray,
    positive_label: int = 1,
) -> float:
    """Compute precision for a specific class label.

    Precision = TP / (TP + FP). Returns 0.0 if (TP + FP) == 0.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        positive_label: The class label to treat as positive.

    Returns:
        Precision score in [0.0, 1.0].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Cannot compute precision on empty inputs")

    tp = int(np.sum((yt == positive_label) & (yp == positive_label)))
    fp = int(np.sum((yt != positive_label) & (yp == positive_label)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(
    y_true: list[int] | NDArray,
    y_pred: list[int] | NDArray,
    positive_label: int = 1,
) -> float:
    """Compute recall (sensitivity) for a specific class label.

    Recall = TP / (TP + FN). Returns 0.0 if (TP + FN) == 0.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        positive_label: The class label to treat as positive.

    Returns:
        Recall score in [0.0, 1.0].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Cannot compute recall on empty inputs")

    tp = int(np.sum((yt == positive_label) & (yp == positive_label)))
    fn = int(np.sum((yt == positive_label) & (yp != positive_label)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(
    y_true: list[int] | NDArray,
    y_pred: list[int] | NDArray,
    positive_label: int = 1,
) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    F1 = 2 * P * R / (P + R). Returns 0.0 if (P + R) == 0.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        positive_label: The class label to treat as positive.

    Returns:
        F1 score in [0.0, 1.0].
    """
    p = precision(y_true, y_pred, positive_label=positive_label)
    r = recall(y_true, y_pred, positive_label=positive_label)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0


def confusion_matrix(
    y_true: list[int] | NDArray,
    y_pred: list[int] | NDArray,
    labels: list[int] | None = None,
) -> NDArray[np.int64]:
    """Compute NxN confusion matrix.

    Row i represents ground-truth class i, column j represents predicted class j:
        C[i, j] = count of instances with true label labels[i] and predicted labels[j].

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        labels: Optional explicit list of class labels in desired order.
                If None, unique labels from y_true and y_pred are used.

    Returns:
        2D numpy array of shape (N, N).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Cannot compute confusion matrix on empty inputs")

    if labels is None:
        unique_labels = sorted(set(yt.tolist()).union(set(yp.tolist())))
    else:
        unique_labels = list(labels)

    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    n_classes = len(unique_labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for t, p in zip(yt, yp):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


def auroc(
    y_true: list[int] | NDArray,
    y_scores: list[float] | NDArray,
    pos_label: int = 1,
) -> float:
    """Compute Area Under the Receiver Operating Characteristic (ROC) curve.

    Applicable for binary evaluation with continuous predicted risk/confidence scores.
    Computed via trapezoidal integration over sorted distinct score thresholds.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_scores: Continuous confidence scores (higher indicates more likely positive).
        pos_label: The label value representing the positive class.

    Returns:
        AUROC score in [0.0, 1.0].

    Raises:
        ValueError: If y_true does not contain both positive and negative examples.
    """
    yt = (np.asarray(y_true) == pos_label).astype(int)
    scores = np.asarray(y_scores, dtype=float)

    if len(yt) != len(scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_scores)={len(scores)}")

    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUROC requires both positive and negative samples. Got {n_pos} positive, {n_neg} negative."
        )

    # Sort by descending score
    desc_order = np.argsort(-scores, kind="mergesort")
    yt_sorted = yt[desc_order]
    scores_sorted = scores[desc_order]

    # Find distinct thresholds
    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, yt_sorted.size - 1]

    # Cumulative true positives and false positives
    tps = np.cumsum(yt_sorted)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps

    # Add origin (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / n_neg
    tpr = tps / n_pos

    # Trapezoidal rule: sum (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    auc_val = float(np.trapz(tpr, fpr)) if hasattr(np, "trapz") else float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return float(np.clip(auc_val, 0.0, 1.0))


def auprc(
    y_true: list[int] | NDArray,
    y_scores: list[float] | NDArray,
    pos_label: int = 1,
) -> float:
    """Compute Area Under the Precision-Recall curve (AUPRC / Average Precision).

    Applicable for binary evaluation with continuous predicted risk/confidence scores.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Continuous confidence scores.
        pos_label: The label value representing the positive class.

    Returns:
        AUPRC score in [0.0, 1.0].

    Raises:
        ValueError: If y_true has no positive instances or lengths mismatch.
    """
    yt = (np.asarray(y_true) == pos_label).astype(int)
    scores = np.asarray(y_scores, dtype=float)

    if len(yt) != len(scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_scores)={len(scores)}")

    n_pos = int(np.sum(yt == 1))
    if n_pos == 0:
        raise ValueError("AUPRC requires at least one positive sample")

    desc_order = np.argsort(-scores, kind="mergesort")
    yt_sorted = yt[desc_order]
    scores_sorted = scores[desc_order]

    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, yt_sorted.size - 1]

    tps = np.cumsum(yt_sorted)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps

    recalls = tps / n_pos
    precisions = tps / (tps + fps)

    # Prepend recall=0, precision=1 for standard PR curve starting point
    recalls = np.r_[0.0, recalls]
    precisions = np.r_[1.0, precisions]

    # Calculate trapezoidal area
    rec_diff = np.diff(recalls)
    auc_val = float(np.sum(rec_diff * precisions[1:]))
    return float(np.clip(auc_val, 0.0, 1.0))


def per_class_metrics(
    y_true: list[int] | NDArray,
    y_pred: list[int] | NDArray,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, and F1 score for each class.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        class_names: Optional mapping from integer label to human-readable name.

    Returns:
        Dict mapping class name to {'precision': float, 'recall': float, 'f1': float, 'support': int}.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if class_names is None:
        classes = sorted(set(yt.tolist()).union(set(yp.tolist())))
        class_names = {c: str(c) for c in classes}

    results: dict[str, dict[str, float]] = {}
    for code, name in class_names.items():
        p = precision(yt, yp, positive_label=code)
        r = recall(yt, yp, positive_label=code)
        f = f1_score(yt, yp, positive_label=code)
        supp = int(np.sum(yt == code))
        results[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": supp,
        }
    return results
