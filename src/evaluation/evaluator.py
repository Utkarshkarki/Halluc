"""Evaluator pipeline for hallucination detection.

Compares detector predictions against ground-truth samples, computing
accuracy, per-class metrics, confusion matrices, and optional AUROC/AUPRC scores.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Sequence

import numpy as np

from src.data.schema import (
    DetectorPrediction,
    EvaluationResult,
    HallucinationLabel,
    NormalizedSample,
)
from src.evaluation.metrics import (
    accuracy,
    auprc,
    auroc,
    confusion_matrix,
    per_class_metrics,
)

logger = logging.getLogger(__name__)

LABEL_TO_INT: dict[HallucinationLabel, int] = {
    HallucinationLabel.SUPPORTED: 0,
    HallucinationLabel.CONTRADICTED: 1,
    HallucinationLabel.INSUFFICIENT_EVIDENCE: 2,
}

DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "SUPPORTED",
    1: "CONTRADICTED",
    2: "INSUFFICIENT_EVIDENCE",
}

BINARY_CLASS_NAMES: dict[int, str] = {
    0: "NON_HALLUCINATED",
    1: "HALLUCINATED",
}


class Evaluator:
    """Evaluates detector predictions against ground truth.

    Usage::

        evaluator = Evaluator()
        result = evaluator.evaluate(predictions, ground_truth_samples)
        print(result.summary())

        # For binary evaluation (hallucinated vs non-hallucinated):
        binary_result = evaluator.evaluate_binary(predictions, ground_truth_samples)
        print(binary_result.summary())
    """

    def __init__(
        self,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def evaluate(
        self,
        predictions: Sequence[DetectorPrediction],
        ground_truth: Sequence[NormalizedSample],
    ) -> EvaluationResult:
        """Evaluate multi-class hallucination predictions against ground truth.

        Args:
            predictions: Detector predictions.
            ground_truth: Ground truth samples.

        Returns:
            EvaluationResult containing accuracy, per-class P/R/F1, and confusion matrix.

        Raises:
            ValueError: If sample IDs do not match or lists are empty.
        """
        if not predictions or not ground_truth:
            raise ValueError("Predictions and ground_truth must be non-empty")

        gt_map = {s.sample_id: s for s in ground_truth}
        pred_map = {p.sample_id: p for p in predictions}

        missing_in_gt = set(pred_map.keys()) - set(gt_map.keys())
        missing_in_pred = set(gt_map.keys()) - set(pred_map.keys())

        if missing_in_gt or missing_in_pred:
            raise ValueError(
                f"Sample ID mismatch! Missing in GT: {len(missing_in_gt)}, "
                f"Missing in predictions: {len(missing_in_pred)}"
            )

        # Align ordered sample IDs
        sample_ids = sorted(gt_map.keys())
        y_true = [LABEL_TO_INT[gt_map[sid].hallucination_label] for sid in sample_ids]
        y_pred = [LABEL_TO_INT[pred_map[sid].predicted_label] for sid in sample_ids]

        acc = accuracy(y_true, y_pred)
        per_class = per_class_metrics(y_true, y_pred, class_names=self.class_names)
        labels_order = sorted(self.class_names.keys())
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        class_dist = dict(Counter(self.class_names[c] for c in y_true))

        return EvaluationResult(
            accuracy=acc,
            per_class=per_class,
            confusion_mat=cm,
            sample_count=len(sample_ids),
            class_distribution=class_dist,
        )

    def evaluate_binary(
        self,
        predictions: Sequence[DetectorPrediction],
        ground_truth: Sequence[NormalizedSample],
    ) -> EvaluationResult:
        """Evaluate binary hallucination detection (non-hallucinated vs. hallucinated).

        Classes:
            0: SUPPORTED
            1: HALLUCINATED (CONTRADICTED or INSUFFICIENT_EVIDENCE)

        If confidence scores are non-zero, AUROC and AUPRC will also be computed.

        Args:
            predictions: Detector predictions.
            ground_truth: Ground truth samples.

        Returns:
            EvaluationResult with binary metrics and optional AUROC/AUPRC.
        """
        if not predictions or not ground_truth:
            raise ValueError("Predictions and ground_truth must be non-empty")

        gt_map = {s.sample_id: s for s in ground_truth}
        pred_map = {p.sample_id: p for p in predictions}

        missing_in_gt = set(pred_map.keys()) - set(gt_map.keys())
        missing_in_pred = set(gt_map.keys()) - set(pred_map.keys())
        if missing_in_gt or missing_in_pred:
            raise ValueError(
                f"Sample ID mismatch! Missing in GT: {len(missing_in_gt)}, "
                f"Missing in predictions: {len(missing_in_pred)}"
            )

        sample_ids = sorted(gt_map.keys())
        # Binary: 0 = SUPPORTED, 1 = ANY HALLUCINATION
        y_true = [
            0 if gt_map[sid].hallucination_label == HallucinationLabel.SUPPORTED else 1
            for sid in sample_ids
        ]
        y_pred = [
            0 if pred_map[sid].predicted_label == HallucinationLabel.SUPPORTED else 1
            for sid in sample_ids
        ]
        confidences = [pred_map[sid].confidence for sid in sample_ids]

        acc = accuracy(y_true, y_pred)
        per_class = per_class_metrics(y_true, y_pred, class_names=BINARY_CLASS_NAMES)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        auroc_score: float | None = None
        auprc_score: float | None = None

        # Compute AUROC and AUPRC if confidence scores are provided (not all zeros)
        if any(c > 0.0 for c in confidences) and len(set(y_true)) > 1:
            try:
                auroc_score = auroc(y_true, confidences, pos_label=1)
                auprc_score = auprc(y_true, confidences, pos_label=1)
            except Exception as e:
                logger.warning("Could not calculate AUROC/AUPRC: %s", e)

        class_dist = dict(Counter(BINARY_CLASS_NAMES[c] for c in y_true))

        return EvaluationResult(
            accuracy=acc,
            per_class=per_class,
            confusion_mat=cm,
            auroc_score=auroc_score,
            auprc_score=auprc_score,
            sample_count=len(sample_ids),
            class_distribution=class_dist,
        )
