"""Unit tests for evaluation metrics and evaluator contracts."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score as sk_f1, precision_score as sk_prec, recall_score as sk_rec, roc_auc_score

from src.data.schema import (
    DetectorPrediction,
    EvaluationResult,
    HallucinationLabel,
    NormalizedSample,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    accuracy,
    auprc,
    auroc,
    confusion_matrix,
    f1_score,
    per_class_metrics,
    precision,
    recall,
)


def test_basic_metrics_correctness():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

    assert accuracy(y_true, y_pred) == accuracy_score(y_true, y_pred)
    assert precision(y_true, y_pred, positive_label=1) == pytest.approx(
        sk_prec(y_true, y_pred, pos_label=1)
    )
    assert recall(y_true, y_pred, positive_label=1) == pytest.approx(
        sk_rec(y_true, y_pred, pos_label=1)
    )
    assert f1_score(y_true, y_pred, positive_label=1) == pytest.approx(
        sk_f1(y_true, y_pred, pos_label=1)
    )


def test_metrics_zero_division():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    # No positives predicted or true
    assert precision(y_true, y_pred, positive_label=1) == 0.0
    assert recall(y_true, y_pred, positive_label=1) == 0.0
    assert f1_score(y_true, y_pred, positive_label=1) == 0.0


def test_confusion_matrix():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert cm.shape == (3, 3)
    # class 0: both predicted 0
    assert cm[0, 0] == 2
    # class 1: 1 pred 1, 1 pred 2
    assert cm[1, 1] == 1
    assert cm[1, 2] == 1
    # class 2: 1 pred 1, 1 pred 2
    assert cm[2, 1] == 1
    assert cm[2, 2] == 1


def test_auroc_and_auprc():
    y_true = [1, 0, 1, 0, 1, 0, 0, 1]
    y_scores = [0.9, 0.1, 0.8, 0.4, 0.7, 0.2, 0.3, 0.85]

    auc_score = auroc(y_true, y_scores)
    expected_auc = roc_auc_score(y_true, y_scores)
    assert auc_score == pytest.approx(expected_auc, rel=1e-3)

    pr_score = auprc(y_true, y_scores)
    assert 0.0 <= pr_score <= 1.0
    assert pr_score > 0.8  # Well separated scores should have high PR area


def test_evaluator_multiclass_and_binary():
    gt_samples = [
        NormalizedSample(
            sample_id="s1",
            source_id="src1",
            task_type="QA",
            question_or_prompt="Q1",
            reference_context="C1",
            generated_response="R1",
            hallucination_label=HallucinationLabel.SUPPORTED,
            split="test",
        ),
        NormalizedSample(
            sample_id="s2",
            source_id="src2",
            task_type="QA",
            question_or_prompt="Q2",
            reference_context="C2",
            generated_response="R2",
            hallucination_label=HallucinationLabel.CONTRADICTED,
            split="test",
        ),
        NormalizedSample(
            sample_id="s3",
            source_id="src3",
            task_type="QA",
            question_or_prompt="Q3",
            reference_context="C3",
            generated_response="R3",
            hallucination_label=HallucinationLabel.INSUFFICIENT_EVIDENCE,
            split="test",
        ),
    ]

    preds = [
        DetectorPrediction(
            sample_id="s1",
            predicted_label=HallucinationLabel.SUPPORTED,
            confidence=0.1,
        ),
        DetectorPrediction(
            sample_id="s2",
            predicted_label=HallucinationLabel.CONTRADICTED,
            confidence=0.9,
        ),
        DetectorPrediction(
            sample_id="s3",
            predicted_label=HallucinationLabel.CONTRADICTED,  # Mistake (predicted CONTRADICTED instead of INSUFFICIENT_EVIDENCE)
            confidence=0.7,
        ),
    ]

    evaluator = Evaluator()

    # Multi-class evaluation
    res_multi = evaluator.evaluate(preds, gt_samples)
    assert isinstance(res_multi, EvaluationResult)
    assert res_multi.sample_count == 3
    assert res_multi.accuracy == pytest.approx(2 / 3)
    assert "SUPPORTED" in res_multi.per_class
    assert "CONTRADICTED" in res_multi.per_class

    # Binary evaluation (s1 is NON_HALLUCINATED; s2, s3 are HALLUCINATED)
    # s1 pred SUPPORTED (0 -> 0): correct
    # s2 pred CONTRADICTED (1 -> 1): correct
    # s3 pred CONTRADICTED (2 -> 1): in binary, 2 is hallucinated, 1 is hallucinated -> correct!
    res_bin = evaluator.evaluate_binary(preds, gt_samples)
    assert res_bin.accuracy == 1.0  # All binary hallucination states match
    assert res_bin.auroc_score is not None
    assert res_bin.auprc_score is not None
    assert res_bin.auroc_score == 1.0


def test_evaluator_mismatched_ids_raises():
    gt = [
        NormalizedSample(
            sample_id="s1",
            source_id="src1",
            task_type="QA",
            question_or_prompt="Q1",
            reference_context="C1",
            generated_response="R1",
            hallucination_label=HallucinationLabel.SUPPORTED,
            split="test",
        )
    ]
    preds = [
        DetectorPrediction(
            sample_id="different_id",
            predicted_label=HallucinationLabel.SUPPORTED,
        )
    ]
    evaluator = Evaluator()
    with pytest.raises(ValueError) as exc:
        evaluator.evaluate(preds, gt)
    assert "Sample ID mismatch" in str(exc.value)
