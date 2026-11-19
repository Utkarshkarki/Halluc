#!/usr/bin/env python3
"""Run and evaluate the pretrained NLI baseline detector on RAGTruth.

Loads MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, performs batched inference,
evaluates using the Phase 1 Evaluator, records throughput/latency, and generates
an error analysis covering false positives, false negatives, and high-confidence errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import DetectorPrediction, HallucinationLabel, NormalizedSample
from src.data.splitter import load_splits
from src.detectors.nli_detector import NLIDetector
from src.detectors.schema import NLIConfig, NLIOutput
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_nli_baseline")


def load_samples_from_jsonl(path: Path) -> list[NormalizedSample]:
    """Load NormalizedSample objects from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(NormalizedSample.model_validate_json(line))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pretrained NLI baseline evaluation on RAGTruth"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/splits/official/test.jsonl"),
        help="Path to evaluation split JSONL (default: data/splits/official/test.jsonl)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("configs/nli_baseline.json"),
        help="Path to NLI config JSON (default: configs/nli_baseline.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory to save evaluation results and error analysis",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate (e.g. for quick benchmarking)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, or auto)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Configuration
    if args.config_file.exists():
        with open(args.config_file) as f:
            cfg_dict = json.load(f)
        cfg = NLIConfig(**cfg_dict)
    else:
        cfg = NLIConfig()

    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device

    # 2. Load Evaluation Data
    logger.info("Loading evaluation samples from %s...", args.test_file)
    samples = load_samples_from_jsonl(args.test_file)
    if args.limit:
        samples = samples[: args.limit]
    logger.info("Loaded %d evaluation samples", len(samples))

    # 3. Load Model
    logger.info("Initializing NLIDetector (%s)...", cfg.model_name)
    detector = NLIDetector.from_pretrained(
        model_name=cfg.model_name,
        revision=cfg.revision,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        fp16=cfg.fp16,
        seed=cfg.seed,
    )

    # 4. Run Batched Inference
    logger.info("Running NLI inference on %d samples (batch_size=%d)...", len(samples), cfg.batch_size)
    pairs = [(s.reference_context, s.generated_response) for s in samples]

    start_time = time.perf_counter()
    nli_outputs = detector.predict_pairs(pairs, batch_size=cfg.batch_size)
    total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / len(samples)) * 1000.0 if samples else 0.0
    throughput_qps = len(samples) / total_time if total_time > 0 else 0.0

    logger.info("Inference completed in %.2fs (%.2f ms/sample, %.2f samples/sec)", total_time, avg_latency_ms, throughput_qps)

    # 5. Formulate Predictions for Evaluator
    predictions: list[DetectorPrediction] = []
    for s, out in zip(samples, nli_outputs):
        # In binary hallucination detection: confidence in hallucination is P(CONTRADICTED) + P(INSUFFICIENT_EVIDENCE)
        hallucination_confidence = 1.0 - out.probabilities.supported
        predictions.append(
            DetectorPrediction(
                sample_id=s.sample_id,
                predicted_label=out.label,
                confidence=hallucination_confidence,
                detector_name=detector.name,
            )
        )

    # 6. Evaluate via Phase 1 Evaluator
    evaluator = Evaluator()

    # Multi-class 3-way evaluation
    multi_res = evaluator.evaluate(predictions, samples)

    # Binary evaluation (Non-hallucinated vs Hallucinated)
    bin_res = evaluator.evaluate_binary(predictions, samples)

    # 7. Error Analysis
    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    high_conf_errors: list[dict] = []
    neutral_cases: list[dict] = []

    for s, pred, out in zip(samples, predictions, nli_outputs):
        is_true_hallucinated = s.is_hallucinated()
        is_pred_hallucinated = pred.predicted_label != HallucinationLabel.SUPPORTED

        error_item = {
            "sample_id": s.sample_id,
            "source_id": s.source_id,
            "task_type": s.task_type,
            "model_evaluated": s.metadata.get("model", "unknown"),
            "ground_truth_label": s.hallucination_label.value,
            "predicted_label": pred.predicted_label.value,
            "confidence": round(out.confidence, 4),
            "probabilities": {
                "supported": round(out.probabilities.supported, 4),
                "insufficient_evidence": round(out.probabilities.insufficient_evidence, 4),
                "contradicted": round(out.probabilities.contradicted, 4),
            },
            "truncated": out.truncated,
            "premise_snippet": s.reference_context[:200] + ("..." if len(s.reference_context) > 200 else ""),
            "hypothesis": s.generated_response,
            "hallucinated_spans": [span.model_dump() for span in s.hallucinated_spans],
        }

        # False Positive: Truth is SUPPORTED, but predicted HALLUCINATED
        if not is_true_hallucinated and is_pred_hallucinated:
            false_positives.append(error_item)

        # False Negative: Truth is HALLUCINATED, but predicted SUPPORTED
        if is_true_hallucinated and not is_pred_hallucinated:
            false_negatives.append(error_item)

        # High confidence error: confidence > 0.85 and mismatch
        if pred.predicted_label != s.hallucination_label and out.confidence >= 0.85:
            high_conf_errors.append(error_item)

        # Neutral cases
        if pred.predicted_label == HallucinationLabel.INSUFFICIENT_EVIDENCE:
            neutral_cases.append(error_item)

    error_analysis_data = {
        "summary": {
            "total_evaluated": len(samples),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "high_conf_errors_count": len(high_conf_errors),
            "neutral_cases_count": len(neutral_cases),
        },
        "representative_false_positives": false_positives[:5],
        "representative_false_negatives": false_negatives[:5],
        "representative_high_conf_errors": high_conf_errors[:5],
        "representative_neutral_cases": neutral_cases[:5],
    }

    # Save outputs
    error_analysis_path = output_dir / "error_analysis.json"
    with open(error_analysis_path, "w") as f:
        json.dump(error_analysis_data, f, indent=2)

    # Save full metrics
    metrics_summary = {
        "model": cfg.model_name,
        "revision": cfg.revision or "default",
        "device": str(detector.device),
        "total_evaluated": len(samples),
        "performance": {
            "total_time_seconds": round(total_time, 2),
            "average_latency_ms": round(avg_latency_ms, 2),
            "throughput_samples_per_sec": round(throughput_qps, 2),
        },
        "multi_class": {
            "accuracy": round(multi_res.accuracy, 4),
            "per_class": multi_res.per_class,
            "confusion_matrix": multi_res.confusion_mat.tolist(),
        },
        "binary_hallucination": {
            "accuracy": round(bin_res.accuracy, 4),
            "per_class": bin_res.per_class,
            "confusion_matrix": bin_res.confusion_mat.tolist(),
            "auroc": round(bin_res.auroc_score, 4) if bin_res.auroc_score is not None else None,
            "auprc": round(bin_res.auprc_score, 4) if bin_res.auprc_score is not None else None,
        },
        "truncation": {
            "truncated_count": sum(1 for o in nli_outputs if o.truncated),
            "truncated_ratio": round(sum(1 for o in nli_outputs if o.truncated) / len(samples), 4) if samples else 0.0,
        },
    }

    metrics_path = output_dir / "nli_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Saved error analysis to %s", error_analysis_path)
    logger.info("Saved metrics to %s", metrics_path)

    # Print summary
    logger.info("=" * 65)
    logger.info("NLI BASELINE EVALUATION REPORT")
    logger.info("=" * 65)
    logger.info("Model: %s | Device: %s", cfg.model_name, detector.device)
    logger.info("Evaluated: %d samples", len(samples))
    logger.info("Latency: %.2f ms/sample | Throughput: %.2f samples/sec", avg_latency_ms, throughput_qps)
    logger.info("--- Multi-Class (3-Way) ---")
    logger.info("Accuracy: %.4f", multi_res.accuracy)
    for cname, vals in multi_res.per_class.items():
        logger.info("  %s: P=%.4f R=%.4f F1=%.4f (support=%d)", cname, vals["precision"], vals["recall"], vals["f1"], vals["support"])
    logger.info("Confusion Matrix:\n%s", multi_res.confusion_mat)
    logger.info("--- Binary Hallucination Detection ---")
    logger.info("Accuracy: %.4f", bin_res.accuracy)
    for cname, vals in bin_res.per_class.items():
        logger.info("  %s: P=%.4f R=%.4f F1=%.4f (support=%d)", cname, vals["precision"], vals["recall"], vals["f1"], vals["support"])
    logger.info("AUROC: %s", f"{bin_res.auroc_score:.4f}" if bin_res.auroc_score else "N/A")
    logger.info("AUPRC: %s", f"{bin_res.auprc_score:.4f}" if bin_res.auprc_score else "N/A")
    logger.info("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
