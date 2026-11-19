#!/usr/bin/env python3
"""Evaluation script and fixture for Atomic Claim Extraction.

Evaluates claim decomposition across representative response types:
1. Compound factual statements (Albert Einstein benchmark)
2. Pure subjective opinions
3. Mixed factual, opinion, and procedural instructions
4. Negations and temporal/date constraints
5. Short responses and questions
6. Real RAGTruth generated responses

Inspects atomicity, coverage, faithfulness, and classification distribution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.cache import ClaimExtractionCache
from src.extraction.extractor import AtomicClaimExtractor
from src.extraction.providers.rule_based_provider import RuleBasedExtractorProvider
from src.extraction.schema import ClaimType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_claim_extraction")

# Manually inspectable representative test fixtures
REPRESENTATIVE_FIXTURES = [
    {
        "fixture_id": "F1_COMPOUND_FACTUAL",
        "description": "Canonical compound factual statement with entities, dates, and awards",
        "response_text": "Albert Einstein was born in Germany in 1879 and received the 1921 Nobel Prize for relativity.",
        "expected_claim_types": ["FACTUAL"],
        "min_atomic_claims": 3,
    },
    {
        "fixture_id": "F2_PURE_OPINION",
        "description": "Subjective aesthetic commentary without empirical facts",
        "response_text": "In my opinion, this restaurant serves the most amazing and delicious pasta in the entire city.",
        "expected_claim_types": ["OPINION"],
        "min_atomic_claims": 1,
    },
    {
        "fixture_id": "F3_MIXED_FACTUAL_OPINION_INSTRUCTION",
        "description": "Combination of factual history, personal opinion, and user instruction",
        "response_text": "Subway was founded in 1965 in Bridgeport, Connecticut. The sandwiches are wonderful. Please check your local store hours before visiting!",
        "expected_claim_types": ["FACTUAL", "OPINION", "INSTRUCTION"],
        "min_atomic_claims": 3,
    },
    {
        "fixture_id": "F4_NEGATION_AND_UNCERTAINTY",
        "description": "Negative factual assertion and temporal boundaries",
        "response_text": "The Palestinian Authority did not join the ICC before 2015 and has never held formal state party status in the UN General Assembly.",
        "expected_claim_types": ["FACTUAL"],
        "min_atomic_claims": 2,
    },
    {
        "fixture_id": "F5_CONVERSATIONAL_QUESTION_FILLER",
        "description": "Interrogative question followed by non-verifiable conversational sign-off",
        "response_text": "What is the procedure for preparing beet greens? Hope this helps! Have a great day!",
        "expected_claim_types": ["QUESTION", "NON_VERIFIABLE"],
        "min_atomic_claims": 2,
    },
    {
        "fixture_id": "F6_VERY_SHORT_ANSWER",
        "description": "Minimal single-clause answer",
        "response_text": "Paris is the capital of France.",
        "expected_claim_types": ["FACTUAL"],
        "min_atomic_claims": 1,
    },
]


def evaluate_fixture(
    extractor: AtomicClaimExtractor,
    fixture: dict,
) -> dict:
    """Run extraction on fixture and inspect qualitative dimensions."""
    text = fixture["response_text"]
    result = extractor.extract_claims(text, sample_id=fixture["fixture_id"])

    extracted_types = [c.claim_type.value for c in result.claims]
    factual_claims = [c.text for c in result.get_factual_claims()]
    excluded_claims = [f"[{c.claim_type.value}] {c.text}" for c in result.get_excluded_claims()]

    # Qualitative heuristics for manual inspection
    atomicity_pass = len(result.claims) >= fixture["min_atomic_claims"]
    has_expected_types = any(t in extracted_types for t in fixture["expected_claim_types"])

    return {
        "fixture_id": fixture["fixture_id"],
        "description": fixture["description"],
        "input_text": text,
        "total_claims": result.total_claims,
        "factual_claims": factual_claims,
        "excluded_claims": excluded_claims,
        "atomicity_flag": "PASS" if atomicity_pass else "NEEDS_SPLITTING",
        "type_alignment_flag": "PASS" if has_expected_types else "REVIEW",
        "latency_ms": result.latency_ms,
        "cached": result.cached,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Atomic Claim Extraction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory to save inspection report (default: data/evaluation)",
    )
    parser.add_argument(
        "--include-ragtruth",
        action="store_true",
        help="Include real RAGTruth test responses in evaluation",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = ClaimExtractionCache(Path("data/cache/extraction"))
    provider = RuleBasedExtractorProvider()
    extractor = AtomicClaimExtractor(provider=provider, cache=cache)

    logger.info("Evaluating representative claim extraction fixtures...")
    eval_results = []
    for fix in REPRESENTATIVE_FIXTURES:
        report = evaluate_fixture(extractor, fix)
        eval_results.append(report)

    # Optionally evaluate on real RAGTruth test samples
    ragtruth_results = []
    if args.include_ragtruth:
        test_path = Path("data/splits/official/test.jsonl")
        if test_path.exists():
            logger.info("Evaluating on real RAGTruth test responses...")
            with open(test_path) as f:
                for idx, line in enumerate(f):
                    if idx >= 5:
                        break
                    data = json.loads(line)
                    res = extractor.extract_claims(data["generated_response"], sample_id=data["sample_id"])
                    ragtruth_results.append({
                        "sample_id": data["sample_id"],
                        "task_type": data["task_type"],
                        "total_claims": res.total_claims,
                        "factual_claims_count": res.factual_claims_count,
                        "excluded_claims_count": res.excluded_claims_count,
                        "sample_claims": [c.text for c in res.claims[:3]],
                    })

    full_report = {
        "extractor_model": f"{provider.name}:{provider.model}",
        "fixtures_evaluated": len(eval_results),
        "fixtures": eval_results,
        "ragtruth_samples": ragtruth_results,
    }

    report_path = args.output_dir / "claim_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("=" * 65)
    logger.info("CLAIM EXTRACTION INSPECTION REPORT")
    logger.info("=" * 65)
    for res in eval_results:
        logger.info("[%s] %s", res["fixture_id"], res["description"])
        logger.info("  Input: %s", res["input_text"])
        logger.info("  Total Claims: %d (Factual: %d, Excluded: %d)", res["total_claims"], len(res["factual_claims"]), len(res["excluded_claims"]))
        for fc in res["factual_claims"]:
            logger.info("    -> [FACTUAL] %s", fc)
        for ec in res["excluded_claims"]:
            logger.info("    -> %s", ec)
        logger.info("  Atomicity: %s | Alignment: %s", res["atomicity_flag"], res["type_alignment_flag"])
        logger.info("-" * 65)

    logger.info("Saved full inspection report to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
