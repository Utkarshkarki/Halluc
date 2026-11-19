"""Dataset adapter for NLI premise-hypothesis pairs.

Converts Phase 1 NormalizedSample objects into premise-hypothesis pairs for
the NLI model, preserving ground-truth traceability and documenting the
formulation characteristics and limitations.

NLI Formulation:
    PREMISE    = reference_context (trusted evidence)
    HYPOTHESIS = generated_response (LLM output to verify)

Taxonomy Mapping:
    ENTAILMENT    -> SUPPORTED
    CONTRADICTION -> CONTRADICTED
    NEUTRAL       -> INSUFFICIENT_EVIDENCE

Important Formulation Analysis & Limitations:
    1. Response-level vs. Claim-level:
       In Phase 1 & 2, the ground-truth annotations in RAGTruth are span-level,
       aggregated into a response-level label:
         - SUPPORTED: No hallucinated spans.
         - CONTRADICTED: At least one span is an Evident/Subtle Conflict.
         - INSUFFICIENT_EVIDENCE: Only Baseless Info spans (unsupported external info).

       When feeding the entire response as HYPOTHESIS to a sentence-pair NLI model:
       - Cross-encoders trained on MNLI/FEVER/ANLI expect short hypothesis pairs.
       - A response containing 3 supported sentences and 1 minor baseless detail
         presents a conjunction: P |= (S1 and S2 and S3 and S4).
       - In formal logic, if any conjunct is ungrounded, the conjunction is Neutral.
         However, neural NLI models often predict Entailment if the majority of
         tokens are supported, leading to false negatives on subtle hallucinated spans.
       - Full atomic claim extraction and per-claim verification is scheduled for Phase 3.

    2. Neutral vs. Hallucination:
       NEUTRAL indicates insufficient evidence in the reference context. It does NOT
       mean the statement is factually false in the real world (e.g., implicit_true
       cases in RAGTruth where external facts are accurate but unmentioned in context).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.data.schema import HallucinationLabel, NormalizedSample


@dataclass
class NLIPair:
    """A premise-hypothesis pair ready for NLI inference."""

    sample_id: str
    source_id: str
    premise: str
    hypothesis: str
    ground_truth_label: HallucinationLabel
    task_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetNLIAdapter:
    """Adapts NormalizedSample records into NLI premise-hypothesis pairs."""

    @staticmethod
    def sample_to_pair(sample: NormalizedSample) -> NLIPair:
        """Convert a single NormalizedSample to a response-level NLIPair.

        Premise = sample.reference_context
        Hypothesis = sample.generated_response
        """
        return NLIPair(
            sample_id=sample.sample_id,
            source_id=sample.source_id,
            premise=sample.reference_context.strip(),
            hypothesis=sample.generated_response.strip(),
            ground_truth_label=sample.hallucination_label,
            task_type=sample.task_type,
            metadata={
                "model": sample.metadata.get("model", "unknown"),
                "quality": sample.metadata.get("quality", "unknown"),
                "span_count": len(sample.hallucinated_spans),
                "has_conflict": any(
                    "conflict" in s.label_type.lower() for s in sample.hallucinated_spans
                ),
            },
        )

    @classmethod
    def adapt_dataset(cls, samples: list[NormalizedSample]) -> list[NLIPair]:
        """Convert a list of NormalizedSample objects into NLIPairs.

        Filters out empty premise or empty hypothesis records.
        """
        pairs: list[NLIPair] = []
        for s in samples:
            pair = cls.sample_to_pair(s)
            if pair.premise and pair.hypothesis:
                pairs.append(pair)
        return pairs
