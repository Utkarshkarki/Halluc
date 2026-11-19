# src/pipeline/__init__.py
"""Public interface for the hallucination detection pipeline (Phase 5).

Exports the core classes that users are expected to import:
- :class:`HallucinationDetector`
- :class:`ResponseScorer`
- :class:`ResponseScore`
- :class:`EvidenceAggregator`
- :class:`CalibrationModule`
"""

from .aggregator import EvidenceAggregator
from .scorer import ResponseScorer, ResponseScore
from .calibration import CalibrationModule
from .detector import HallucinationDetector

__all__ = [
    "EvidenceAggregator",
    "ResponseScorer",
    "ResponseScore",
    "CalibrationModule",
    "HallucinationDetector",
]
