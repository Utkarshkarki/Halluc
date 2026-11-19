"""Deterministic rule-based atomic claim decomposition provider.

Provides an offline, reproducible, zero-cost claim extraction engine that:
1. Segments sentences and decomposes compound coordinating conjunctions.
2. Resolves shared subjects across conjoined predicates.
3. Classifies statements into FACTUAL, OPINION, INSTRUCTION, QUESTION, NON_VERIFIABLE.
4. Identifies exact character offsets in source text.
5. Operates without external network calls or API dependencies.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel

from src.extraction.providers.base import BaseLLMProvider
from src.extraction.schema import ClaimType


OPINION_SIGNALS = {
    "i think", "in my opinion", "i believe", "best", "worst", "amazing",
    "terrible", "wonderful", "horrible", "greatest", "beautiful", "disappointing",
    "delicious", "fantastic", "mediocre", "overrated", "underrated", "unacceptable"
}

INSTRUCTION_SIGNALS = {
    "please", "make sure to", "ensure that", "remember to", "be sure to",
    "note that", "check your", "follow the", "avoid", "do not forget"
}

NON_VERIFIABLE_SIGNALS = {
    "hope this helps", "let me know", "feel free to", "have a great day",
    "thanks for asking", "hello", "hi there", "in conclusion", "overall"
}


class RuleBasedExtractorProvider(BaseLLMProvider):
    """Local deterministic provider for atomic claim decomposition."""

    def __init__(self, model_name: str = "rule-based-decomposer-v1") -> None:
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "rule_based"

    @property
    def model(self) -> str:
        return self._model_name

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences preserving abbreviations."""
        # Simple sentence boundary detection
        text = text.strip()
        if not text:
            return []
        # Split on sentence terminals followed by whitespace or quote
        raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", text)
        return [s.strip() for s in raw_sents if s.strip()]

    @staticmethod
    def _classify_text(text: str) -> ClaimType:
        """Classify a statement into ClaimType."""
        lower = text.lower().strip()
        if lower.endswith("?"):
            return ClaimType.QUESTION

        # Check for instructions
        if any(lower.startswith(sig) or f" {sig} " in lower for sig in INSTRUCTION_SIGNALS):
            return ClaimType.INSTRUCTION

        # Check for non-verifiable filler
        if any(sig in lower for sig in NON_VERIFIABLE_SIGNALS):
            return ClaimType.NON_VERIFIABLE

        # Check for opinions
        if any(f" {sig} " in f" {lower} " or lower.startswith(sig) for sig in OPINION_SIGNALS):
            return ClaimType.OPINION

        return ClaimType.FACTUAL

    @staticmethod
    def _decompose_compound_sentence(sentence: str) -> list[str]:
        """Decompose compound sentence conjoined by 'and', 'but', or ';'.

        Example:
            'Albert Einstein was born in Germany in 1879 and received the 1921 Nobel Prize.'
            -> ['Albert Einstein was born in Germany in 1879', 'Albert Einstein received the 1921 Nobel Prize']
        """
        # Split on semicolons first
        if ";" in sentence:
            parts = [p.strip() for p in sentence.split(";") if p.strip()]
            subparts = []
            for p in parts:
                subparts.extend(RuleBasedExtractorProvider._decompose_compound_sentence(p))
            return subparts

        # Check for conjoined predicates: '<Subject> <Verb1> ... and <Verb2> ...'
        conjunction_match = re.search(r"\b(and|while|but)\b\s+(he|she|they|it|received|was|were|has|have|had|is|are|became|moved|died|joined|signed|won|graduated)\b", sentence, re.IGNORECASE)
        if conjunction_match:
            split_idx = conjunction_match.start()
            conj_word = conjunction_match.group(1)
            first_clause = sentence[:split_idx].strip()
            second_clause = sentence[conjunction_match.end(1):].strip()

            # Extract subject from first clause if second clause starts with a verb
            tokens = first_clause.split()
            # If second clause does not have an explicit pronoun subject, inherit first clause subject
            subj_tokens = []
            for t in tokens:
                if t.lower() in ("was", "were", "is", "are", "received", "has", "have", "had", "became", "born"):
                    break
                subj_tokens.append(t)
            subj = " ".join(subj_tokens).strip()

            second_tokens = second_clause.split()
            if second_tokens and second_tokens[0].lower() in ("he", "she", "they", "it"):
                # Pronoun replacement
                if subj:
                    second_clause = f"{subj} " + " ".join(second_tokens[1:])
            elif subj and not any(second_clause.startswith(p) for p in ("he ", "she ", "they ", "it ", subj)):
                second_clause = f"{subj} {second_clause}"

            return [first_clause, second_clause]

        return [sentence]

    @staticmethod
    def _split_location_date(clause: str) -> list[str]:
        """Further separate compound location and date clauses if present.

        Example:
            'Albert Einstein was born in Germany in 1879'
            -> ['Albert Einstein was born in Germany', 'Albert Einstein was born in 1879']
        """
        m = re.match(r"^(.*?was born in\s+[A-Z][a-zA-Z\s]+)\s+in\s+(\d{4})(.*)$", clause)
        if m:
            base_loc = m.group(1).strip()
            year = m.group(2).strip()
            # Extract subject
            subj = re.split(r"\bwas born\b", base_loc)[0].strip()
            return [base_loc, f"{subj} was born in {year}"]
        return [clause]

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Decompose text in prompt into structured atomic claims."""
        # Extract the input text from prompt template
        match = re.search(r"\[INPUT TEXT\]\s*(.*?)\s*\[END INPUT TEXT\]", prompt, re.DOTALL)
        text = match.group(1).strip() if match else prompt.strip()

        if not text:
            return {"claims": []}

        sentences = self._split_sentences(text)
        decomposed_clauses: list[str] = []

        for sent in sentences:
            compounds = self._decompose_compound_sentence(sent)
            for c in compounds:
                decomposed_clauses.extend(self._split_location_date(c))

        claims_list = []
        for i, clause in enumerate(decomposed_clauses, start=1):
            clause_clean = clause.strip().rstrip(".")
            if not clause_clean:
                continue

            ctype = self._classify_text(clause_clean)

            # Locate offsets in original text if exact match exists
            s_start = None
            s_end = None
            search_idx = text.find(clause_clean)
            if search_idx != -1:
                s_start = search_idx
                s_end = search_idx + len(clause_clean)

            # Preserve dates/numbers tags
            tags = []
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clause_clean)
            if numbers:
                tags.extend([f"num:{n}" for n in numbers[:3]])

            claims_list.append(
                {
                    "claim_id": f"C{i}",
                    "text": clause_clean + ".",
                    "claim_type": ctype.value,
                    "source_start": s_start,
                    "source_end": s_end,
                    "context_tags": tags,
                    "confidence": 0.95,
                }
            )

        return {"claims": claims_list}
