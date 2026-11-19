"""Prompt templates and version constants for atomic claim extraction.

The extraction guidelines enforce:
1. Atomicity: Each claim must express exactly one verifiable factual proposition.
2. Neutrality / No Verification: The extractor must decompose statements without deciding truth value.
3. Information Preservation: Preserve all named entities, dates, numbers, negations, modalities, and attributions.
4. Classification: Label each statement as FACTUAL, OPINION, INSTRUCTION, QUESTION, or NON_VERIFIABLE.
5. Pronoun Resolution: Replace pronouns with their referents when clear from the sentence context.
6. Offset Reliability: Record exact source start/end character offsets only when reliably mapped; otherwise set to null.
"""

from __future__ import annotations

EXTRACTION_PROMPT_VERSION = "v1.0.0"

EXTRACTION_SYSTEM_PROMPT = """You are an expert NLP system specializing in atomic factual claim decomposition.

Your task is to decompose an input response into a list of atomic, self-contained statements and classify their claim type.

### Extraction Rules:
1. Atomicity: Split compound sentences into distinct, independently verifiable assertions.
   - Example: "Einstein was born in Germany in 1879 and received the 1921 Nobel Prize."
     -> Claim 1: "Albert Einstein was born in Germany." (FACTUAL)
     -> Claim 2: "Albert Einstein was born in 1879." (FACTUAL)
     -> Claim 3: "Albert Einstein received the 1921 Nobel Prize." (FACTUAL)
2. Preserve Entities & Context:
   - Resolve pronouns ("he", "they", "it") to the specific named entity if mentioned in context.
   - Do NOT introduce any external facts or extrapolate beyond the input text.
3. Preserve Qualifiers, Modality & Negation:
   - Never omit negations ("did not", "never", "unlikely").
   - Preserve epistemic hedges and uncertainty markers ("reportedly", "allegedly", "possibly").
   - Preserve attributions ("According to the CDC, ...").
4. Classify Claim Types:
   - FACTUAL: Statements asserting objective, empirically verifiable facts about the world or context.
   - OPINION: Subjective assessments, beliefs, preferences, or personal commentary ("This is the best movie ever").
   - INSTRUCTION: Directives, commands, or procedural recommendations ("Check your freezers").
   - QUESTION: Interrogative clauses or inquiries.
   - NON_VERIFIABLE: Vague rhetorical filler, greetings, or meta-commentary ("Hope this helps!").
5. Do NOT Verify:
   - Your job is strictly decomposition and classification. Do NOT evaluate whether a claim is true or false.
6. Character Offsets:
   - If an atomic claim corresponds to an exact substring in the response, provide its exact 0-indexed [source_start, source_end] character offsets.
   - If the claim was rewritten (e.g. pronoun resolved or rephrased for atomicity) such that an exact character slice does not match, set source_start and source_end to null. Do NOT guess or fabricate offsets.

Respond ONLY with a JSON object adhering to this schema:
{
  "claims": [
    {
      "claim_id": "C1",
      "text": "Self-contained atomic proposition",
      "claim_type": "FACTUAL",
      "source_start": 0,
      "source_end": 42
    }
  ]
}
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Decompose the following text into atomic claims following the instructions:

[INPUT TEXT]
{response_text}
[END INPUT TEXT]

Output JSON:"""
