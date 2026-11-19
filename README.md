# Haluc: Claim-Level Hallucination Detection in LLM Responses

A production-quality research platform for fine-grained claim-level and response-level hallucination detection in Retrieval-Augmented Generation (RAG) and LLM-generated responses.

---

## Project Objective

In Retrieval-Augmented Generation (RAG) settings, language models are provided with a user question and reference context passages. Despite having access to factual contexts, LLMs can generate hallucinations:
- **Evident Conflicts**: Claims that directly contradict the trusted reference context.
- **Baseless Information**: Claims that introduce unsupported external facts without sufficient grounding.

The ultimate objective of this system is to accept:
1. User question / prompt
2. LLM-generated response
3. Trusted / retrieved reference context

and determine which factual claims in the answer are:
- `SUPPORTED`: Fully grounded in the reference context.
- `CONTRADICTED`: Conflicting with facts in the reference context.
- `INSUFFICIENT_EVIDENCE`: Unsupported by the reference context.

From these claim-level predictions, the system computes claim-level risk distributions and an aggregate response-level hallucination score.

---

## Phase 1 Scope & Architecture

**Phase 1 implements exclusively the experimental foundation, data engineering, ground-truth representation, and evaluation infrastructure.**

No downstream NLI models, claim extractors, RAG retrievers, FastAPI backends, or LLM-as-judge modules are implemented in this phase.

```
                    ┌────────────────────────────────────────┐
                    │ RAGTruth Official Corpus               │
                    │ (ParticleMedia / HuggingFace Hub)      │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                             [ scripts/download_dataset.py ]
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ data/raw/                              │
                    │  ├── response.jsonl                    │
                    │  ├── source_info.jsonl                 │
                    │  └── dataset_info.json                 │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                             [ scripts/prepare_dataset.py ]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    ┌───────────────────────────┐                ┌───────────────────────────┐
    │ Schema & Span Validation  │                │ Group-Based Splitting     │
    │  - Offset exact matching  │                │  - No source_id leakage   │
    │  - Normalised taxonomy    │                │  - Preserves official     │
    └────────────┬──────────────┘                └─────────────┬─────────────┘
                 │                                             │
                 ▼                                             ▼
    ┌───────────────────────────┐                ┌───────────────────────────┐
    │ data/processed/           │                │ data/splits/              │
    │  ├── normalized_samples   │                │  ├── official/            │
    │  ├── invalid_records      │                │  │    ├── train.jsonl     │
    │  └── dataset_stats.json   │                │  │    └── test.jsonl      │
    └───────────────────────────┘                │  └── group_split/         │
                                                 │       ├── train.jsonl     │
                                                 │       ├── calibration.j   │
                                                 │       └── test.jsonl      │
                                                 └───────────────────────────┘
```

---

## Primary Dataset: RAGTruth

- **Source**: ParticleMedia/RAGTruth ([GitHub](https://github.com/ParticleMedia/RAGTruth) / [HuggingFace](https://huggingface.co/datasets/wandb/RAGTruth-processed))
- **Paper**: *"RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"* (Wu et al., 2024; arXiv:2401.00396)
- **Tasks**: Question Answering (`QA`), Summarization (`Summary`), Data-to-Text (`Data2txt`)
- **Models Annotated**: GPT-4-0613, GPT-3.5-Turbo-0613, Llama-2 (7B, 13B, 70B), Mistral-7B-Instruct

### Field Mapping (Original RAGTruth -> Internal Schema)

| Original File | Original Field | Internal Field | Description |
| :--- | :--- | :--- | :--- |
| `response.jsonl` | `id` | `sample_id` | Unique ID of the model response |
| `response.jsonl` | `source_id` | `source_id` | Shared key linking to source document |
| `response.jsonl` | `response` | `generated_response` | LLM response string |
| `response.jsonl` | `labels` | `hallucinated_spans` | List of `HallucinationSpan` offset annotations |
| `response.jsonl` | `labels` (type) | `hallucination_label` | Derived 3-class label (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`) |
| `response.jsonl` | `split` | `split` | Official split (`train` or `test`) |
| `response.jsonl` | `model`, `temperature`, `quality` | `metadata` | Generation & quality metadata |
| `source_info.jsonl` | `task_type` | `task_type` | `QA`, `Summary`, or `Data2txt` |
| `source_info.jsonl` | `prompt` | `question_or_prompt` | Input prompt given to LLM |
| `source_info.jsonl` | `source_info` | `reference_context` | Reference passages or facts |
| `source_info.jsonl` | `source` | `metadata["source"]` | Source benchmark (MARCO, Yelp, CNN/DM) |

---

## Directory Structure

```
Haluc/
├── configs/                   # Future detector / pipeline configuration
├── data/
│   ├── raw/                   # Unmodified raw JSONL files & dataset_info.json
│   ├── processed/             # normalized_samples.jsonl, invalid_records.jsonl, dataset_stats.json
│   └── splits/
│       ├── official/          # Official RAGTruth train/test partition
│       └── group_split/       # Group-based 70/15/15 train/calib/test partition (zero leakage)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Multi-strategy loader (GitHub raw, HuggingFace Hub, Local)
│   │   ├── preprocessing.py   # Merge, validation, span verification, stats
│   │   ├── schema.py          # Pydantic v2 data models & detector contracts
│   │   └── splitter.py        # Group-based splitting & leakage detection
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py         # Pure standalone metrics (Accuracy, Precision, Recall, F1, AUROC, AUPRC, CM)
│       └── evaluator.py       # Multi-class and binary evaluation pipeline
├── scripts/
│   ├── download_dataset.py    # Downloads real RAGTruth without mock fallbacks
│   └── prepare_dataset.py     # End-to-end dataset normalization, validation, and splitting
├── tests/
│   ├── test_loader.py         # Loader and persistence unit tests
│   ├── test_preprocessing.py  # Schema, span offset, and label mapping tests
│   ├── test_metrics.py        # Numerical metric correctness vs scikit-learn
│   └── test_splitter.py       # Group-based split & leakage detection tests
├── pyproject.toml             # Python package configuration and dependencies
└── README.md                  # Project documentation and Phase 1 report
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Recommended: [uv](https://github.com/astral-sh/uv) or standard `venv`

### Virtual Environment & Installation
```bash
# Using uv (fastest):
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using standard pip:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running the Pipeline

### 1. Download Dataset
Acquires real, authentic RAGTruth raw data directly from official sources:
```bash
python scripts/download_dataset.py
```
*Note: If network access blocks automatic downloading, place official `response.jsonl` and `source_info.jsonl` into `data/raw/` manually.*

### 2. Prepare & Validate Dataset
Merges responses with contexts, verifies span offsets against substrings, segregates invalid records, computes real dataset statistics, and generates leak-free group splits:
```bash
python scripts/prepare_dataset.py
```

### 4. Run NLI Baseline Evaluation (Phase 2)
Runs local zero-shot NLI baseline (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) on the RAGTruth test split:
```bash
python scripts/run_nli_baseline.py --limit 500 --batch-size 32
```
Outputs are written to:
- `data/evaluation/nli_baseline_metrics.json`: Detailed 3-way, binary, and latency metrics.
- `data/evaluation/error_analysis.json`: Representative false positives, false negatives, and high-confidence errors.

---

## Phase 2: Open-Source Pretrained NLI Baseline

Phase 2 establishes the first zero-shot detection baseline using open-source Natural Language Inference:
- **Model**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (DeBERTa-v3-base fine-tuned on MNLI, FEVER, ANLI)
- **Formulation**:
  - `Premise`: Trusted reference context
  - `Hypothesis`: Generated response
  - `Entailment` -> `SUPPORTED`
  - `Contradiction` -> `CONTRADICTED`
  - `Neutral` -> `INSUFFICIENT_EVIDENCE`
- **Output Contract**:
  Every pair outputs typed predictions with complete 3-way probabilities (`supported`, `insufficient_evidence`, `contradicted`), winning confidence, model name, and a `truncated` flag.

### Baseline Evaluation Results (500 Test Samples on NVIDIA GTX 1650)
- **Overall Accuracy (3-Way)**: 67.4%
- **Binary Hallucination Accuracy**: 71.2%
- **AUROC**: 0.6279 | **AUPRC**: 0.3608
- **Average Latency**: 338.35 ms/sample | **Throughput**: 2.96 samples/sec
- **Key Findings & Error Analysis**:
  1. *Context Window Truncation*: 69.6% of evaluation pairs exceeded DeBERTa's 512-token limit. In Summarization tasks, truncating the latter half of the source article resulted in high-confidence false positives (`INSUFFICIENT_EVIDENCE`), as the model could not see facts mentioned later in the text.
  2. *Conjunction Dominance in Response-Level Hypothesis*: In multi-sentence responses containing a single contradicted span among otherwise faithful sentences, the cross-encoder attended to the predominantly supported tokens, causing false negatives. This motivates atomic claim extraction in Phase 3.

---

## Evaluation Infrastructure

The evaluation framework provides:
- **Multi-class evaluation**:
  - `SUPPORTED` (0): Non-hallucinated
  - `CONTRADICTED` (1): Factual conflict
  - `INSUFFICIENT_EVIDENCE` (2): Unsubstantiated information
- **Binary evaluation**:
  - Non-hallucinated (0) vs. Hallucinated (1)
- **Metrics implemented**:
  - Accuracy
  - Per-class Precision, Recall, F1
  - Confusion Matrix ($N \times N$)
  - AUROC & AUPRC (for probabilistic detector outputs)
- **Contract Interface**:
  - Detectors implement `BaseDetector.predict(sample: NormalizedSample) -> DetectorPrediction`
  - Evaluator consumes predictions: `Evaluator.evaluate(predictions, ground_truth) -> EvaluationResult`

---

---

## Phase 3: Atomic Factual Claim Extraction

Phase 3 implements independent, fine-grained factual claim extraction to decompose compound responses into atomic propositions before verification:
- **Architecture**:
  - `BaseLLMProvider`: Abstract provider interface decoupling the system from specific vendors.
  - `AnthropicProvider`: Native Claude structured tool calling provider reading credentials exclusively from environment variables.
  - `RuleBasedExtractorProvider`: Fast, deterministic, zero-cost offline decomposer splitting compound conjunctions, resolving shared subjects, and identifying exact character offsets.
  - `AtomicClaimExtractor`: Core engine featuring bounded retry/repair, schema validation, and offset verification.
  - `ClaimExtractionCache`: Persistent SHA-256 disk and memory caching preventing redundant LLM invocations.
- **Claim Categories**:
  - `FACTUAL`: Verifiable empirical claims entering factual verification.
  - `OPINION`: Subjective assessments and aesthetic commentary (excluded from verification but preserved for auditing).
  - `INSTRUCTION`: User commands, directives, and advice.
  - `QUESTION`: Interrogative sentences.
  - `NON_VERIFIABLE`: Conversational filler and rhetorical meta-commentary.
- **Evaluation Fixtures**:
  - Verified across 6 canonical test fixtures covering compound facts, pure opinions, mixed texts, negations, questions, and real RAGTruth test responses (`scripts/evaluate_claim_extraction.py`).

---

## Phase 4: Evidence Indexing and Retrieval

Phase 4 implements deterministic passage chunking, dense vector indexing, and source-scoped Top-K candidate evidence retrieval for extracted factual claims:
- **Architecture**:
  - `DeterministicChunker`: Sentence-aligned sliding-window chunking (default: 300 chars, 60 char overlap) preserving provenance, source IDs, and exact character offsets.
  - `EvidenceIndex`: Local vector index combining SentenceTransformers (`all-MiniLM-L6-v2` / `BAAI/bge-small-en-v1.5`, 384 dimensions) and `faiss.IndexFlatIP` (cosine similarity over normalized embeddings).
  - `EvidenceRetriever`: Orchestrates retrieval with strict source isolation (`source_id_filter`) to prevent cross-sample evaluation leakage, and supports extensible reranking (`BaseReranker`).
- **Safety & Evaluation Integrity**:
  - *No Label Leakage*: Only raw reference texts (`source_info.jsonl`) are indexed. Annotations, labels, and model responses are never indexed.
  - *Source Scoping*: When querying claims from an evaluation sample, search is strictly constrained to chunks belonging to that sample's assigned source document.
  - *Compatibility Validation*: Index loader strictly enforces model and dimension parity (`IndexCompatibilityError`), preventing silent querying across mismatched vector spaces.
  - *Semantic Thresholds*: Semantic similarity is treated strictly as evidence relevance, not as proof of claim veracity (handled in later verification phases).

---

## Planned Future Phases

- **Phase 5**: Claim Verification & Calibrated Ensemble Scoring (NLI + LLM-as-Judge)
- **Phase 6**: Production FastAPI Service & Claim Inspection Frontend



