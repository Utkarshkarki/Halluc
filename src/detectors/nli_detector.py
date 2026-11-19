"""Open-source pretrained NLI detector for hallucination detection.

Utilizes DeBERTa-v3 sequence classification models (e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
Supports batched inference, GPU/CPU selection, FP16 execution, dynamic label mapping,
and documented long-context handling.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schema import (
    BaseDetector,
    DetectorPrediction,
    HallucinationLabel,
    NormalizedSample,
)
from src.detectors.schema import NLIConfig, NLIOutput, NLIProbabilities

logger = logging.getLogger(__name__)


class NLIDetector(BaseDetector):
    """Hallucination detector based on pretrained Natural Language Inference (NLI).

    NLI Formulation:
        Premise    = Reference / trusted evidence
        Hypothesis = Claim or generated statement to verify

    Label Mapping:
        ENTAILMENT    -> SUPPORTED
        CONTRADICTION -> CONTRADICTED
        NEUTRAL       -> INSUFFICIENT_EVIDENCE
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: NLIConfig,
        device: torch.device,
        entailment_idx: int,
        neutral_idx: int,
        contradiction_idx: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.config = config
        self._device = device
        self._entailment_idx = entailment_idx
        self._neutral_idx = neutral_idx
        self._contradiction_idx = contradiction_idx

        param = next(self._model.parameters(), None) if hasattr(self._model, "parameters") else None
        dtype_str = str(param.dtype) if param is not None else "unknown"

        logger.info(
            "NLIDetector initialized on %s (dtype=%s). Indices: Entailment=%d, Neutral=%d, Contradiction=%d",
            self._device,
            dtype_str,
            self._entailment_idx,
            self._neutral_idx,
            self._contradiction_idx,
        )

    @property
    def name(self) -> str:
        return f"nli-{self.config.model_name}"

    @property
    def device(self) -> torch.device:
        return self._device

    # ----- Factory: from_pretrained ---------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
        long_context_strategy: str = "chunk_evidence",
        seed: int = 42,
    ) -> NLIDetector:
        """Load pretrained NLI model and tokenizer from Hugging Face Hub."""
        torch.manual_seed(seed)

        # Device selection
        if device == "auto":
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected_device = torch.device(device)

        logger.info("Loading %s on %s...", model_name, selected_device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
        )

        # Use FP16 on CUDA to minimize VRAM usage
        if selected_device.type == "cuda" and fp16:
            model = model.half()

        model.to(selected_device)
        model.eval()

        # Dynamically inspect model id2label mapping
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        entailment_idx = -1
        neutral_idx = -1
        contradiction_idx = -1

        for idx, lbl_str in id2label.items():
            if "entail" in lbl_str:
                entailment_idx = idx
            elif "contra" in lbl_str:
                contradiction_idx = idx
            elif "neut" in lbl_str:
                neutral_idx = idx

        if -1 in (entailment_idx, neutral_idx, contradiction_idx):
            # Fallback for standard mnli ordering
            logger.warning(
                "Could not automatically parse all label names from %s; using standard order [entailment=0, neutral=1, contradiction=2]",
                id2label,
            )
            entailment_idx = 0
            neutral_idx = 1
            contradiction_idx = 2

        config = NLIConfig(
            model_name=model_name,
            revision=revision,
            batch_size=batch_size,
            max_length=max_length,
            device=str(selected_device),
            fp16=fp16 if selected_device.type == "cuda" else False,
            long_context_strategy=long_context_strategy,
            seed=seed,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=selected_device,
            entailment_idx=entailment_idx,
            neutral_idx=neutral_idx,
            contradiction_idx=contradiction_idx,
        )

    # ----- Inference Methods ----------------------------------------------

    def predict_pair(self, premise: str, hypothesis: str) -> NLIOutput:
        """Run NLI prediction for a single premise-hypothesis pair."""
        return self.predict_pairs([(premise, hypothesis)])[0]

    def predict_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int | None = None,
    ) -> list[NLIOutput]:
        """Run batched NLI inference over premise-hypothesis pairs.

        Handles tokenization, truncation tracking, probability extraction,
        and mapping to HallucinationLabel.
        """
        bs = batch_size or self.config.batch_size
        results: list[NLIOutput] = []

        for i in range(0, len(pairs), bs):
            batch_slice = pairs[i : i + bs]
            premises = [p[0].strip() if p[0] else "" for p in batch_slice]
            hypotheses = [p[1].strip() if p[1] else "" for p in batch_slice]

            # Detect truncation per sample by tokenizing without truncation first
            untruncated_lens = [
                len(self._tokenizer.encode(p, h, add_special_tokens=True, truncation=False))
                for p, h in zip(premises, hypotheses)
            ]
            truncated_flags = [l > self.config.max_length for l in untruncated_lens]

            # Tokenize with truncation and padding
            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, prob_vec in enumerate(probs):
                p_entail = float(prob_vec[self._entailment_idx])
                p_neut = float(prob_vec[self._neutral_idx])
                p_contra = float(prob_vec[self._contradiction_idx])

                # Build probabilities object
                nli_probs = NLIProbabilities(
                    supported=p_entail,
                    insufficient_evidence=p_neut,
                    contradicted=p_contra,
                )

                # Determine winning label
                max_prob = max(p_entail, p_neut, p_contra)
                if max_prob == p_entail:
                    pred_label = HallucinationLabel.SUPPORTED
                elif max_prob == p_contra:
                    pred_label = HallucinationLabel.CONTRADICTED
                else:
                    pred_label = HallucinationLabel.INSUFFICIENT_EVIDENCE

                premise_toks = len(self._tokenizer.encode(premises[j], add_special_tokens=False))
                hypo_toks = len(self._tokenizer.encode(hypotheses[j], add_special_tokens=False))

                results.append(
                    NLIOutput(
                        label=pred_label,
                        confidence=max_prob,
                        probabilities=nli_probs,
                        model=self.name,
                        truncated=truncated_flags[j],
                        premise_tokens=premise_toks,
                        hypothesis_tokens=hypo_toks,
                    )
                )

        return results

    # ----- BaseDetector Implementation ------------------------------------

    def predict(self, sample: NormalizedSample) -> DetectorPrediction:
        """Predict hallucination label for a single NormalizedSample."""
        premise = sample.reference_context
        hypothesis = sample.generated_response
        output = self.predict_pair(premise, hypothesis)

        return DetectorPrediction(
            sample_id=sample.sample_id,
            predicted_label=output.label,
            confidence=output.confidence,
            detector_name=self.name,
        )

    def predict_batch(
        self,
        samples: list[NormalizedSample],
        batch_size: int | None = None,
    ) -> list[DetectorPrediction]:
        """Predict for a batch of NormalizedSample objects."""
        pairs = [(s.reference_context, s.generated_response) for s in samples]
        outputs = self.predict_pairs(pairs, batch_size=batch_size)

        predictions: list[DetectorPrediction] = []
        for s, out in zip(samples, outputs):
            predictions.append(
                DetectorPrediction(
                    sample_id=s.sample_id,
                    predicted_label=out.label,
                    confidence=out.confidence,
                    detector_name=self.name,
                )
            )
        return predictions
