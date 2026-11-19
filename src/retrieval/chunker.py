"""Deterministic configurable chunking engine for trusted reference contexts.

RAGTruth contexts span:
1. QA (MS-MARCO): Multiple retrieved passages, often with headers ("Passage 1: ...").
2. Summary (CNN/DM): News articles with multiple paragraphs.
3. Data2txt (Yelp): Business details and customer reviews.

Rationale for Chunk Size:
    A target chunk size of ~300 characters (~50-60 tokens) with ~60 character overlap:
    - Captures a complete atomic factual proposition together with its subject and temporal anchor.
    - Prevents Dilution: Large chunks (e.g. 1000+ chars) dilute specific factual details in embedding space.
    - Aligns with downstream NLI cross-encoders (which perform best when premise and hypothesis are concise).
    - Overlap preserves boundary-straddling facts across adjacent chunks.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.retrieval.schema import ChunkConfig, EvidenceChunk


class DeterministicChunker:
    """Chunks source reference contexts into discrete, traceable evidence passages."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    @staticmethod
    def _split_into_sentences(text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences while tracking [start, end] character offsets."""
        sentences: list[tuple[str, int, int]] = []
        if not text.strip():
            return []

        # Split on sentence terminals followed by whitespace
        pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])|\n+")
        last_idx = 0
        for match in pattern.finditer(text):
            sent_text = text[last_idx:match.start()].strip()
            if sent_text:
                start_offset = text.find(sent_text, last_idx)
                end_offset = start_offset + len(sent_text)
                sentences.append((sent_text, start_offset, end_offset))
            last_idx = match.end()

        # Final sentence
        tail = text[last_idx:].strip()
        if tail:
            start_offset = text.find(tail, last_idx)
            end_offset = start_offset + len(tail)
            sentences.append((tail, start_offset, end_offset))

        return sentences

    def chunk_document(
        self,
        source_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[EvidenceChunk]:
        """Chunk a single source document into EvidenceChunk objects.

        Preserves exact character offsets and metadata.
        """
        raw_text = text.strip() if text else ""
        if not raw_text:
            return []

        # If text is shorter than chunk size, return single chunk
        if len(raw_text) <= self.config.chunk_size:
            return [
                EvidenceChunk(
                    chunk_id=f"{source_id}_chk_0",
                    source_id=source_id,
                    text=raw_text,
                    char_start=0,
                    char_end=len(raw_text),
                    metadata=metadata or {},
                )
            ]

        # Check for explicit passage delimiters (e.g. MS MARCO "Passage 1: ... Passage 2:")
        passage_matches = list(re.finditer(r"(Passage\s+\d+:)", raw_text, re.IGNORECASE))
        if len(passage_matches) >= 2:
            chunks: list[EvidenceChunk] = []
            for i in range(len(passage_matches)):
                start = passage_matches[i].start()
                end = passage_matches[i + 1].start() if i + 1 < len(passage_matches) else len(raw_text)
                p_text = raw_text[start:end].strip()
                if p_text:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{source_id}_chk_{i}",
                            source_id=source_id,
                            text=p_text,
                            char_start=start,
                            char_end=start + len(p_text),
                            metadata=dict(metadata or {}, passage_index=i + 1),
                        )
                    )
            return chunks

        # Standard sentence boundary sliding window
        sentences = self._split_into_sentences(raw_text)
        if not sentences:
            return []

        chunks: list[EvidenceChunk] = []
        cur_sentences: list[tuple[str, int, int]] = []
        cur_len = 0
        chunk_idx = 0

        for sent, s_start, s_end in sentences:
            sent_len = len(sent)

            if cur_sentences and (cur_len + sent_len > self.config.chunk_size):
                # Form chunk from accumulated sentences
                c_text = " ".join(s[0] for s in cur_sentences)
                c_start = cur_sentences[0][1]
                c_end = cur_sentences[-1][2]

                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{source_id}_chk_{chunk_idx}",
                        source_id=source_id,
                        text=c_text,
                        char_start=c_start,
                        char_end=c_end,
                        metadata=dict(metadata or {}, sentence_count=len(cur_sentences)),
                    )
                )
                chunk_idx += 1

                # Retain overlapping sentences for continuity
                overlap_accum: list[tuple[str, int, int]] = []
                overlap_len = 0
                for prev_sent in reversed(cur_sentences):
                    if overlap_len + len(prev_sent[0]) <= self.config.chunk_overlap:
                        overlap_accum.insert(0, prev_sent)
                        overlap_len += len(prev_sent[0])
                    else:
                        break

                cur_sentences = overlap_accum
                cur_len = overlap_len

            cur_sentences.append((sent, s_start, s_end))
            cur_len += sent_len

        # Final chunk
        if cur_sentences:
            c_text = " ".join(s[0] for s in cur_sentences)
            c_start = cur_sentences[0][1]
            c_end = cur_sentences[-1][2]
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{source_id}_chk_{chunk_idx}",
                    source_id=source_id,
                    text=c_text,
                    char_start=c_start,
                    char_end=c_end,
                    metadata=dict(metadata or {}, sentence_count=len(cur_sentences)),
                )
            )

        return chunks
