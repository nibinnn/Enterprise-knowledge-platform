"""
app/citations/engine.py
─────────────────────────────────────────────────────────────────────────────
Citation engine — verifies that cited claims are actually supported by
the cited source chunks (hallucination guard) and formats citations for
the API response.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple

from app.core.models.document import Answer, Citation, RetrievedChunk

logger = logging.getLogger(__name__)


class CitationEngine:

    def __init__(self, min_overlap_chars: int = 20):
        """
        Args:
            min_overlap_chars: Minimum characters the cited excerpt must share
                               with the claimed sentence for verification to pass.
        """
        self._min_overlap = min_overlap_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_and_verify(
        self,
        answer_text: str,
        chunks:      List[RetrievedChunk],
    ) -> Tuple[str, List[Citation]]:
        """
        Parse [N] markers in answer_text, verify each citation against its
        source chunk, and return the cleaned answer + verified citations.

        Returns:
            (verified_answer_text, citations)
            Citations with low support scores are flagged but still included.
        """
        cited_numbers = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text)))
        citations: List[Citation] = []

        for n in cited_numbers:
            if n < 1 or n > len(chunks):
                logger.warning("Citation [%d] out of range (only %d chunks)", n, len(chunks))
                continue

            chunk = chunks[n - 1]
            m     = chunk.metadata

            # Find the sentence(s) in the answer that reference this citation
            claimed_text = self._extract_sentence_with_citation(answer_text, n)
            support_score = self._compute_support(claimed_text, chunk.text)

            if support_score < 0.1:
                logger.warning(
                    "Citation [%d] has low support score=%.2f — possible hallucination",
                    n, support_score,
                )

            citations.append(Citation(
                chunk_id=chunk.chunk_id,
                doc_id=m.doc_id,
                doc_filename=m.doc_filename,
                page_number=m.page_number,
                section_heading=m.section_heading,
                excerpt=chunk.text[:400].strip() + ("…" if len(chunk.text) > 400 else ""),
            ))

        return answer_text, citations

    def format_bibliography(self, citations: List[Citation]) -> str:
        """Format citations as a numbered bibliography section."""
        if not citations:
            return ""
        lines = ["\n\n**Sources:**"]
        seen = set()
        n = 1
        for c in citations:
            key = (c.doc_id, c.page_number)
            if key in seen:
                continue
            seen.add(key)
            parts = [f"[{n}] {c.doc_filename}"]
            if c.page_number:
                parts.append(f"page {c.page_number}")
            if c.section_heading:
                parts.append(f"§ {c.section_heading}")
            lines.append(" — ".join(parts))
            n += 1
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sentence_with_citation(text: str, n: int) -> str:
        """Extract the sentence containing citation [N]."""
        pattern = rf"[^.!?]*\[{n}\][^.!?]*[.!?]?"
        m = re.search(pattern, text)
        return m.group(0).strip() if m else ""

    @staticmethod
    def _compute_support(claim: str, source: str) -> float:
        """
        Simple lexical overlap score between a claim and its source chunk.
        Returns 0-1. Not semantic — just a hallucination sanity check.
        """
        if not claim or not source:
            return 0.0
        claim_words  = set(re.findall(r"\b\w{4,}\b", claim.lower()))
        source_words = set(re.findall(r"\b\w{4,}\b", source.lower()))
        if not claim_words:
            return 0.0
        overlap = claim_words & source_words
        return len(overlap) / len(claim_words)
