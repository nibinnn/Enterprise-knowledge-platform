"""
app/rag/context_builder.py
─────────────────────────────────────────────────────────────────────────────
ContextBuilder — assembles the LLM context from retrieved chunks.

Responsibilities:
  1. Deduplicate near-identical chunks (same doc + page + similar text)
  2. Trim to the configured token budget
  3. Format into a numbered source block the LLM can cite by number
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import List

from app.chunkers.token_counter import TokenCounter
from app.core.models.document import QueryContext, RetrievedChunk

logger = logging.getLogger(__name__)


class ContextBuilder:

    def __init__(self, max_tokens: int = 8_000, tiktoken_model: str = "cl100k_base"):
        self._max_tokens = max_tokens
        self._tc = TokenCounter(model=tiktoken_model)

    def build(self, query: str, chunks: List[RetrievedChunk]) -> QueryContext:
        deduplicated = self._deduplicate(chunks)
        selected, truncated = self._apply_token_budget(deduplicated)
        formatted = self._format_context(selected)

        total_tokens = self._tc.count(formatted)
        logger.info(
            "Context built: %d chunks → %d tokens (truncated=%s)",
            len(selected), total_tokens, truncated,
        )
        return QueryContext(
            query=query,
            chunks=selected,
            formatted_context=formatted,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove chunks that are near-duplicates (same doc + page + overlapping text)."""
        seen_keys: set = set()
        result: List[RetrievedChunk] = []
        for chunk in chunks:
            # Key: doc_id + first 120 chars (catches overlap copies from chunking)
            key = (chunk.metadata.doc_id, chunk.text[:120].strip())
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(chunk)
        return result

    def _apply_token_budget(
        self, chunks: List[RetrievedChunk]
    ) -> tuple[List[RetrievedChunk], bool]:
        selected: List[RetrievedChunk] = []
        used_tokens = 0
        for chunk in chunks:
            chunk_tokens = self._tc.count(chunk.text) + 50  # 50 token overhead per source header
            if used_tokens + chunk_tokens > self._max_tokens:
                return selected, True
            selected.append(chunk)
            used_tokens += chunk_tokens
        return selected, False

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        """Format chunks as numbered sources for LLM citation."""
        if not chunks:
            return "No relevant sources found."
        lines = []
        for i, chunk in enumerate(chunks, 1):
            m = chunk.metadata
            header_parts = [f"Source [{i}]", m.doc_filename]
            if m.page_number:
                header_parts.append(f"p.{m.page_number}")
            if m.section_heading:
                header_parts.append(f"§ {m.section_heading}")
            lines.append(" | ".join(header_parts))
            lines.append(chunk.text.strip())
            lines.append("")
        return "\n".join(lines)
