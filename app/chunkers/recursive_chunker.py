"""
app/chunkers/recursive_chunker.py
─────────────────────────────────────────────────────────────────────────────
Recursive character-based chunker.

Inspired by LangChain's RecursiveCharacterTextSplitter but:
  - Operates on Tokens (not characters) for accurate size estimation
  - Carries full DocumentSection provenance per chunk
  - Has section-aware mode: each DocumentSection is chunked independently
    so chunk boundaries never cross section boundaries

Algorithm:
  Given a separator hierarchy  ["\n\n", "\n", ". ", " ", ""]

  split_recursive(text, separators):
    sep = separators[0]
    pieces = text.split(sep)
    good, too_big = partition pieces by chunk_size

    for each too_big piece:
        sub_chunks = split_recursive(piece, separators[1:])

    merge good pieces into chunks (greedy, respecting chunk_size)
    add overlap tokens from the previous merged chunk

Best for:
  - Structured documents (paragraphs, sentences respected)
  - General-purpose enterprise documents (SOPs, policies, manuals)
  - Default strategy in the platform
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.chunkers.token_counter import TokenCounter
from app.core.base.chunker import BaseChunker
from app.core.models.document import Chunk, ChunkingStrategy, Document, DocumentSection

logger = logging.getLogger(__name__)

# Ordered separator hierarchy — try each one in sequence
_DEFAULT_SEPARATORS: List[str] = [
    "\n\n",   # paragraph break  (highest priority)
    "\n",     # line break
    ". ",     # sentence end
    "! ",
    "? ",
    "; ",
    ", ",     # clause
    " ",      # word
    "",       # character (last resort — splits anywhere)
]


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators, then merges
    small pieces into token-sized chunks with overlap.

    Args:
        chunk_size:    Target chunk size in tokens (default 512).
        chunk_overlap: Overlap in tokens between consecutive chunks (default 64).
        separators:    Custom separator hierarchy (default: paragraph → char).
        section_aware: If True (default), chunk each DocumentSection separately
                       so chunks never straddle section boundaries. This
                       preserves heading provenance in chunk metadata.
        tiktoken_model: tiktoken encoding (default "cl100k_base").
    """

    strategy = ChunkingStrategy.RECURSIVE

    def __init__(
        self,
        chunk_size:     int = 512,
        chunk_overlap:  int = 64,
        separators:     Optional[List[str]] = None,
        section_aware:  bool = True,
        tiktoken_model: str = "cl100k_base",
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._separators    = separators or _DEFAULT_SEPARATORS
        self._section_aware = section_aware
        self._tc            = TokenCounter(model=tiktoken_model)

    # ── BaseChunker interface ─────────────────────────────────────────────────

    def _chunk(self, text: str, document: Document) -> List[Chunk]:
        if self._section_aware and document.has_sections:
            return self._chunk_by_sections(document)
        return self._chunk_flat(text, document)

    # ── Section-aware mode ────────────────────────────────────────────────────

    def _chunk_by_sections(self, document: Document) -> List[Chunk]:
        """
        Chunk each DocumentSection independently.
        Preserves page_number, section_heading, and section_id per chunk.
        """
        all_chunks: List[Chunk] = []
        for section in document.sections:
            if not section.text.strip():
                continue
            raw_texts = self._split_recursive(section.text, self._separators)
            merged    = self._merge_with_overlap(raw_texts)
            for text in merged:
                if not text.strip():
                    continue
                chunk = self._make_chunk(
                    text=text,
                    document=document,
                    page_number=section.page_start,
                    section_heading=section.heading,
                    section_id=section.section_id,
                )
                chunk.metadata.token_count = self._tc.count(text)
                all_chunks.append(chunk)
        return all_chunks

    # ── Flat mode ─────────────────────────────────────────────────────────────

    def _chunk_flat(self, text: str, document: Document) -> List[Chunk]:
        """Chunk the full text without section awareness."""
        raw_texts = self._split_recursive(text, self._separators)
        merged    = self._merge_with_overlap(raw_texts)
        chunks: List[Chunk] = []
        for t in merged:
            if not t.strip():
                continue
            chunk = self._make_chunk(text=t, document=document)
            chunk.metadata.token_count = self._tc.count(t)
            chunks.append(chunk)
        return chunks

    # ── Core recursive splitting ──────────────────────────────────────────────

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        Returns a flat list of pieces, each <= chunk_size tokens.
        """
        if not text.strip():
            return []

        # Base case: text fits in one chunk
        if self._tc.count(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: hard split by token window
                return self._tc.chunk_by_tokens(
                    text, self.chunk_size, chunk_overlap=0
                )

            if sep not in text:
                continue

            # Split on this separator
            pieces = text.split(sep)
            # Restore the separator at the end of each piece (except last)
            # so merging later can reconstruct proper spacing
            pieces = [
                (p + sep) if idx < len(pieces) - 1 else p
                for idx, p in enumerate(pieces)
            ]

            result: List[str] = []
            remaining_separators = separators[i + 1:]

            for piece in pieces:
                if not piece.strip():
                    continue
                if self._tc.count(piece) <= self.chunk_size:
                    result.append(piece)
                else:
                    # Piece is still too large — recurse with finer separators
                    sub = self._split_recursive(piece, remaining_separators)
                    result.extend(sub)

            return result

        # Fallback: return the text as-is (shouldn't happen in practice)
        return [text]

    # ── Overlap merging ───────────────────────────────────────────────────────

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """
        Greedily merge small pieces into chunks up to chunk_size tokens.
        Prepend `chunk_overlap` tokens from the end of the previous chunk
        to the start of the next one.
        """
        if not pieces:
            return []

        chunks:  List[str]  = []
        current: List[str]  = []
        current_tokens:int  = 0
        prev_chunk_tail: str = ""

        def flush() -> None:
            nonlocal current, current_tokens, prev_chunk_tail
            if not current:
                return
            text = "".join(current).strip()
            if text:
                full = (prev_chunk_tail + " " + text).strip() if prev_chunk_tail else text
                chunks.append(full)
                # Compute the tail for the next chunk's overlap
                prev_chunk_tail = self._tail_tokens(text, self.chunk_overlap)
            current = []
            current_tokens = 0

        for piece in pieces:
            piece_tokens = self._tc.count(piece)

            # Single piece already exceeds chunk_size: emit as its own chunk
            if piece_tokens > self.chunk_size:
                flush()
                text = piece.strip()
                full = (prev_chunk_tail + " " + text).strip() if prev_chunk_tail else text
                chunks.append(full)
                prev_chunk_tail = self._tail_tokens(text, self.chunk_overlap)
                continue

            # Adding this piece would exceed the limit → flush first
            if current_tokens + piece_tokens > self.chunk_size:
                flush()

            current.append(piece)
            current_tokens += piece_tokens

        flush()
        return chunks

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        """Return the last *n_tokens* tokens of *text* as a decoded string."""
        if n_tokens <= 0:
            return ""
        if not self._tc.has_tiktoken:
            # Word-level fallback
            words = text.split()
            n_words = max(1, round(n_tokens * 0.75))
            return " ".join(words[-n_words:])
        ids = self._tc.encode(text)
        tail_ids = ids[-n_tokens:] if len(ids) > n_tokens else ids
        return self._tc.decode(tail_ids)
