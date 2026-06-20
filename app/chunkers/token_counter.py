"""
app/chunkers/token_counter.py
─────────────────────────────────────────────────────────────────────────────
Shared token-counting utility used by every chunker.

Priority:
  1. tiktoken (OpenAI BPE)  — most accurate for Claude/GPT models
  2. Word-count heuristic   — ~0.75 tokens per word — always available

All chunkers import TokenCounter; they never call tiktoken directly.
This keeps the fallback in one place and makes testing easy.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

# Tokens-per-word ratio used when tiktoken is unavailable
_WORDS_PER_TOKEN_RATIO: float = 0.75


class TokenCounter:
    """
    Counts tokens and splits text into token windows.

    Args:
        model: tiktoken encoding name (e.g. "cl100k_base" for GPT-4 / Claude).
               Falls back to word-count heuristic if tiktoken is not installed.
    """

    def __init__(self, model: str = "cl100k_base"):
        self._model = model
        self._enc = self._load_encoding(model)
        if self._enc is None:
            logger.warning(
                "tiktoken not available — using word-count heuristic "
                "(1 token ≈ %.2f words). Install tiktoken for accuracy.",
                _WORDS_PER_TOKEN_RATIO,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        if self._enc:
            return len(self._enc.encode(text))
        return max(1, round(len(text.split()) / _WORDS_PER_TOKEN_RATIO))

    def encode(self, text: str) -> List[int]:
        """Return the token id list for *text* (empty list if no tiktoken)."""
        if self._enc:
            return self._enc.encode(text)
        # Fake token ids: one per word (only used for split/merge logic)
        return list(range(len(text.split())))

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token ids back to text."""
        if self._enc:
            return self._enc.decode(token_ids)
        # Without tiktoken we can't invert fake ids — callers must not rely on this
        raise RuntimeError("decode() requires tiktoken")

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split *text* into overlapping token windows and return the decoded strings.

        This is used ONLY by FixedChunker. RecursiveChunker and SemanticChunker
        use `count()` for size-checking but do their own splitting.
        """
        if not self._enc:
            return self._chunk_by_words(text, chunk_size, chunk_overlap)

        ids = self._enc.encode(text)
        if not ids:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(ids):
            end = min(start + chunk_size, len(ids))
            chunk_ids = ids[start:end]
            chunks.append(self._enc.decode(chunk_ids))
            if end == len(ids):
                break
            start += chunk_size - chunk_overlap

        return chunks

    # ── Fallback word-based split ─────────────────────────────────────────────

    @staticmethod
    def _chunk_by_words(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Word-level fallback for chunk_by_tokens when tiktoken unavailable."""
        word_size    = max(1, round(chunk_size    * _WORDS_PER_TOKEN_RATIO))
        word_overlap = max(0, round(chunk_overlap * _WORDS_PER_TOKEN_RATIO))
        words = text.split()
        if not words:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + word_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += word_size - word_overlap
        return chunks

    # ── Loader ────────────────────────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_encoding(model: str):
        try:
            import tiktoken
            return tiktoken.get_encoding(model)
        except Exception:
            return None

    @property
    def has_tiktoken(self) -> bool:
        return self._enc is not None
