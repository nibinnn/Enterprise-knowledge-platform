"""
app/parsers/text_cleaner.py
─────────────────────────────────────────────────────────────────────────────
PDF text cleaning and normalisation utilities.

Problems this module solves:
  1. Broken hyphenation  — "connec-\ntion" → "connection"
  2. Header / footer noise — page numbers, repeated company names
  3. Ligature substitution — "ﬁle" → "file", "ﬀ" → "ff"
  4. Whitespace explosions — multiple blank lines, trailing spaces
  5. Null bytes / control characters from corrupted PDFs
  6. CID font garbage — "（cid:123）" placeholders when font map is missing
  7. Repeated section detection — detect headers/footers repeated across pages
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Optional


# ─────────────────────────── Ligature map ────────────────────────────────────

_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",  "\ufb01": "fi",  "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st",
    "\ufb06": "st",  "\u0132": "IJ",  "\u0133": "ij",
    "\u00e6": "ae",  "\u00c6": "AE",  "\u0153": "oe",
    "\u0152": "OE",
}

_LIGATURE_TABLE = str.maketrans(_LIGATURES)


# ─────────────────────────── PDFTextCleaner ──────────────────────────────────

class PDFTextCleaner:
    """
    Stateful cleaner that learns repeated patterns (headers/footers)
    across pages so it can strip them from every page's text.

    Usage:
        cleaner = PDFTextCleaner()
        cleaner.learn_repeated_lines(all_page_texts)   # optional but recommended
        clean_pages = [cleaner.clean_page(p) for p in raw_page_texts]
        full_text   = cleaner.join_pages(clean_pages)
    """

    # Minimum character fraction for a page to be considered "text-bearing"
    MIN_TEXT_DENSITY_CHARS: int = 30

    def __init__(self, remove_headers_footers: bool = True):
        self._remove_headers_footers = remove_headers_footers
        self._repeated_lines: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def learn_repeated_lines(
        self, page_texts: List[str], min_occurrences: int = 3
    ) -> None:
        """
        Scan all pages and record lines that appear on ≥ min_occurrences pages.
        These are almost certainly headers or footers and will be stripped.
        """
        if not self._remove_headers_footers:
            return

        # Count the first + last 3 lines of each page (where headers/footers live)
        line_counter: Counter[str] = Counter()
        for text in page_texts:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            candidates = lines[:3] + lines[-3:]
            for line in candidates:
                # Only count non-trivial lines (not just page numbers)
                if len(line) > 4 and not self._is_page_number(line):
                    line_counter[line] += 1

        self._repeated_lines = {
            line for line, count in line_counter.items()
            if count >= min_occurrences
        }

    def clean_page(self, text: str, page_number: Optional[int] = None) -> str:
        """Full cleaning pipeline for a single page's text."""
        text = self._fix_encoding(text)
        text = self._remove_cid_garbage(text)
        text = self._fix_ligatures(text)
        text = self._fix_hyphenation(text)
        if self._remove_headers_footers:
            text = self._strip_repeated_lines(text)
            text = self._strip_page_number_lines(text)
        text = self._normalise_whitespace(text)
        return text.strip()

    def clean_raw(self, text: str) -> str:
        """
        Lighter cleaning for non-page-aware contexts
        (e.g. cleaning a single extracted string, not a full page).
        """
        text = self._fix_encoding(text)
        text = self._remove_cid_garbage(text)
        text = self._fix_ligatures(text)
        text = self._fix_hyphenation(text)
        text = self._normalise_whitespace(text)
        return text.strip()

    def join_pages(self, page_texts: List[str]) -> str:
        """Join cleaned pages with consistent double newlines."""
        return "\n\n".join(p for p in page_texts if p.strip())

    def is_scanned_page(self, text: str) -> bool:
        """
        Returns True if the page likely has no selectable text
        (indicating a scanned/image-only page that needs OCR).
        """
        return len(text.strip()) < self.MIN_TEXT_DENSITY_CHARS

    # ── Cleaning steps ────────────────────────────────────────────────────────

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Normalise unicode, drop null bytes and non-printable control chars."""
        # NFC normalisation handles composed vs decomposed forms
        text = unicodedata.normalize("NFC", text)
        # Remove null bytes and ASCII control chars (except tab/newline/CR)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Replace Windows-style CRLF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    @staticmethod
    def _remove_cid_garbage(text: str) -> str:
        """
        Remove CID font placeholders that appear when the font's
        glyph-to-character mapping is missing in PyMuPDF.
        e.g. "(cid:68)(cid:111)(cid:99)" → ""
        """
        text = re.sub(r"\(cid:\d+\)", "", text)
        # Also strip lone replacement characters
        text = text.replace("\ufffd", "")
        return text

    @staticmethod
    def _fix_ligatures(text: str) -> str:
        """Replace Unicode ligature characters with ASCII equivalents."""
        return text.translate(_LIGATURE_TABLE)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """
        Merge words that were split across lines by a soft hyphen.
        "connec-\ntion" → "connection"
        "multi-\nmodal" → "multi-modal"   (keep real compound hyphens)
        """
        # Soft hyphen at end of line followed immediately by lowercase = merge
        text = re.sub(r"(\w)-\n(\s*)([a-z])", r"\1\3", text)
        return text

    def _strip_repeated_lines(self, text: str) -> str:
        """Remove header/footer lines learned from `learn_repeated_lines()`."""
        if not self._repeated_lines:
            return text
        lines = text.splitlines()
        cleaned = [
            ln for ln in lines
            if ln.strip() not in self._repeated_lines
        ]
        return "\n".join(cleaned)

    @staticmethod
    def _strip_page_number_lines(text: str) -> str:
        """Remove standalone page number lines (e.g. '— 42 —', 'Page 3 of 10')."""
        lines = text.splitlines()
        cleaned = []
        for ln in lines:
            stripped = ln.strip()
            if PDFTextCleaner._is_page_number(stripped):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned)

    @staticmethod
    def _is_page_number(line: str) -> bool:
        """Return True if this line is just a page number."""
        patterns = [
            r"^[-–—]?\s*\d+\s*[-–—]?$",            # — 42 —  or  42
            r"^[Pp]age\s+\d+(\s+of\s+\d+)?$",       # Page 3 of 10
            r"^\d+\s*/\s*\d+$",                      # 3 / 10
        ]
        return any(re.match(p, line.strip()) for p in patterns)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse runs of blank lines (max 2), clean trailing spaces."""
        # Trailing whitespace on each line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # More than 2 consecutive blank lines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Multiple spaces → single (but preserve indentation)
        text = re.sub(r"(?<!\n) {2,}", " ", text)
        return text
