"""app/parsers/utils.py — shared helpers for all parser implementations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from app.core.models.document import DocumentSection


def clean_text(text: str) -> str:
    """Normalize whitespace; collapse 3+ newlines to 2."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    """Read a text file, trying common encodings before falling back."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def table_to_markdown(rows: List[List[str]]) -> str:
    """Convert a list-of-rows (each row is a list of cell strings) to a Markdown table."""
    if not rows:
        return ""
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


class SectionAccumulator:
    """
    Stateful helper for HTML/structured parsers.
    Call push_heading() when a new heading is found, push_text() for body
    content, then flush() to get the final list of DocumentSection objects.
    """

    def __init__(self) -> None:
        self._sections: List[DocumentSection] = []
        self._current_heading: Optional[str] = None
        self._current_level: int = 0
        self._current_parts: List[str] = []
        self._has_table: bool = False

    def push_heading(self, text: str, level: int) -> None:
        self._commit()
        self._current_heading = text
        self._current_level = level
        self._current_parts = []
        self._has_table = False

    def push_text(
        self,
        text: str,
        is_table: bool = False,
        is_image: bool = False,
    ) -> None:
        if text.strip():
            self._current_parts.append(text)
            if is_table:
                self._has_table = True

    def flush(self) -> List[DocumentSection]:
        self._commit()
        return self._sections

    def _commit(self) -> None:
        body = "\n\n".join(self._current_parts).strip()
        if body:
            self._sections.append(
                DocumentSection(
                    heading=self._current_heading,
                    level=self._current_level,
                    text=body,
                    has_table=self._has_table,
                )
            )
        self._current_parts = []
        self._has_table = False
