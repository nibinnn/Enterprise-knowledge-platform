"""app/parsers/text_parser.py — plain-text and Markdown parser."""
from __future__ import annotations

from pathlib import Path

from app.core.base.parser import BaseParser
from app.core.models.document import (
    Document, DocumentMetadata, DocumentSection, DocumentType,
)
from app.parsers.utils import clean_text, read_text_file


class TextParser(BaseParser):
    """Parser for .txt and .md files."""

    supported_extensions: frozenset[str] = frozenset({".txt", ".md"})

    def _parse(self, path: Path) -> Document:
        raw = read_text_file(path)
        text = clean_text(raw)

        doc_type = DocumentType.MD if path.suffix.lower() == ".md" else DocumentType.TXT

        return Document(
            filename=path.name,
            doc_type=doc_type,
            raw_text=text,
            sections=[DocumentSection(heading=None, level=0, text=text)],
            metadata=DocumentMetadata(
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                word_count=len(text.split()),
            ),
        )

    def extract_metadata(self, path: Path) -> dict:
        return {"filename": path.name, "extension": path.suffix}
