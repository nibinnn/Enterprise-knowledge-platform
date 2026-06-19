"""
app/core/base/parser.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all document parsers.

Each concrete parser (PDFParser, DOCXParser, etc.) inherits from this.
The factory pattern in ParserRegistry auto-selects the right parser
based on file extension — callers never import concrete classes directly.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from app.core.models.document import Document, DocumentType

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Contract every parser must fulfil.

    Implement `parse()` and `supported_extensions`.
    Everything else (file validation, logging, error wrapping) is handled here.
    """

    # Each subclass declares which extensions it handles, e.g. {".pdf"}
    supported_extensions: frozenset[str] = frozenset()

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path) -> Document:
        """
        Parse a document file and return a structured Document.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file type is not supported by this parser.
            ParseError: if parsing fails for any other reason.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ValueError(
                f"{self.__class__.__name__} does not support '{path.suffix}'. "
                f"Supported: {self.supported_extensions}"
            )

        logger.info("Parsing %s with %s", path.name, self.__class__.__name__)
        try:
            doc = self._parse(path)
            doc.metadata.source_path = str(path)
            logger.info(
                "Parsed '%s' → %d chars, %d sections",
                path.name, len(doc.raw_text), len(doc.sections),
            )
            return doc
        except Exception as exc:
            logger.error("Failed to parse '%s': %s", path.name, exc)
            raise ParseError(f"Error parsing '{path.name}': {exc}") from exc

    def supports(self, file_path: str | Path) -> bool:
        """Return True if this parser handles the given file extension."""
        return Path(file_path).suffix.lower() in self.supported_extensions

    # ── To implement ──────────────────────────────────────────────────────────

    @abstractmethod
    def _parse(self, path: Path) -> Document:
        """
        Core parsing logic. The public `parse()` wraps this with
        validation and error handling, so implement only the happy path here.
        """

    @abstractmethod
    def extract_metadata(self, path: Path) -> dict:
        """
        Extract raw metadata from the file (author, dates, title, page count …).
        Called internally by `_parse()`; exposed for testing.
        """


# ─────────────────────────── Registry ────────────────────────────────────────

class ParserRegistry:
    """
    Singleton registry that maps file extensions to parser instances.
    Concrete parsers register themselves via `register()`.

    Usage:
        registry = ParserRegistry.instance()
        parser   = registry.get_parser("report.pdf")
        doc      = parser.parse("report.pdf")
    """

    _instance: Optional[ParserRegistry] = None
    _parsers: Dict[str, BaseParser]

    def __init__(self) -> None:
        self._parsers: Dict[str, BaseParser] = {}

    @classmethod
    def instance(cls) -> ParserRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, parser: BaseParser) -> None:
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser
            logger.debug("Registered %s for extension '%s'", parser.__class__.__name__, ext)

    def get_parser(self, file_path: str | Path) -> BaseParser:
        ext = Path(file_path).suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            supported = list(self._parsers.keys())
            raise UnsupportedFileTypeError(
                f"No parser registered for '{ext}'. Supported: {supported}"
            )
        return parser

    def supported_extensions(self) -> List[str]:
        return list(self._parsers.keys())


# ─────────────────────────── Exceptions ──────────────────────────────────────

class ParseError(Exception):
    """Raised when a parser encounters an unrecoverable error."""


class UnsupportedFileTypeError(ValueError):
    """Raised when no parser is registered for a file extension."""
