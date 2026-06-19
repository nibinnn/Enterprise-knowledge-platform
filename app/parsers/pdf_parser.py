"""
app/parsers/pdf_parser.py
─────────────────────────────────────────────────────────────────────────────
PDF parser — the most important parser in the system.

Extraction strategy (applied per page):

  1. PyMuPDF (fitz) — primary text + layout extraction
       ├─ Text blocks with font-size metadata → heading detection
       ├─ Block bounding boxes → reading order
       └─ Page-level metadata (title, author, dates)

  2. pdfplumber — table detection & extraction
       └─ Tables replaced with Markdown table strings in the output text

  3. Tesseract / EasyOCR — fallback for scanned/image pages
       └─ Triggered when a page yields < 30 chars after PyMuPDF extraction

Output: a fully populated Document object with:
  - raw_text   : all pages concatenated
  - sections   : one DocumentSection per detected heading block
  - metadata   : title, author, dates, page count, word count
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mode
from typing import Dict, List, Optional, Tuple

from app.core.base.parser import BaseParser
from app.core.models.document import (
    Document, DocumentMetadata, DocumentSection, DocumentType,
)
from app.parsers.ocr_engine import OCREngine
from app.parsers.table_extractor import PageTableResult, TableExtractor
from app.parsers.text_cleaner import PDFTextCleaner

logger = logging.getLogger(__name__)


# ─────────────────────────── Internal page model ─────────────────────────────

@dataclass
class _ParsedPage:
    """Intermediate representation of one extracted PDF page."""
    page_number: int
    text: str = ""
    was_ocr: bool = False
    blocks: List[dict] = field(default_factory=list)    # raw PyMuPDF blocks
    tables: List[str] = field(default_factory=list)     # markdown table strings


# ─────────────────────────── PDFParser ───────────────────────────────────────

class PDFParser(BaseParser):
    """
    Concrete parser for .pdf files.
    Registered automatically with ParserRegistry on import.
    """

    supported_extensions: frozenset[str] = frozenset({".pdf"})

    # A page is "scanned" if it has fewer than this many characters
    SCANNED_PAGE_THRESHOLD: int = 30

    # Font-size ratios (relative to body) for heading detection
    HEADING_RATIOS: List[Tuple[float, int]] = [
        (1.6, 1),   # ≥ 1.6× body size → H1
        (1.3, 2),   # ≥ 1.3× body size → H2
        (1.1, 3),   # ≥ 1.1× body size → H3
    ]

    def __init__(
        self,
        ocr_engine: str = "auto",
        ocr_languages: Optional[List[str]] = None,
        extract_tables: bool = True,
        remove_headers_footers: bool = True,
    ):
        self._ocr = OCREngine(
            engine=ocr_engine,
            languages=ocr_languages or ["en"],
        )
        self._table_extractor = TableExtractor() if extract_tables else None
        self._cleaner = PDFTextCleaner(
            remove_headers_footers=remove_headers_footers
        )

    # ── BaseParser interface ──────────────────────────────────────────────────

    def _parse(self, path: Path) -> Document:
        import fitz  # PyMuPDF — imported here so tests can mock

        fitz_doc = fitz.open(str(path))

        # ── 1. Extract metadata ───────────────────────────────────────────────
        metadata = self._build_metadata(fitz_doc, path)

        # ── 2. Extract tables (all pages at once via pdfplumber) ──────────────
        table_map: Dict[int, PageTableResult] = {}
        if self._table_extractor:
            table_results = self._table_extractor.extract_all_pages(path)
            table_map = {r.page_number: r for r in table_results}

        # ── 3. Extract text per page ──────────────────────────────────────────
        raw_pages: List[str] = []
        parsed_pages: List[_ParsedPage] = []

        for page_index in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_index]
            page_number = page_index + 1
            page_tables = table_map.get(page_number, None)

            parsed = self._extract_page(fitz_page, page_number, page_tables)
            parsed_pages.append(parsed)
            raw_pages.append(parsed.text)

        fitz_doc.close()

        # ── 4. Clean text (learn repeated patterns first) ─────────────────────
        self._cleaner.learn_repeated_lines(raw_pages)
        clean_pages = [
            self._cleaner.clean_page(p.text, p.page_number)
            for p in parsed_pages
        ]

        # ── 5. Detect sections from heading blocks ────────────────────────────
        sections = self._build_sections(parsed_pages, clean_pages)

        # ── 6. Build full raw_text ────────────────────────────────────────────
        raw_text = self._cleaner.join_pages(clean_pages)
        metadata.word_count = len(raw_text.split())

        return Document(
            filename=path.name,
            doc_type=DocumentType.PDF,
            raw_text=raw_text,
            sections=sections,
            metadata=metadata,
        )

    def extract_metadata(self, path: Path) -> dict:
        """Public metadata-only extraction (used in tests and admin tools)."""
        import fitz
        fitz_doc = fitz.open(str(path))
        meta = fitz_doc.metadata or {}
        fitz_doc.close()
        return meta

    # ── Page extraction ───────────────────────────────────────────────────────

    def _extract_page(
        self,
        fitz_page,
        page_number: int,
        page_tables: Optional[PageTableResult],
    ) -> _ParsedPage:
        """
        Extract text + layout blocks from a single fitz page.
        Falls back to OCR if the page appears to be scanned.
        """
        parsed = _ParsedPage(page_number=page_number)

        # Get structured blocks (type 0 = text, type 1 = image)
        page_dict = fitz_page.get_text("dict", sort=True)
        text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
        parsed.blocks = text_blocks

        # Raw text for OCR detection
        plain_text = fitz_page.get_text("text", sort=True)

        # ── Scanned page → OCR ───────────────────────────────────────────────
        if len(plain_text.strip()) < self.SCANNED_PAGE_THRESHOLD and self._ocr.is_available:
            logger.info("Page %d looks scanned — running OCR", page_number)
            pix = fitz_page.get_pixmap(dpi=300)
            from PIL import Image as PILImage
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_result = self._ocr.run(img)
            parsed.text = ocr_result.text
            parsed.was_ocr = True
        else:
            parsed.text = plain_text

        # ── Inject table markdown into page text ─────────────────────────────
        if page_tables and page_tables.has_tables:
            table_blocks = page_tables.as_markdown_blocks
            parsed.tables = table_blocks
            # Append tables after the page text so they are chunked with context
            parsed.text += "\n\n" + "\n\n".join(table_blocks)

        return parsed

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _build_metadata(self, fitz_doc, path: Path) -> DocumentMetadata:
        """Extract PDF metadata from the fitz document object."""
        raw_meta = fitz_doc.metadata or {}

        def clean_meta_str(val: Optional[str]) -> Optional[str]:
            if not val or val.strip() in ("", "None", "null"):
                return None
            return self._cleaner.clean_raw(val.strip())

        title = (
            clean_meta_str(raw_meta.get("title"))
            or self._infer_title_from_first_page(fitz_doc)
            or path.stem.replace("_", " ").replace("-", " ").title()
        )

        return DocumentMetadata(
            title=title,
            author=clean_meta_str(raw_meta.get("author")),
            created_date=clean_meta_str(raw_meta.get("creationDate")),
            modified_date=clean_meta_str(raw_meta.get("modDate")),
            page_count=len(fitz_doc),
            extra={
                "producer": clean_meta_str(raw_meta.get("producer")),
                "creator":  clean_meta_str(raw_meta.get("creator")),
                "subject":  clean_meta_str(raw_meta.get("subject")),
                "keywords": clean_meta_str(raw_meta.get("keywords")),
            },
        )

    def _infer_title_from_first_page(self, fitz_doc) -> Optional[str]:
        """
        If the PDF has no metadata title, guess it from the largest text
        on the first page (usually the document title).
        """
        if len(fitz_doc) == 0:
            return None
        try:
            page = fitz_doc[0]
            page_dict = page.get_text("dict", sort=True)
            blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

            # Find the span with the largest font size on the first page
            best_span = None
            best_size = 0.0
            for block in blocks[:5]:   # only check first 5 blocks
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("size", 0) > best_size and span.get("text", "").strip():
                            best_size = span["size"]
                            best_span = span

            if best_span:
                text = best_span["text"].strip()
                if 3 < len(text) < 120:
                    return text
        except Exception:
            pass
        return None

    # ── Section detection ─────────────────────────────────────────────────────

    def _build_sections(
        self,
        parsed_pages: List[_ParsedPage],
        clean_page_texts: List[str],
    ) -> List[DocumentSection]:
        """
        Walk through all pages and their layout blocks.
        Group text under detected headings into DocumentSection objects.
        """
        # Estimate body font size from all spans across the document
        body_size = self._estimate_body_font_size(parsed_pages)
        logger.debug("Estimated body font size: %.1f pt", body_size)

        sections: List[DocumentSection] = []
        current_heading: Optional[str] = None
        current_level: int = 0
        current_text_parts: List[str] = []
        current_page_start: int = 1
        current_page_end: int = 1

        def flush_section() -> None:
            """Save accumulated text as a new DocumentSection."""
            if not current_text_parts:
                return
            text_body = "\n".join(current_text_parts).strip()
            if not text_body:
                return
            sections.append(DocumentSection(
                heading=current_heading,
                level=current_level,
                text=text_body,
                page_start=current_page_start,
                page_end=current_page_end,
                has_table=bool(any("| --- |" in p for p in current_text_parts)),
            ))

        for page_index, parsed in enumerate(parsed_pages):
            page_num = parsed.page_number
            clean_text = clean_page_texts[page_index]

            if parsed.was_ocr or not parsed.blocks:
                # OCR pages have no block layout — treat as one flat section
                if clean_text.strip():
                    current_text_parts.append(clean_text)
                    current_page_end = page_num
                continue

            # Walk structured blocks to detect headings
            for block in parsed.blocks:
                block_heading, block_level, block_text = self._classify_block(
                    block, body_size
                )
                if not block_text.strip():
                    continue

                block_text = self._cleaner.clean_raw(block_text)

                if block_heading is not None:
                    # Heading found → flush previous section, start new one
                    flush_section()
                    current_heading = block_heading
                    current_level = block_level
                    current_text_parts = []
                    current_page_start = page_num
                    current_page_end = page_num
                else:
                    current_text_parts.append(block_text)
                    current_page_end = page_num

            # Add table markdown blocks for this page
            for table_md in parsed.tables:
                current_text_parts.append(table_md)

        flush_section()

        # If no sections were detected, make one big section from raw_text
        if not sections:
            all_text = "\n\n".join(t for t in clean_page_texts if t.strip())
            if all_text.strip():
                sections.append(DocumentSection(
                    heading=None,
                    level=0,
                    text=all_text,
                    page_start=1,
                    page_end=len(parsed_pages),
                ))

        logger.info("Detected %d section(s)", len(sections))
        return sections

    def _classify_block(
        self, block: dict, body_size: float
    ) -> Tuple[Optional[str], int, str]:
        """
        Classify a PyMuPDF text block as either a heading or body text.

        Returns:
            (heading_text, heading_level, block_plain_text)
            heading_text is None for body text blocks.
        """
        lines = block.get("lines", [])
        if not lines:
            return None, 0, ""

        # Collect all spans across all lines in this block
        all_spans = [
            span
            for line in lines
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if not all_spans:
            return None, 0, ""

        # Full block text
        block_text = " ".join(s["text"] for s in all_spans).strip()
        if not block_text:
            return None, 0, ""

        # Check if this block is a heading
        max_span_size = max(s.get("size", 0) for s in all_spans)
        is_bold = any(s.get("flags", 0) & 2**4 for s in all_spans)  # bold flag

        heading_level = self._get_heading_level(max_span_size, body_size, is_bold)

        # Headings are typically short (< 200 chars) and on their own block
        if heading_level > 0 and len(block_text) < 200:
            return block_text, heading_level, block_text

        return None, 0, block_text

    def _get_heading_level(
        self, font_size: float, body_size: float, is_bold: bool
    ) -> int:
        """Return heading level 1-3, or 0 for body text."""
        if body_size == 0:
            return 0

        ratio = font_size / body_size
        for min_ratio, level in self.HEADING_RATIOS:
            if ratio >= min_ratio:
                return level

        # Bold text at body size can also be H3
        if is_bold and ratio >= 1.0:
            return 3

        return 0

    @staticmethod
    def _estimate_body_font_size(parsed_pages: List[_ParsedPage]) -> float:
        """
        Find the most common font size across the document.
        This is almost always the body text size.
        """
        sizes: List[int] = []
        for parsed in parsed_pages:
            for block in parsed.blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0:
                            # Round to nearest 0.5pt to cluster near-identical sizes
                            sizes.append(round(size * 2) / 2)

        if not sizes:
            return 12.0   # reasonable default

        try:
            return float(mode(sizes))
        except Exception:
            return float(sorted(sizes)[len(sizes) // 2])   # median fallback
