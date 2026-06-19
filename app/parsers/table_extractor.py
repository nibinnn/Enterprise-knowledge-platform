"""
app/parsers/table_extractor.py
─────────────────────────────────────────────────────────────────────────────
Table extraction from PDF pages using pdfplumber.

pdfplumber is better than PyMuPDF for tables because it uses whitespace
analysis and line detection, not just the raw text stream.

Output: each table is converted to a Markdown table string so it
        can be included in a DocumentSection's text field and chunked
        like normal text. The LLM handles markdown tables natively.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────── Data structures ─────────────────────────────────

@dataclass
class ExtractedTable:
    """A single table extracted from one PDF page."""
    page_number: int
    table_index: int            # order of the table on the page (0-based)
    raw_data: List[List[Optional[str]]]
    markdown: str = ""          # rendered Markdown representation
    row_count: int = 0
    col_count: int = 0

    def __post_init__(self) -> None:
        if self.raw_data:
            self.row_count = len(self.raw_data)
            self.col_count = max(len(row) for row in self.raw_data) if self.raw_data else 0


@dataclass
class PageTableResult:
    """All tables found on a single PDF page."""
    page_number: int
    tables: List[ExtractedTable] = field(default_factory=list)

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def as_markdown_blocks(self) -> List[str]:
        """Return each table as a separate markdown string."""
        return [t.markdown for t in self.tables if t.markdown]


# ─────────────────────────── TableExtractor ──────────────────────────────────

class TableExtractor:
    """
    Extracts tables from a PDF using pdfplumber.

    Usage:
        extractor = TableExtractor()
        results   = extractor.extract_all_pages("/path/to/doc.pdf")
        for page_result in results:
            for table in page_result.tables:
                print(table.markdown)
    """

    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        snap_tolerance: int = 3,
        join_tolerance: int = 3,
    ):
        """
        Args:
            min_rows:        Skip tables with fewer rows (avoids single-row noise).
            min_cols:        Skip tables with fewer columns.
            snap_tolerance:  pdfplumber pixel tolerance for line snapping.
            join_tolerance:  pdfplumber pixel tolerance for line joining.
        """
        self.min_rows = min_rows
        self.min_cols = min_cols
        self._table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": snap_tolerance,
            "join_tolerance": join_tolerance,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_all_pages(self, pdf_path: str | Path) -> List[PageTableResult]:
        """
        Extract tables from every page of a PDF.
        Returns one PageTableResult per page (even if that page has no tables).
        """
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — table extraction disabled.")
            return []

        results: List[PageTableResult] = []
        path = Path(pdf_path)

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_result = self._extract_page(page, page_num)
                    results.append(page_result)
                    if page_result.has_tables:
                        logger.debug(
                            "Page %d: %d table(s) found",
                            page_num, len(page_result.tables)
                        )
        except Exception as exc:
            logger.warning("Table extraction failed for '%s': %s", path.name, exc)

        total = sum(len(r.tables) for r in results)
        logger.info("Extracted %d table(s) from '%s'", total, path.name)
        return results

    def extract_page(self, pdf_path: str | Path, page_number: int) -> PageTableResult:
        """Extract tables from a single page (1-based page_number)."""
        try:
            import pdfplumber
        except ImportError:
            return PageTableResult(page_number=page_number)

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    raise ValueError(f"Page {page_number} out of range (1–{len(pdf.pages)})")
                page = pdf.pages[page_number - 1]
                return self._extract_page(page, page_number)
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_number, exc)
            return PageTableResult(page_number=page_number)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_page(self, page, page_number: int) -> PageTableResult:
        result = PageTableResult(page_number=page_number)

        try:
            raw_tables = page.extract_tables(self._table_settings)
        except Exception as exc:
            logger.debug("pdfplumber error on page %d: %s", page_number, exc)
            return result

        for idx, raw in enumerate(raw_tables):
            if not raw:
                continue

            # Filter trivially small tables
            non_empty_rows = [r for r in raw if any(c and c.strip() for c in r)]
            if len(non_empty_rows) < self.min_rows:
                continue
            if not raw[0] or len(raw[0]) < self.min_cols:
                continue

            table = ExtractedTable(
                page_number=page_number,
                table_index=idx,
                raw_data=raw,
            )
            table.markdown = self._to_markdown(raw)
            result.tables.append(table)

        return result

    @staticmethod
    def _to_markdown(raw: List[List[Optional[str]]]) -> str:
        """
        Convert a raw table (list of rows, each row a list of cell strings)
        to a GitHub-flavored Markdown table.

        If the first row looks like a header (has content in most cells),
        it is treated as the header row with a separator line beneath it.
        """
        if not raw:
            return ""

        # Normalise cells: strip whitespace, replace None with ""
        def cell(v: Optional[str]) -> str:
            if v is None:
                return ""
            # Collapse internal whitespace
            return " ".join(str(v).split())

        # Determine column count (max across all rows)
        col_count = max(len(row) for row in raw)

        # Pad rows to uniform width
        rows = [
            [cell(row[i]) if i < len(row) else "" for i in range(col_count)]
            for row in raw
        ]

        if not rows:
            return ""

        # Build markdown
        lines: List[str] = []
        header = rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
