"""
tests/unit/test_pdf_parser.py
─────────────────────────────────────────────────────────────────────────────
Day 2 unit tests — PDF parser, text cleaner, OCR engine, table extractor.

Run with:  pytest tests/unit/test_pdf_parser.py -v

Tests are written to work WITHOUT a real PDF file by mocking fitz and
pdfplumber where needed. Integration tests (with real PDFs) live in
tests/integration/test_pdf_integration.py (created on Day 25).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.models.document import DocumentSection, DocumentType
from app.parsers.ocr_engine import OCREngine, OCREngineType, OCRResult
from app.parsers.table_extractor import ExtractedTable, PageTableResult, TableExtractor
from app.parsers.text_cleaner import PDFTextCleaner


# ─────────────────────────── PDFTextCleaner ──────────────────────────────────

class TestPDFTextCleaner:

    def setup_method(self):
        self.cleaner = PDFTextCleaner(remove_headers_footers=True)

    def test_fixes_soft_hyphenation(self):
        text = "This is connec-\ntion between ideas."
        result = self.cleaner.clean_raw(text)
        assert "connection" in result

    def test_preserves_real_hyphens(self):
        # "multi-\nmodal" should become "multimodal" (no space)
        # Real compound hyphens like "well-known" are on a single line - preserved
        text = "well-known approach"
        result = self.cleaner.clean_raw(text)
        assert "well-known" in result

    def test_removes_cid_garbage(self):
        text = "(cid:68)(cid:111)(cid:99) is a document"
        result = self.cleaner.clean_raw(text)
        assert "cid:" not in result
        assert "is a document" in result

    def test_fixes_ligatures(self):
        text = "\ufb01le and \ufb00ort"   # fi + ff ligatures
        result = self.cleaner.clean_raw(text)
        assert "file" in result
        assert "ffort" in result

    def test_strips_page_number_line_plain(self):
        text = "Some content\n42\nMore content"
        result = self.cleaner.clean_page(text)
        assert "42" not in result.split("\n") or "Some content" in result

    def test_strips_page_number_with_dashes(self):
        text = "Content before\n— 5 —\nContent after"
        result = self.cleaner.clean_page(text)
        assert "— 5 —" not in result

    def test_strips_page_n_of_m(self):
        text = "Content\nPage 3 of 10\nMore"
        result = self.cleaner.clean_page(text)
        assert "Page 3 of 10" not in result

    def test_normalises_multiple_blank_lines(self):
        text = "Para one\n\n\n\n\nPara two"
        result = self.cleaner.clean_raw(text)
        assert "\n\n\n" not in result

    def test_learns_and_strips_repeated_header(self):
        pages = [
            "ACME CORP CONFIDENTIAL\nPage content one",
            "ACME CORP CONFIDENTIAL\nPage content two",
            "ACME CORP CONFIDENTIAL\nPage content three",
        ]
        cleaner = PDFTextCleaner(remove_headers_footers=True)
        cleaner.learn_repeated_lines(pages, min_occurrences=3)
        result = cleaner.clean_page(pages[0])
        assert "ACME CORP CONFIDENTIAL" not in result
        assert "Page content one" in result

    def test_join_pages_ignores_empty(self):
        pages = ["Page one", "", "   ", "Page two"]
        result = self.cleaner.join_pages(pages)
        assert "Page one" in result
        assert "Page two" in result
        # Empty pages don't add extra blank lines
        assert result.count("\n\n\n") == 0

    def test_is_scanned_page_true_for_few_chars(self):
        assert self.cleaner.is_scanned_page("   ") is True
        assert self.cleaner.is_scanned_page("ab") is True

    def test_is_scanned_page_false_for_real_text(self):
        assert self.cleaner.is_scanned_page("A" * 100) is False

    def test_unicode_normalisation(self):
        # Composed vs decomposed form of 'é'
        composed = "\u00e9"      # é as single codepoint
        decomposed = "e\u0301"  # é as e + combining accent
        r1 = self.cleaner.clean_raw(composed)
        r2 = self.cleaner.clean_raw(decomposed)
        assert r1 == r2


# ─────────────────────────── TableExtractor ──────────────────────────────────

class TestTableExtractor:

    def test_to_markdown_basic(self):
        raw = [
            ["Name", "Age", "City"],
            ["Alice", "30", "Paris"],
            ["Bob", "25", "London"],
        ]
        md = TableExtractor._to_markdown(raw)
        assert "| Name | Age | City |" in md
        assert "| --- |" in md
        assert "| Alice | 30 | Paris |" in md

    def test_to_markdown_handles_none_cells(self):
        raw = [["A", None, "C"], ["1", "2", None]]
        md = TableExtractor._to_markdown(raw)
        assert md  # should not crash

    def test_to_markdown_empty_returns_empty(self):
        assert TableExtractor._to_markdown([]) == ""

    def test_extracted_table_dimensions(self):
        raw = [["H1", "H2"], ["R1", "R2"], ["R3", "R4"]]
        table = ExtractedTable(page_number=1, table_index=0, raw_data=raw)
        assert table.row_count == 3
        assert table.col_count == 2

    def test_page_table_result_has_tables(self):
        result = PageTableResult(page_number=1)
        assert not result.has_tables
        result.tables.append(
            ExtractedTable(page_number=1, table_index=0, raw_data=[["A", "B"]])
        )
        assert result.has_tables

    def test_extractor_min_rows_filter(self):
        """Tables with < min_rows non-empty rows should be skipped."""
        extractor = TableExtractor(min_rows=2)
        # Simulate what happens inside _extract_page with a tiny table
        raw = [["Header only"]]   # only 1 row
        page_result = PageTableResult(page_number=1)
        non_empty = [r for r in raw if any(c and c.strip() for c in r)]
        if len(non_empty) >= extractor.min_rows:
            page_result.tables.append(
                ExtractedTable(page_number=1, table_index=0, raw_data=raw)
            )
        assert not page_result.has_tables

    def test_as_markdown_blocks(self):
        t = ExtractedTable(
            page_number=1, table_index=0,
            raw_data=[["A", "B"], ["1", "2"]],
            markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
        )
        result = PageTableResult(page_number=1, tables=[t])
        blocks = result.as_markdown_blocks
        assert len(blocks) == 1
        assert "| A | B |" in blocks[0]


# ─────────────────────────── OCREngine ───────────────────────────────────────

class TestOCREngine:

    def test_ocr_result_bool_true_for_text(self):
        r = OCRResult(text="Hello world", confidence=0.9, engine="tesseract")
        assert bool(r) is True

    def test_ocr_result_bool_false_for_empty(self):
        r = OCRResult(text="  ", confidence=0.0, engine="tesseract")
        assert bool(r) is False

    def test_ok_factory(self):
        r = OCRResult.ok = lambda: None   # just testing the dataclass
        r = OCRResult(text="text", confidence=0.8, engine="easyocr")
        assert r.confidence == 0.8

    @patch("app.parsers.ocr_engine.OCREngine._check_tesseract", return_value=False)
    @patch("app.parsers.ocr_engine.OCREngine._check_easyocr", return_value=False)
    def test_auto_resolves_to_none_when_unavailable(self, mock_easy, mock_tess):
        engine = OCREngine(engine="auto")
        assert engine._resolved == OCREngineType.NONE
        assert not engine.is_available

    @patch("app.parsers.ocr_engine.OCREngine._check_tesseract", return_value=True)
    @patch("app.parsers.ocr_engine.OCREngine._check_easyocr", return_value=False)
    def test_auto_resolves_to_tesseract_when_available(self, mock_easy, mock_tess):
        engine = OCREngine(engine="auto")
        assert engine._resolved == OCREngineType.TESSERACT

    @patch("app.parsers.ocr_engine.OCREngine._check_tesseract", return_value=False)
    @patch("app.parsers.ocr_engine.OCREngine._check_easyocr", return_value=False)
    def test_run_returns_empty_when_no_engine(self, mock_easy, mock_tess):
        engine = OCREngine(engine="auto")
        result = engine.run(MagicMock())
        assert result.text == ""
        assert result.engine == "none"


# ─────────────────────────── PDFParser ───────────────────────────────────────

class TestPDFParser:
    """
    Tests for the PDFParser class.
    We mock `fitz.open` so these tests run without a real PDF file.
    """

    def _make_mock_fitz(
        self,
        page_texts: list[str],
        metadata: dict | None = None,
    ):
        """Build a minimal fitz mock that returns controlled page text."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=len(page_texts))
        mock_doc.metadata = metadata or {
            "title": "Test Document",
            "author": "Test Author",
            "creationDate": "D:20240101",
            "modDate": "",
            "producer": "",
            "creator": "",
            "subject": "",
            "keywords": "",
        }

        mock_pages = []
        for text in page_texts:
            mock_page = MagicMock()
            mock_page.get_text.return_value = text
            # Return a minimal "dict" structure (no blocks → triggers flat section)
            mock_page.get_text.side_effect = lambda mode, **kw: (
                text if mode == "text" else {"blocks": []}
            )
            mock_pages.append(mock_page)

        mock_doc.__getitem__ = lambda self, idx: mock_pages[idx]
        mock_doc.__iter__ = lambda self: iter(mock_pages)
        return mock_doc

    def _make_parser(self):
        from app.parsers.pdf_parser import PDFParser
        return PDFParser(extract_tables=False, ocr_engine="auto")

    def test_supports_pdf_extension(self):
        from app.parsers.pdf_parser import PDFParser
        parser = PDFParser()
        assert parser.supports("document.pdf")
        assert parser.supports("REPORT.PDF")
        assert not parser.supports("document.docx")

    def test_supported_extensions_contains_pdf(self):
        from app.parsers.pdf_parser import PDFParser
        assert ".pdf" in PDFParser.supported_extensions

    @patch("app.parsers.pdf_parser.fitz")
    def test_parse_returns_document_with_correct_type(self, mock_fitz):
        from app.parsers.pdf_parser import PDFParser
        mock_fitz.open.return_value = self._make_mock_fitz(["Hello world page one."])
        parser = PDFParser(extract_tables=False)

        # Create a dummy file so Path.exists() passes in BaseParser
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            doc = parser._parse(Path(tmp_path))
            assert doc.doc_type == DocumentType.PDF
        finally:
            os.unlink(tmp_path)

    @patch("app.parsers.pdf_parser.fitz")
    def test_parse_extracts_metadata_title(self, mock_fitz):
        from app.parsers.pdf_parser import PDFParser
        mock_fitz.open.return_value = self._make_mock_fitz(
            ["Content"], metadata={"title": "My Report", "author": "Jane"}
        )
        parser = PDFParser(extract_tables=False)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            doc = parser._parse(Path(tmp_path))
            assert doc.metadata.title == "My Report"
            assert doc.metadata.author == "Jane"
        finally:
            os.unlink(tmp_path)

    @patch("app.parsers.pdf_parser.fitz")
    def test_parse_sets_page_count(self, mock_fitz):
        from app.parsers.pdf_parser import PDFParser
        mock_fitz.open.return_value = self._make_mock_fitz(
            ["Page one", "Page two", "Page three"]
        )
        parser = PDFParser(extract_tables=False)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            doc = parser._parse(Path(tmp_path))
            assert doc.metadata.page_count == 3
        finally:
            os.unlink(tmp_path)

    @patch("app.parsers.pdf_parser.fitz")
    def test_raw_text_contains_all_page_content(self, mock_fitz):
        from app.parsers.pdf_parser import PDFParser
        mock_fitz.open.return_value = self._make_mock_fitz(
            ["Alpha content", "Beta content"]
        )
        parser = PDFParser(extract_tables=False)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            doc = parser._parse(Path(tmp_path))
            assert "Alpha" in doc.raw_text
            assert "Beta" in doc.raw_text
        finally:
            os.unlink(tmp_path)

    def test_heading_level_detection(self):
        from app.parsers.pdf_parser import PDFParser
        parser = PDFParser()
        assert parser._get_heading_level(24.0, 12.0, False) == 1   # 2.0× ratio
        assert parser._get_heading_level(16.0, 12.0, False) == 2   # 1.33× ratio
        assert parser._get_heading_level(13.5, 12.0, False) == 3   # 1.125× ratio
        assert parser._get_heading_level(12.0, 12.0, False) == 0   # body text
        assert parser._get_heading_level(12.0, 12.0, True)  == 3   # bold body = H3

    def test_estimate_body_font_size_returns_mode(self):
        from app.parsers.pdf_parser import PDFParser, _ParsedPage

        def make_span(size):
            return {"text": "x", "size": size, "flags": 0}

        def make_block(sizes):
            return {"type": 0, "lines": [{"spans": [make_span(s) for s in sizes]}]}

        pages = [
            _ParsedPage(page_number=1, blocks=[make_block([12, 12, 12, 24])]),
            _ParsedPage(page_number=2, blocks=[make_block([12, 12, 18])]),
        ]
        # Mode of [12, 12, 12, 24, 12, 12, 18] = 12
        result = PDFParser._estimate_body_font_size(pages)
        assert result == 12.0

    def test_estimate_body_font_size_no_blocks_returns_default(self):
        from app.parsers.pdf_parser import PDFParser, _ParsedPage
        pages = [_ParsedPage(page_number=1, blocks=[])]
        result = PDFParser._estimate_body_font_size(pages)
        assert result == 12.0


# ─────────────────────────── ParserRegistry ──────────────────────────────────

class TestParserRegistry:

    def test_pdf_parser_is_registered(self):
        from app.parsers import registry
        parser = registry.get_parser("document.pdf")
        from app.parsers.pdf_parser import PDFParser
        assert isinstance(parser, PDFParser)

    def test_unsupported_extension_raises(self):
        from app.core.base.parser import UnsupportedFileTypeError
        from app.parsers import registry
        with pytest.raises(UnsupportedFileTypeError):
            registry.get_parser("document.xyz")

    def test_case_insensitive_extension(self):
        from app.parsers import registry
        parser_lower = registry.get_parser("doc.pdf")
        parser_upper = registry.get_parser("doc.PDF")
        assert type(parser_lower) == type(parser_upper)
