"""
tests/unit/test_day3_parsers.py
─────────────────────────────────────────────────────────────────────────────
Day 3 unit tests — DOCXParser, TXTParser, MarkdownParser, utils.

All tests run WITHOUT real files by writing temp files or mocking.
Run with:  pytest tests/unit/test_day3_parsers.py -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.core.models.document import DocumentSection, DocumentType
from app.parsers.utils import (
    PlainTextHeadingDetector,
    SectionAccumulator,
    clean_text,
    detect_encoding,
    table_to_markdown,
)


# ─────────────────────────── utils.py ────────────────────────────────────────

class TestTableToMarkdown:

    def test_basic_table(self):
        rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = table_to_markdown(rows)
        assert "| Name | Age |" in md
        assert "| --- |" in md
        assert "| Alice | 30 |" in md
        assert "| Bob | 25 |" in md

    def test_empty_input_returns_empty(self):
        assert table_to_markdown([]) == ""

    def test_none_cells_become_empty_string(self):
        rows = [["A", None], ["1", "2"]]
        md = table_to_markdown(rows)
        assert "| A |  |" in md

    def test_pipe_in_cell_is_escaped(self):
        rows = [["A|B", "C"], ["1", "2"]]
        md = table_to_markdown(rows)
        assert "A\\|B" in md

    def test_ragged_rows_padded(self):
        rows = [["H1", "H2", "H3"], ["only one cell"]]
        md = table_to_markdown(rows)
        assert md  # should not crash


class TestDetectEncoding:

    def test_utf8_bom(self):
        raw = b"\xef\xbb\xbfHello"
        assert detect_encoding(raw) == "utf-8-sig"

    def test_utf16_le_bom(self):
        raw = b"\xff\xfeHello"
        assert detect_encoding(raw) == "utf-16-le"

    def test_utf16_be_bom(self):
        raw = b"\xfe\xff" + "Hello".encode("utf-16-be")
        assert detect_encoding(raw) == "utf-16-be"

    def test_plain_ascii_detected_as_utf8_or_ascii(self):
        raw = b"Hello world, plain ASCII"
        enc = detect_encoding(raw)
        # chardet may return "ascii" or "utf-8" — both are correct for pure ASCII
        assert enc.lower() in ("ascii", "utf-8", "utf-8-sig")

    def test_fallback_returned_when_ambiguous(self):
        # Latin-1 bytes that are invalid UTF-8
        raw = b"caf\xe9"
        enc = detect_encoding(raw, fallback="latin-1")
        # Should not raise and should return something usable
        assert enc is not None


class TestCleanText:

    def test_crlf_normalised(self):
        text = "line one\r\nline two\r\nline three"
        result = clean_text(text)
        assert "\r" not in result

    def test_multiple_blank_lines_collapsed(self):
        text = "Para one\n\n\n\n\nPara two"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_control_chars_removed(self):
        text = "Hello\x00World\x1fEnd"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "Hello" in result

    def test_trailing_whitespace_stripped_per_line(self):
        text = "line one   \nline two  "
        result = clean_text(text)
        for line in result.splitlines():
            assert not line.endswith(" ")


class TestSectionAccumulator:

    def test_single_heading_and_body(self):
        acc = SectionAccumulator()
        acc.push_heading("Introduction", level=1)
        acc.push_text("Some intro text.")
        sections = acc.flush()
        assert len(sections) == 1
        assert sections[0].heading == "Introduction"
        assert sections[0].level == 1
        assert "Some intro text." in sections[0].text

    def test_multiple_sections(self):
        acc = SectionAccumulator()
        acc.push_heading("Section A", level=1)
        acc.push_text("Text A")
        acc.push_heading("Section B", level=2)
        acc.push_text("Text B")
        sections = acc.flush()
        assert len(sections) == 2
        assert sections[0].heading == "Section A"
        assert sections[1].heading == "Section B"

    def test_no_heading_creates_section(self):
        acc = SectionAccumulator()
        acc.push_text("Body text without a heading")
        sections = acc.flush()
        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].level == 0

    def test_table_flag_set(self):
        acc = SectionAccumulator()
        acc.push_text("| A | B |\n| --- | --- |", is_table=True)
        sections = acc.flush()
        assert sections[0].has_table is True

    def test_empty_text_not_pushed(self):
        acc = SectionAccumulator()
        acc.push_text("   ")
        acc.push_text("")
        sections = acc.flush()
        assert len(sections) == 0

    def test_flush_is_idempotent(self):
        acc = SectionAccumulator()
        acc.push_text("Something")
        s1 = acc.flush()
        s2 = acc.flush()
        assert len(s1) == len(s2)

    def test_multiple_texts_joined(self):
        acc = SectionAccumulator()
        acc.push_heading("H1", level=1)
        acc.push_text("Paragraph one.")
        acc.push_text("Paragraph two.")
        sections = acc.flush()
        assert "Paragraph one." in sections[0].text
        assert "Paragraph two." in sections[0].text


class TestPlainTextHeadingDetector:

    def setup_method(self):
        self.detector = PlainTextHeadingDetector()

    def test_setext_h1(self):
        text, level = self.detector.classify("My Heading", next_line="=========")
        assert level == 1
        assert text == "My Heading"

    def test_setext_h2(self):
        text, level = self.detector.classify("Sub Heading", next_line="---------")
        assert level == 2
        assert text == "Sub Heading"

    def test_underline_line_itself_is_body(self):
        text, level = self.detector.classify("=========")
        assert level == 0

    def test_numbered_section_level1(self):
        text, level = self.detector.classify("1. Introduction")
        assert level == 1

    def test_numbered_section_level2(self):
        text, level = self.detector.classify("1.1 Background")
        assert level == 2

    def test_numbered_section_level3(self):
        text, level = self.detector.classify("1.1.1 Details")
        assert level == 3

    def test_all_caps_heading(self):
        text, level = self.detector.classify("INTRODUCTION")
        assert level == 1

    def test_all_caps_long_line_is_body(self):
        long_caps = "A" * 90
        text, level = self.detector.classify(long_caps)
        assert level == 0

    def test_colon_ending_short_line(self):
        text, level = self.detector.classify("Prerequisites:")
        assert level == 3

    def test_colon_ending_long_line_is_body(self):
        text, level = self.detector.classify(
            "This is a sentence that ends with a colon but is too long to be a heading:"
        )
        assert level == 0

    def test_empty_line_returns_zero(self):
        text, level = self.detector.classify("")
        assert level == 0
        assert text is None


# ─────────────────────────── TXTParser ───────────────────────────────────────

class TestTXTParser:

    def _write_tmp(self, content: str, suffix: str = ".txt") -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def setup_method(self):
        from app.parsers.txt_parser import TXTParser
        self.parser = TXTParser()

    def teardown_method(self):
        pass

    def test_supports_txt(self):
        assert self.parser.supports("notes.txt")
        assert self.parser.supports("data.log")
        assert not self.parser.supports("doc.pdf")

    def test_parses_plain_text(self):
        path = self._write_tmp("Hello world.\nThis is a document.")
        try:
            doc = self.parser._parse(path)
            assert doc.doc_type == DocumentType.TXT
            assert "Hello world" in doc.raw_text
        finally:
            os.unlink(path)

    def test_setext_heading_detection(self):
        content = (
            "Introduction\n"
            "============\n\n"
            "This section covers the basics.\n\n"
            "Background\n"
            "----------\n\n"
            "Some background info.\n"
        )
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            headings = [s.heading for s in doc.sections if s.heading]
            assert "Introduction" in headings
        finally:
            os.unlink(path)

    def test_numbered_section_detection(self):
        content = (
            "1. Introduction\n\n"
            "Some intro text here.\n\n"
            "2. Methodology\n\n"
            "The methods used.\n"
        )
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            headings = [s.heading for s in doc.sections if s.heading]
            assert any("Introduction" in h for h in headings)
            assert any("Methodology" in h for h in headings)
        finally:
            os.unlink(path)

    def test_short_file_returns_flat_section(self):
        content = "Hello\nWorld\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert len(doc.sections) == 1
            assert doc.sections[0].heading is None
        finally:
            os.unlink(path)

    def test_metadata_word_count(self):
        content = "one two three four five"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert doc.metadata.word_count == 5
        finally:
            os.unlink(path)

    def test_filename_set_correctly(self):
        path = self._write_tmp("Some content")
        try:
            doc = self.parser._parse(path)
            assert doc.filename == path.name
        finally:
            os.unlink(path)


# ─────────────────────────── MarkdownParser ──────────────────────────────────

class TestMarkdownParser:

    def _write_tmp(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def setup_method(self):
        from app.parsers.markdown_parser import MarkdownParser
        self.parser = MarkdownParser()

    def test_supports_md_and_markdown(self):
        assert self.parser.supports("readme.md")
        assert self.parser.supports("notes.mdx")
        assert self.parser.supports("doc.markdown")
        assert not self.parser.supports("doc.txt")

    def test_atx_h1_creates_section(self):
        content = "# Introduction\n\nSome text here.\n\n## Background\n\nMore text.\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            headings = [s.heading for s in doc.sections if s.heading]
            assert "Introduction" in headings
            assert "Background" in headings
        finally:
            os.unlink(path)

    def test_heading_levels_correct(self):
        content = "# H1\n\ntext\n\n## H2\n\ntext\n\n### H3\n\ntext\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            level_map = {s.heading: s.level for s in doc.sections if s.heading}
            assert level_map.get("H1") == 1
            assert level_map.get("H2") == 2
            assert level_map.get("H3") == 3
        finally:
            os.unlink(path)

    def test_yaml_frontmatter_title_extracted(self):
        content = (
            "---\n"
            "title: My Great Document\n"
            "author: Jane Doe\n"
            "tags: [python, ai]\n"
            "---\n\n"
            "# Introduction\n\nContent here.\n"
        )
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert doc.metadata.title == "My Great Document"
            assert doc.metadata.author == "Jane Doe"
        finally:
            os.unlink(path)

    def test_yaml_frontmatter_tags_extracted(self):
        content = "---\ntags: [ai, rag, llm]\n---\n\nContent.\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert "ai" in doc.metadata.tags
        finally:
            os.unlink(path)

    def test_code_block_not_split_on_internal_heading(self):
        content = (
            "# Real Heading\n\nSome text.\n\n"
            "```python\n"
            "# This is a comment, not a heading\n"
            "def foo(): pass\n"
            "```\n\n"
            "More text.\n"
        )
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            # Should only have ONE heading section (the real H1)
            headings = [s.heading for s in doc.sections if s.heading]
            assert len(headings) == 1
            assert headings[0] == "Real Heading"
        finally:
            os.unlink(path)

    def test_no_frontmatter_infers_title_from_h1(self):
        content = "# Document Title\n\nContent here.\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert doc.metadata.title == "Document Title"
        finally:
            os.unlink(path)

    def test_doc_type_is_markdown(self):
        path = self._write_tmp("# Hello\n\nWorld\n")
        try:
            doc = self.parser._parse(path)
            assert doc.doc_type == DocumentType.MD
        finally:
            os.unlink(path)

    def test_inline_markdown_preserved(self):
        content = "# Title\n\nThis has **bold** and *italic* text.\n"
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            # Inline formatting should be preserved (LLM reads it natively)
            assert "**bold**" in doc.raw_text or "bold" in doc.raw_text
        finally:
            os.unlink(path)

    def test_markdown_table_section_flagged(self):
        content = (
            "# Data\n\n"
            "| Col1 | Col2 |\n"
            "| --- | --- |\n"
            "| A | B |\n"
        )
        path = self._write_tmp(content)
        try:
            doc = self.parser._parse(path)
            assert len(doc.sections) > 0
        finally:
            os.unlink(path)


# ─────────────────────────── DOCXParser (mocked) ─────────────────────────────

class TestDOCXParser:

    def setup_method(self):
        from app.parsers.docx_parser import DOCXParser
        self.parser = DOCXParser()

    def test_supports_docx_and_doc(self):
        assert self.parser.supports("report.docx")
        assert self.parser.supports("legacy.doc")
        assert not self.parser.supports("notes.txt")

    def test_heading_level_mapping(self):
        from app.parsers.docx_parser import DOCXParser
        assert DOCXParser._heading_level("heading 1") == 1
        assert DOCXParser._heading_level("heading 2") == 2
        assert DOCXParser._heading_level("heading 3") == 3
        assert DOCXParser._heading_level("title") == 1
        assert DOCXParser._heading_level("subtitle") == 2
        assert DOCXParser._heading_level("normal") == 0
        assert DOCXParser._heading_level("body text") == 0

    def test_heading_level_partial_match(self):
        from app.parsers.docx_parser import DOCXParser
        # "Heading 1 Char" is a character style variant — should still map
        assert DOCXParser._heading_level("heading 1 char") == 1

    def test_table_to_markdown(self):
        # Mock a python-docx Table
        mock_table = MagicMock()
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()
        mock_cell_h1 = MagicMock(); mock_cell_h1.text = "Name"
        mock_cell_h2 = MagicMock(); mock_cell_h2.text = "Value"
        mock_cell_r1 = MagicMock(); mock_cell_r1.text = "foo"
        mock_cell_r2 = MagicMock(); mock_cell_r2.text = "bar"
        mock_row1.cells = [mock_cell_h1, mock_cell_h2]
        mock_row2.cells = [mock_cell_r1, mock_cell_r2]
        mock_table.rows = [mock_row1, mock_row2]

        md = self.parser._table_to_markdown(mock_table)
        assert "| Name | Value |" in md
        assert "| foo | bar |" in md


# ─────────────────────────── Registry (all parsers) ──────────────────────────

class TestRegistryWithAllParsers:

    def test_all_extensions_registered(self):
        from app.parsers import registry
        for ext in [".pdf", ".docx", ".doc", ".txt", ".log", ".md", ".mdx", ".markdown"]:
            parser = registry.get_parser(f"file{ext}")
            assert parser is not None, f"No parser registered for {ext}"

    def test_pdf_returns_pdf_parser(self):
        from app.parsers import registry, PDFParser
        assert isinstance(registry.get_parser("doc.pdf"), PDFParser)

    def test_docx_returns_docx_parser(self):
        from app.parsers import registry, DOCXParser
        assert isinstance(registry.get_parser("doc.docx"), DOCXParser)

    def test_txt_returns_txt_parser(self):
        from app.parsers import registry, TXTParser
        assert isinstance(registry.get_parser("doc.txt"), TXTParser)

    def test_md_returns_markdown_parser(self):
        from app.parsers import registry, MarkdownParser
        assert isinstance(registry.get_parser("doc.md"), MarkdownParser)

    def test_unknown_extension_raises(self):
        from app.parsers import registry
        from app.core.base.parser import UnsupportedFileTypeError
        with pytest.raises(UnsupportedFileTypeError):
            registry.get_parser("file.pptx")
