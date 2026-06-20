"""
app/parsers/html_parser.py
─────────────────────────────────────────────────────────────────────────────
HTML parser for wiki exports, internal knowledge bases, and web-saved pages.

Enterprise sources that produce HTML:
  - Confluence wiki exports       (.html)
  - SharePoint page saves
  - Notion HTML exports
  - Internal documentation sites

Extraction pipeline:
  1. BeautifulSoup parse          → DOM tree
  2. Noise removal                → nav, footer, script, style, ads, sidebars
  3. Metadata extraction          → <title>, <meta name="author">, Open Graph
  4. Heading-based section split  → h1-h6 tags → DocumentSection boundaries
  5. Table extraction             → <table> → Markdown
  6. Code block preservation      → <pre><code> → fenced markdown
  7. Link text extraction         → href kept as [text](url) for context
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from app.core.base.parser import BaseParser
from app.core.models.document import (
    Document, DocumentMetadata, DocumentSection, DocumentType,
)
from app.parsers.utils import SectionAccumulator, clean_text, read_text_file

logger = logging.getLogger(__name__)

# HTML tags to strip entirely (with their content)
_NOISE_TAGS = {
    "script", "style", "nav", "header", "footer",
    "aside", "advertisement", "noscript", "iframe",
    "form", "button", "input", "select", "option",
}

# Tags that signal block-level elements to separate with newlines
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main",
    "li", "dt", "dd", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "br", "hr",
}

# Heading tag → level integer
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class HTMLParser(BaseParser):
    """
    Parser for HTML files and wiki exports.
    Uses BeautifulSoup4 for DOM parsing.
    """

    supported_extensions: frozenset[str] = frozenset({
        ".html", ".htm", ".xhtml",
    })

    def __init__(self, preserve_links: bool = False):
        """
        Args:
            preserve_links: If True, keep hyperlinks as [text](url).
                            Default False — only the anchor text is kept.
        """
        self._preserve_links = preserve_links

    # ── BaseParser interface ───────────────────────────────────────────────────

    def _parse(self, path: Path) -> Document:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise RuntimeError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install with: pip install beautifulsoup4 lxml"
            )

        raw_html = read_text_file(path)
        soup = BeautifulSoup(raw_html, "lxml")

        metadata = self._build_metadata(soup, path)
        sections = self._build_sections(soup)

        raw_text = "\n\n".join(s.text for s in sections if s.text.strip())
        raw_text = clean_text(raw_text)
        metadata.word_count = len(raw_text.split())

        return Document(
            filename=path.name,
            doc_type=DocumentType.HTML,
            raw_text=raw_text,
            sections=sections,
            metadata=metadata,
        )

    def extract_metadata(self, path: Path) -> dict:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {}
        soup = BeautifulSoup(read_text_file(path), "lxml")
        return self._extract_meta_tags(soup)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _build_metadata(self, soup, path: Path) -> DocumentMetadata:
        meta = self._extract_meta_tags(soup)

        # <title> tag
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else None

        title = (
            meta.get("og:title")
            or meta.get("title")
            or page_title
            or path.stem.replace("_", " ").replace("-", " ").title()
        )

        return DocumentMetadata(
            title=title,
            author=meta.get("author") or meta.get("og:site_name"),
            created_date=meta.get("article:published_time") or meta.get("date"),
            modified_date=meta.get("article:modified_time") or meta.get("last-modified"),
            extra={"url": meta.get("og:url"), "description": meta.get("description")},
        )

    @staticmethod
    def _extract_meta_tags(soup) -> dict:
        """Extract all <meta> tags into a flat dict."""
        meta: dict = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property") or tag.get("http-equiv")
            content = tag.get("content")
            if name and content:
                meta[name.lower()] = content.strip()
        return meta

    # ── Section building ──────────────────────────────────────────────────────

    def _build_sections(self, soup) -> List[DocumentSection]:
        """
        Remove noise, then walk the document tree top-to-bottom.
        Each h1-h6 tag starts a new DocumentSection.
        """
        # 1. Remove all noise tags
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # 2. Find the main content area (if identifiable)
        body = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"content|main|article", re.I))
            or soup.find(class_=re.compile(r"content|main|article|wiki", re.I))
            or soup.find("body")
            or soup
        )

        acc = SectionAccumulator()
        self._walk(body, acc)
        sections = acc.flush()

        if not sections:
            text = clean_text(body.get_text(separator="\n"))
            return [DocumentSection(heading=None, level=0, text=text)]

        return sections

    def _walk(self, element, acc: SectionAccumulator) -> None:
        """
        Recursively walk the DOM, pushing headings and text into the accumulator.
        Tables and <pre> blocks are handled atomically.
        """
        from bs4 import NavigableString, Tag

        for child in element.children:

            if isinstance(child, NavigableString):
                text = clean_text(str(child))
                if text.strip():
                    acc.push_text(text)
                continue

            if not isinstance(child, Tag):
                continue

            tag_name = child.name.lower() if child.name else ""

            # ── Heading ────────────────────────────────────────────────────────
            if tag_name in _HEADING_LEVELS:
                heading_text = clean_text(child.get_text(separator=" "))
                if heading_text.strip():
                    acc.push_heading(heading_text, level=_HEADING_LEVELS[tag_name])
                continue

            # ── Table → Markdown ───────────────────────────────────────────────
            if tag_name == "table":
                md = self._table_to_markdown(child)
                if md:
                    acc.push_text(md, is_table=True)
                continue

            # ── Pre / Code block ───────────────────────────────────────────────
            if tag_name in ("pre", "code"):
                code_text = child.get_text()
                if code_text.strip():
                    acc.push_text(f"```\n{code_text.strip()}\n```")
                continue

            # ── Image ──────────────────────────────────────────────────────────
            if tag_name == "img":
                alt = child.get("alt", "").strip()
                if alt:
                    acc.push_text(f"[Image: {alt}]", is_image=True)
                continue

            # ── Anchor (hyperlink) ─────────────────────────────────────────────
            if tag_name == "a" and self._preserve_links:
                href = child.get("href", "")
                link_text = clean_text(child.get_text(separator=" ")).strip()
                if link_text and href:
                    acc.push_text(f"[{link_text}]({href})")
                    continue
                # Fall through to recurse if no href

            # ── Everything else — recurse ──────────────────────────────────────
            self._walk(child, acc)

    # ── Table extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _table_to_markdown(table_tag) -> str:
        """Convert an HTML <table> element to a Markdown table string."""
        from app.parsers.utils import table_to_markdown

        rows: List[List[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                text = " ".join(cell.get_text(separator=" ").split())
                cells.append(text)
            if any(cells):
                rows.append(cells)

        return table_to_markdown(rows)
