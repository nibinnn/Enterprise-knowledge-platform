"""app/parsers/__init__.py — build the parser registry and expose it."""
from app.core.base.parser import ParserRegistry
from app.parsers.pdf_parser import PDFParser
from app.parsers.html_parser import HTMLParser
from app.parsers.text_parser import TextParser

registry = ParserRegistry.instance()
registry.register(PDFParser())
registry.register(HTMLParser())
registry.register(TextParser())
