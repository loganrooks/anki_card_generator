"""EPUB document parser."""

import logging
import re
from typing import Optional
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from .base import DocumentParser, ParserConfig
from ..core.models import Document, Chunk
from ..core.exceptions import ParserError

logger = logging.getLogger(__name__)


# Default selectors for common EPUB structures
DEFAULT_FOOTNOTE_SELECTOR = {"name": "div", "class_": "footnotesection"}
DEFAULT_TOC_SELECTOR = {"name": "div", "class_": "toc"}
DEFAULT_COPYRIGHT_SELECTOR = {"name": "div", "class_": "copyrightpage"}
DEFAULT_HEADER_SELECTORS = [
    {"name": "h1"},
    {"name": "h2"},
    {"name": "h3"},
    {"name": "h4"},
]
DEFAULT_PARAGRAPH_SELECTORS = [
    {"name": "p"},
    {"name": "div", "class_": re.compile(r"^(para|para1|paragraph)$", re.I)},
]


class EpubParser(DocumentParser):
    """Parser for EPUB documents."""

    def get_supported_extensions(self) -> list[str]:
        return ["epub"]

    def parse(self, file_path: str, config: Optional[ParserConfig] = None) -> Document:
        """Parse an EPUB file and extract structured content.

        Args:
            file_path: Path to the EPUB file
            config: Optional parser configuration

        Returns:
            Document with extracted text and metadata
        """
        config = config or ParserConfig()

        if not Path(file_path).exists():
            raise ParserError(f"File not found: {file_path}", file_path=file_path)

        try:
            book = epub.read_epub(file_path)
        except Exception as e:
            raise ParserError(f"Failed to read EPUB: {e}", file_path=file_path)

        # Extract metadata
        title = self._get_metadata(book, "title") or Path(file_path).stem
        author = self._get_metadata(book, "creator") or "Unknown"

        # Get document items (pages)
        items = [
            item for item in book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        ]

        # Apply page range
        start = config.start_page
        end = config.end_page if config.end_page else len(items)
        items = items[start:end]

        # Extract text from each page
        sections = []
        for page_num, item in enumerate(items, start=start):
            try:
                content = item.get_body_content()
                soup = BeautifulSoup(content, "html.parser")

                # Remove unwanted elements
                if config.remove_footnotes:
                    self._remove_elements(soup, config.footnote_selector or DEFAULT_FOOTNOTE_SELECTOR)
                if config.remove_toc:
                    self._remove_elements(soup, config.toc_selector or DEFAULT_TOC_SELECTOR)
                if config.remove_copyright:
                    self._remove_elements(soup, config.copyright_selector or DEFAULT_COPYRIGHT_SELECTOR)

                # Extract hierarchical structure
                section = self._extract_section(soup, page_num, config)
                if section["paragraphs"]:
                    sections.append(section)

            except Exception as e:
                logger.warning(f"Error parsing page {page_num}: {e}")
                continue

        # Convert sections to chunks (one chunk per section for now)
        # Chunking strategy will handle further splitting
        chunks = []
        for idx, section in enumerate(sections):
            text = "\n\n".join(section["paragraphs"])
            if text.strip():
                chunks.append(Chunk(
                    text=text,
                    index=idx,
                    citation=self._format_citation(title, section),
                    start_page=section.get("page_num"),
                    section_title=section.get("title"),
                    paragraph_indices=list(range(len(section["paragraphs"]))),
                ))

        return Document(
            title=title,
            author=author,
            file_path=file_path,
            chunks=chunks,
            total_pages=len(items),
        )

    def _get_metadata(self, book: epub.EpubBook, field: str) -> Optional[str]:
        """Extract metadata field from EPUB."""
        try:
            metadata = book.get_metadata("DC", field)
            if metadata:
                return metadata[0][0]
        except (IndexError, TypeError):
            pass
        return None

    def _remove_elements(self, soup: BeautifulSoup, selector: dict) -> None:
        """Remove elements matching selector from soup."""
        try:
            for element in soup.find_all(**selector):
                element.decompose()
        except Exception as e:
            logger.debug(f"Error removing elements with selector {selector}: {e}")

    def _extract_section(
        self,
        soup: BeautifulSoup,
        page_num: int,
        config: ParserConfig
    ) -> dict:
        """Extract a section with headers and paragraphs."""
        section = {
            "page_num": page_num,
            "title": None,
            "headers": [],
            "paragraphs": [],
        }

        # Find headers
        header_selectors = config.header_selectors or DEFAULT_HEADER_SELECTORS
        for selector in header_selectors:
            for header in soup.find_all(**selector):
                text = header.get_text(strip=True)
                if text:
                    if not section["title"]:
                        section["title"] = text
                    section["headers"].append(text)

        # Find paragraphs
        paragraph_selectors = config.paragraph_selectors or DEFAULT_PARAGRAPH_SELECTORS
        seen_text = set()  # Avoid duplicates

        for selector in paragraph_selectors:
            for para in soup.find_all(**selector):
                text = para.get_text(strip=True)
                if text and text not in seen_text:
                    seen_text.add(text)
                    section["paragraphs"].append(text)

        # Fallback: get all text if no paragraphs found
        if not section["paragraphs"]:
            text = soup.get_text(separator="\n", strip=True)
            if text:
                # Split by double newlines to get paragraph-like chunks
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                section["paragraphs"] = paragraphs

        return section

    def _format_citation(self, title: str, section: dict) -> str:
        """Format citation for a section."""
        parts = [title]

        if section.get("title"):
            parts.append(section["title"])
        elif section.get("headers"):
            parts.append(section["headers"][0])

        if section.get("page_num") is not None:
            parts.append(f"p. {section['page_num'] + 1}")

        return f"[{': '.join(parts)}]"


def extract_text_simple(epub_path: str) -> tuple[str, str, list[str]]:
    """Simple extraction for backward compatibility.

    Returns:
        tuple: (author, title, list of text chunks)
    """
    parser = EpubParser()
    doc = parser.parse(epub_path)
    texts = [chunk.text for chunk in doc.chunks]
    return doc.author, doc.title, texts
