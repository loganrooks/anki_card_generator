"""PDF document parser using PyMuPDF (fitz).

PyMuPDF is fast and handles most PDFs well, including:
- Text extraction with layout preservation
- Page-by-page processing
- Table of contents extraction
"""

import logging
from pathlib import Path
from typing import Optional

from ..core.models import Document, Chunk
from .base import DocumentParser

logger = logging.getLogger(__name__)


class PdfParser(DocumentParser):
    """Parser for PDF documents using PyMuPDF."""

    def __init__(self):
        self._fitz = None

    def _get_fitz(self):
        """Lazy-load PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz  # PyMuPDF
                self._fitz = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF not installed. Install with: pip install pymupdf"
                )
        return self._fitz

    def parse(self, file_path: str) -> Document:
        """Parse a PDF file into a Document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document with chunks (one per page or section)
        """
        fitz = self._get_fitz()
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Parsing PDF: {path.name}")

        doc = fitz.open(file_path)

        # Extract metadata
        metadata = doc.metadata or {}
        title = metadata.get("title") or path.stem
        author = metadata.get("author") or "Unknown"

        # Extract table of contents for section titles
        toc = doc.get_toc()  # List of [level, title, page]
        section_map = self._build_section_map(toc)

        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Skip empty pages
            if not text or not text.strip():
                continue

            # Clean up text
            text = self._clean_text(text)

            # Get section title if available
            section_title = section_map.get(page_num + 1)  # TOC uses 1-indexed pages

            # Create citation
            citation = f"{title}, p. {page_num + 1}"
            if section_title:
                citation = f"{title}, \"{section_title}\", p. {page_num + 1}"

            chunk = Chunk(
                text=text,
                index=len(chunks),
                citation=citation,
                start_page=page_num + 1,
                end_page=page_num + 1,
                section_title=section_title,
            )
            chunks.append(chunk)

        total_pages = len(doc)
        doc.close()

        logger.info(f"Extracted {len(chunks)} pages from PDF")

        return Document(
            title=title,
            author=author,
            file_path=str(path.absolute()),
            chunks=chunks,
            total_pages=total_pages,
        )

    def _build_section_map(self, toc: list) -> dict[int, str]:
        """Build a mapping from page number to section title.

        Args:
            toc: Table of contents from PyMuPDF [level, title, page]

        Returns:
            Dict mapping page numbers to the most recent section title
        """
        section_map = {}
        current_section = None

        # Sort by page number
        sorted_toc = sorted(toc, key=lambda x: x[2])

        for level, title, page in sorted_toc:
            # Only use top-level sections (level 1) or chapters
            if level <= 2:
                current_section = title
            section_map[page] = current_section

        return section_map

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline (paragraph break)
        import re

        # Fix common PDF extraction issues
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Fix single newlines in the middle of sentences
        text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)

        # Remove page headers/footers (lines with just numbers)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # Normalize whitespace
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["pdf"]
