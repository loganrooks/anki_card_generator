"""MOBI/AZW/AZW3 document parser for Kindle formats.

Supports Amazon Kindle ebook formats:
- .mobi (Mobipocket)
- .azw (Amazon Kindle)
- .azw3 (KF8/Kindle Format 8)

Uses the mobi library to extract HTML content, then parses with BeautifulSoup.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from ..core.models import Document, Chunk
from ..core.exceptions import ParserError
from .base import DocumentParser

logger = logging.getLogger(__name__)


class MobiParser(DocumentParser):
    """Parser for MOBI/AZW/AZW3 Kindle documents."""

    def __init__(self):
        self._mobi = None

    def _get_mobi(self):
        """Lazy-load the mobi library."""
        if self._mobi is None:
            try:
                import mobi
                self._mobi = mobi
            except ImportError:
                raise ImportError(
                    "mobi package not installed. Install with: pip install mobi"
                )
        return self._mobi

    def parse(self, file_path: str) -> Document:
        """Parse a MOBI/AZW file into a Document.

        Args:
            file_path: Path to the MOBI/AZW file

        Returns:
            Document with extracted content
        """
        mobi = self._get_mobi()
        path = Path(file_path)

        if not path.exists():
            raise ParserError(f"File not found: {file_path}", file_path=file_path)

        logger.info(f"Parsing Kindle file: {path.name}")

        try:
            # Extract MOBI content to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path, extracted = mobi.extract(str(path))

                # Get metadata
                metadata = self._extract_metadata(extracted)
                title = metadata.get("title", path.stem)
                author = metadata.get("author", "Unknown")

                # Extract text content
                chunks = self._extract_chunks(extracted, title)

        except Exception as e:
            raise ParserError(
                f"Failed to parse Kindle file: {e}",
                file_path=file_path,
            )

        logger.info(f"Extracted {len(chunks)} sections from Kindle file")

        return Document(
            title=title,
            author=author,
            file_path=str(path.absolute()),
            chunks=chunks,
        )

    def _extract_metadata(self, extracted: dict) -> dict:
        """Extract metadata from MOBI structure."""
        metadata = {}

        try:
            # mobi.extract returns a dict with metadata
            if isinstance(extracted, dict):
                metadata["title"] = extracted.get("title", "")
                metadata["author"] = extracted.get("author", "")
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return metadata

    def _extract_chunks(self, extracted: dict, title: str) -> list[Chunk]:
        """Extract text chunks from MOBI content."""
        from bs4 import BeautifulSoup

        chunks = []

        try:
            # Get HTML content from extraction
            html_content = None
            if isinstance(extracted, dict):
                # Try to get the main content
                html_content = extracted.get("html", "")
                if not html_content and "content" in extracted:
                    html_content = extracted["content"]

            if not html_content:
                logger.warning("No HTML content found in Kindle file")
                return chunks

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Try to find chapters or sections
            sections = self._find_sections(soup)

            if sections:
                for idx, (section_title, content) in enumerate(sections):
                    text = self._clean_text(content)
                    if text and len(text) > 50:  # Skip very short sections
                        citation = f"{title}"
                        if section_title:
                            citation = f"{title}, \"{section_title}\""

                        chunks.append(Chunk(
                            text=text,
                            index=idx,
                            citation=citation,
                            section_title=section_title,
                        ))
            else:
                # Fall back to extracting all text
                text = self._clean_text(soup.get_text())
                if text:
                    chunks.append(Chunk(
                        text=text,
                        index=0,
                        citation=title,
                    ))

        except Exception as e:
            logger.warning(f"Error extracting chunks: {e}")

        return chunks

    def _find_sections(self, soup) -> list[tuple[str, str]]:
        """Find chapter/section divisions in the HTML."""
        sections = []

        # Try common chapter markers
        chapter_markers = [
            soup.find_all("h1"),
            soup.find_all("h2"),
            soup.find_all(class_=lambda c: c and "chapter" in c.lower() if c else False),
            soup.find_all(id=lambda i: i and "chapter" in i.lower() if i else False),
        ]

        for markers in chapter_markers:
            if markers and len(markers) > 1:
                for i, marker in enumerate(markers):
                    title = marker.get_text(strip=True)

                    # Get content until next marker
                    content_parts = []
                    for sibling in marker.find_next_siblings():
                        if sibling in markers:
                            break
                        content_parts.append(sibling.get_text(separator=" ", strip=True))

                    content = " ".join(content_parts)
                    if content:
                        sections.append((title, content))

                if sections:
                    return sections

        # Try to split by paragraphs if no chapters found
        paragraphs = soup.find_all("p")
        if len(paragraphs) > 10:
            # Group paragraphs into chunks of ~10
            chunk_size = 10
            for i in range(0, len(paragraphs), chunk_size):
                chunk_paragraphs = paragraphs[i:i + chunk_size]
                content = " ".join(p.get_text(strip=True) for p in chunk_paragraphs)
                if content:
                    sections.append((f"Section {i // chunk_size + 1}", content))

        return sections

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        import re

        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["mobi", "azw", "azw3"]
