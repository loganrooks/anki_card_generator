"""Tests for document parsers."""

import pytest
import tempfile
from pathlib import Path

from anki_forge.parsers import get_parser, EpubParser, PdfParser
from anki_forge.parsers.base import DocumentParser


class TestGetParser:
    """Tests for parser factory function."""

    def test_epub_extension(self):
        """Test that .epub files get EpubParser."""
        parser = get_parser("book.epub")
        assert isinstance(parser, EpubParser)

    def test_epub_uppercase(self):
        """Test case-insensitive extension matching."""
        parser = get_parser("book.EPUB")
        assert isinstance(parser, EpubParser)

    def test_pdf_extension(self):
        """Test that .pdf files get PdfParser."""
        parser = get_parser("document.pdf")
        assert isinstance(parser, PdfParser)

    def test_unsupported_extension(self):
        """Test that unsupported extensions raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_parser("file.docx")
        assert "Unsupported file type" in str(exc_info.value)
        assert ".docx" in str(exc_info.value)

    def test_full_path(self):
        """Test with full file path."""
        parser = get_parser("/home/user/documents/book.epub")
        assert isinstance(parser, EpubParser)


class TestEpubParser:
    """Tests for EPUB parser."""

    @pytest.fixture
    def parser(self):
        return EpubParser()

    def test_supported_extensions(self, parser):
        """Test supported file extensions."""
        extensions = parser.get_supported_extensions()
        assert "epub" in extensions

    def test_parse_nonexistent_file(self, parser):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.epub")

    def test_is_document_parser(self, parser):
        """Test that EpubParser is a DocumentParser."""
        assert isinstance(parser, DocumentParser)


class TestPdfParser:
    """Tests for PDF parser."""

    @pytest.fixture
    def parser(self):
        return PdfParser()

    def test_supported_extensions(self, parser):
        """Test supported file extensions."""
        extensions = parser.get_supported_extensions()
        assert "pdf" in extensions

    def test_parse_nonexistent_file(self, parser):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pdf")

    def test_is_document_parser(self, parser):
        """Test that PdfParser is a DocumentParser."""
        assert isinstance(parser, DocumentParser)

    def test_clean_text_hyphenation(self, parser):
        """Test that hyphenation at line breaks is fixed."""
        text = "phenom-\nenology"
        cleaned = parser._clean_text(text)
        assert "phenomenology" in cleaned

    def test_clean_text_multiple_newlines(self, parser):
        """Test that multiple newlines are collapsed."""
        text = "First paragraph.\n\n\n\nSecond paragraph."
        cleaned = parser._clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_clean_text_preserves_paragraphs(self, parser):
        """Test that paragraph breaks are preserved."""
        text = "First paragraph.\n\nSecond paragraph."
        cleaned = parser._clean_text(text)
        assert "First paragraph." in cleaned
        assert "Second paragraph." in cleaned


class TestDocumentParserBase:
    """Tests for base DocumentParser interface."""

    def test_abstract_methods(self):
        """Test that DocumentParser has required abstract methods."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            DocumentParser()

    def test_subclass_must_implement_parse(self):
        """Test that subclasses must implement parse()."""
        class IncompleteParser(DocumentParser):
            def get_supported_extensions(self):
                return ["test"]

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_subclass_must_implement_extensions(self):
        """Test that subclasses must implement get_supported_extensions()."""
        class IncompleteParser(DocumentParser):
            def parse(self, file_path):
                pass

        with pytest.raises(TypeError):
            IncompleteParser()


class TestParserIntegration:
    """Integration tests for parsers with real files."""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a simple test PDF if PyMuPDF is available."""
        try:
            import fitz
            pdf_path = tmp_path / "test.pdf"

            # Create a simple PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Test PDF content for Anki Forge.")
            page.insert_text((72, 100), "This is a second line with Dasein.")
            doc.save(str(pdf_path))
            doc.close()

            return pdf_path
        except ImportError:
            pytest.skip("PyMuPDF not installed")

    def test_pdf_parse_extracts_text(self, sample_pdf_path):
        """Test that PDF parser extracts text content."""
        parser = PdfParser()
        doc = parser.parse(str(sample_pdf_path))

        assert doc.title  # Has a title
        assert len(doc.chunks) > 0  # Has content

        # Check text was extracted
        all_text = " ".join(c.text for c in doc.chunks)
        assert "Test PDF" in all_text or "Dasein" in all_text

    def test_pdf_parse_creates_citations(self, sample_pdf_path):
        """Test that PDF parser creates page citations."""
        parser = PdfParser()
        doc = parser.parse(str(sample_pdf_path))

        if doc.chunks:
            # Citations should include page number
            assert "p. 1" in doc.chunks[0].citation or "p." in doc.chunks[0].citation
